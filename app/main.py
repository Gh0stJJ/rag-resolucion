#main.py
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
import requests
from typing import Optional, Dict, Any
from ingest import ingest_folder, reset_collection
from rag import retrieve, format_citations, generate_answer
from settings import MAX_ANSWER_CHUNKS, TOP_K, K_LEX, RERANK_TOP, USE_HYDE, USE_MULTIQUERY, MULTIQUERY_N,FILTER_AGENT_URL
from health import full_health
import json, time
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime, timezone
from pathlib import Path

ASK_TOTAL = 0
ASK_EMPTY = 0


app = FastAPI(title="RAG Resoluciones CU", version="0.1.0")


# enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Add specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Logging setup
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
NDJSON_PATH = LOG_DIR / "events.ndjson"

def append_ndjson(data: Dict[str, Any]):
    with NDJSON_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")



 # Pydantic models
class AskRequest(BaseModel):
    query: str


class FeedbackPayload(BaseModel):
    trace_id: str = Field(..., description="Trace ID given by the /ask endpoint")
    session_id: str = Field(..., description="Session ID of the user")
    prompt: str
    feedback: str = Field(..., pattern="^(like|dislike)$", description="Feedback type: 'like' or 'dislike'")
    backend_response: dict
    extra : Optional[dict] = None # Additional comments from the user

@app.post("/feedback")
def feedback(evt: FeedbackPayload):
    record = {
        "event": "feedback",
        "ts": datetime.now(timezone.utc).isoformat(),
        **evt.model_dump(),
    }
    append_ndjson(record)
    return {"status": "feedback recorded!"}


@app.post("/ingest")
def ingest(path: str = "/data/json"):
    try:
        stats = ingest_folder(path)
        return {"status": "ok", **stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask")
def ask(payload: AskRequest):
    global ASK_TOTAL, ASK_EMPTY
    t0 = time.time()
    trace_id = str(uuid4())

    # === Filtros desde el Agente ===
    # Llamar al agente para obtener los filtros
    try:
        agent_response = requests.post(FILTER_AGENT_URL, json={"promt": payload.query})
        agent_response.raise_for_status() #comprueba errores HTTP
        extracted_data = agent_response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error calling agent service: {str(e)}")
    # Contruir los filtros
    # ID resolución
    filtros = {}
    if extracted_data.get("id_resolucion"):
        filtros["id_reso"] = extracted_data["id_resolucion"]
    # Rango de fechas
    rango_fechas = extracted_data.get("rango_fechas")
    if rango_fechas and rango_fechas.get("fecha_inicio") and rango_fechas.get("fecha_fin"):
        filtros["date_from"] = rango_fechas["fecha_inicio"]
        filtros["date_to"] = rango_fechas["fecha_fin"]
    # Temas Principales
    temas = extracted_data.get("temas_principales")
    if temas:
        filtros["temas_principales"] = list(temas)
    # Nombres mencionados
    nombres = extracted_data.get("nombres_involucrados")
    if nombres:
        filtros["nombres"] = list(nombres)
    # Referencias a artículos y otros documentos
    referencias = extracted_data.get("numeros_referencia")
    if referencias and referencias.get("articulos"):
        filtros["articulos"] = [art.lower() for art in referencias["articulos"]]
    if referencias and referencias.get("otros_doc"):
        filtros["otros_doc"] = list(referencias["otros_doc"])
    # tipo de resolución
    if extracted_data.get("tipo_session"):
        filtros["tipo_sesion"] = extracted_data["tipo_session"]

    results, extra_results = retrieve(payload.query, filtros)
    citations = format_citations(results)
    answer = generate_answer(payload.query, results)
    used_docs = list({c["id_reso"] for c in citations})
    elapsed = int((time.time() - t0) * 1000)
    n = len(results)
    avg_dist = sum(r.distance for r in results) / n if n > 0 else 0.0
    min_dist = min((r.distance for r in results), default=0.0)
    max_dist = max((r.distance for r in results), default=0.0)

    wanted_id = filtros.get("id_reso")
    hres_doc_ids = [ (r.metadata or {}).get("id_reso") for r in results ]
    hit_at_k = (wanted_id in hres_doc_ids) if wanted_id else None
    rank_of_wanted = (hres_doc_ids.index(wanted_id) + 1) if (wanted_id and wanted_id in hres_doc_ids) else None


    ASK_TOTAL += 1

    empty = (n == 0)
    if empty:
        ASK_EMPTY += 1

    metrics = {
        "top_k": TOP_K,
        "k_lex": K_LEX,
        "rerank_top": RERANK_TOP,
        "max_answer_chunks": MAX_ANSWER_CHUNKS,
        "returned_chunks": n,
        "avg_distance": avg_dist,
        "min_distance": min_dist,
        "max_distance": max_dist,
        "empty": empty,
        "elapsed_ms": elapsed,
        "empty_context_rate": round(ASK_EMPTY / ASK_TOTAL, 3),
        "hit_at_k": hit_at_k,
        "wanted_id": wanted_id,
        "rank_of_wanted": rank_of_wanted,
        "hyde_doc": extra_results.get("hyde") if USE_HYDE else None,
        "subqueries": extra_results.get("subqueries") if USE_MULTIQUERY else None,
    }

    print(json.dumps({"event": "ask_metrics", **metrics}, ensure_ascii=False))

    append_ndjson({
        "event": "ask",
        "trace_id": trace_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "request": {"query": payload.query},
        "response_fragment": {
            "used_docs": used_docs,
            "chunks_used": min(len(results), MAX_ANSWER_CHUNKS),
            "metrics": metrics,
            "filters_query": filtros
        },
    })

    return {
        "trace_id": trace_id,
        "answer": answer,
        "citations": citations,
        "used_docs": used_docs,
        "chunks_used": min(len(results), MAX_ANSWER_CHUNKS),
        "metrics": metrics,
        "filters_query": filtros
    }

# Reset the ingestion data (for testing purposes)
@app.post("/reset")
def reset_data():
    stats = reset_collection()
    return {"status": "reset done", **stats}

# Health check endpoint

@app.get("/health")
def health_check():
    return full_health()