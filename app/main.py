#main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
from ingest import ingest_folder, reset_collection
from rag import retrieve, format_citations, generate_answer
from settings import MAX_ANSWER_CHUNKS, TOP_K, K_LEX, RERANK_TOP
from health import full_health
import json, time
from utils import extract_id_reso, extract_month_range

ASK_TOTAL = 0
ASK_EMPTY = 0


app = FastAPI(title="RAG Resoluciones CU", version="0.1.0")


class AskRequest(BaseModel):
    query: str
    filtros: Optional[Dict[str, Any]] = None

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

    filtros = payload.filtros or {}
    wanted_id = extract_id_reso(payload.query) # get id_reso if mentioned in the query
    if wanted_id and not filtros.get("id_reso"):
        filtros["id_reso"] = wanted_id

    mr = extract_month_range(payload.query)
    if mr and not filtros.get("date_from_yyyymmdd") and not filtros.get("date_to_yyyymmdd"):
        y_from, y_to, anio, mes = mr
        filtros["anio"] = anio
        filtros["mes"] = mes
        filtros["date_from_yyyymmdd"] = y_from
        filtros["date_to_yyyymmdd"] = y_to
    
    results = retrieve(payload.query, filtros)
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
    }

    # add date metrics if any
    if mr:
        y_from, y_to, anio, mes = mr
        metrics.update({
            "filter_anio": anio,
            "filter_mes": mes,
            "filter_date_from_yyyymmdd": y_from,
            "filter_date_to_yyyymmdd": y_to,
        })

    print(json.dumps({"event": "ask_metrics", **metrics}, ensure_ascii=False))


    return {
        "answer": answer,
        "citations": citations,
        "used_docs": used_docs,
        "chunks_used": min(len(results), MAX_ANSWER_CHUNKS),
        "metrics": metrics
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