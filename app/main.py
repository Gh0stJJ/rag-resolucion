#main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
from ingest import ingest_folder, reset_collection
from rag import retrieve, format_citations, generate_answer
from settings import MAX_ANSWER_CHUNKS, TOP_K, K_LEX, RERANK_TOP
from health import full_health
import json, time

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
    results = retrieve(payload.query, payload.filtros)
    citations = format_citations(results)
    answer = generate_answer(payload.query, results)
    used_docs = list({c["id_reso"] for c in citations})
    elapsed = int((time.time() - t0) * 1000)
    n = len(results)
    avg_dist = sum(r.distance for r in results) / n if n > 0 else 0.0
    min_dist = min((r.distance for r in results), default=0.0)
    max_dist = max((r.distance for r in results), default=0.0)
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
        "empty_context_rate": round(ASK_EMPTY / ASK_TOTAL, 3)
    }

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