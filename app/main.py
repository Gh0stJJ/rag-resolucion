from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
from ingest import ingest_folder, reset_collection
from rag import retrieve, format_citations, generate_answer
from settings import MAX_ANSWER_CHUNKS
from health import full_health


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
    results = retrieve(payload.query, payload.filtros)
    citations = format_citations(results)
    answer = generate_answer(payload.query, results)
    used_docs = list({c["id_reso"] for c in citations})
    return {
        "answer": answer,
        "citations": citations,
        "used_docs": used_docs,
        "chunks_used": min(len(results), MAX_ANSWER_CHUNKS)
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