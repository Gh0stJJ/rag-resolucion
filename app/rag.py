from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from settings import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBED_MODEL,
    TOP_K, MAX_ANSWER_CHUNKS, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY,
    CHROMA_TENANT, CHROMA_DATABASE, TEMPERATURE, MAX_TOKENS
)
from pydantic import BaseModel
import numpy as np
from embeddins import embed_query
from openai import OpenAI


class RetrievalResult(BaseModel):
    id: str
    texto: str
    metadata: Dict[str, Any]
    distance: float

def build_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(allow_reset=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

def build_client_collection():
    client = build_client()
    try:
        coll = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        # Mensaje más claro + listar colecciones disponibles
        available = [c.name for c in client.list_collections()]
        raise RuntimeError(
            f"La colección '{CHROMA_COLLECTION}' no existe. Colecciones disponibles: {available}. "
            "Ejecuta primero /ingest o crea la colección durante el ingest."
        )
    return client, coll

def get_or_create_coll(client):
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return client.get_or_create_collection(CHROMA_COLLECTION)

  
  

  


_model = None
def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _mmr(doc_vectors, query_vec, lambda_mult=0.5, top_n = 10):
    """
    Maximal Marginal Relevance (MMR) for retrieval re-ranking.
    doc_vectors: list of document vectors (numpy arrays)
    query_vec: query vector (numpy array)
    lambda_mult: trade-off between relevance and diversity (0 <= lambda_mult <= 1)
    top_n: number of documents to select

    Returns indices of selected documents.
    """

    D = np.array(doc_vectors)
    q = np.array(query_vec)

    #simi  query - documentos
    qsim = D @ q 
    selected, candidates = [], list(range(len(D)))
    while len(selected) < min(top_n, len(D)):
        if not selected:
            i = int(np.argmax(qsim))
            selected.append(i); candidates.remove(i); continue
        # min sim btn selected

        Dsel = D[selected]
        red = D @ Dsel.T # doc - doc sim

        max_red = red.max(axis=1)
        score = lambda_mult * qsim - (1 - lambda_mult) * max_red
        score[list(selected)] = -np.inf
        i = int(np.argmax(score))
        selected.append(i); candidates.discard(i)
    return selected


def retrieve(query: str, filtros: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
    _, coll = build_client_collection()
    qvec = embed_query(query, normalize=True) 

    conditions = []
    if filtros:
        if "anio" in filtros and filtros["anio"]:
            conditions.append({"anio": {"$eq": int(filtros["anio"])}})
        if "tipo" in filtros and filtros["tipo"]:
            conditions.append({"tipo": {"$eq": filtros["tipo"]}})
        if "id_reso" in filtros and filtros["id_reso"]:
            conditions.append({"id_reso": {"$eq": filtros["id_reso"]}})

    if not conditions:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}


    # Retrieve more initial results for MMR reranking
    initial_k = min(TOP_K * 2, 100)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=initial_k,
        where=where or None,
        include=["embeddings", "documents", "metadatas", "distances"]
    )

    # Apply MMR reranking
    doc_vectors = np.array(res["embeddings"][0])
    mmr_indices = _mmr(doc_vectors, qvec, lambda_mult=0.5, top_n=TOP_K)
    
    # Create RetrievalResult objects for reranked results
    out = []
    for idx in mmr_indices:
        out.append(RetrievalResult(
            id=res["ids"][0][idx],
            texto=res["documents"][0][idx],
            metadata=res["metadatas"][0][idx],
            distance=res["distances"][0][idx],
        ))
    return out

def format_citations(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    cites = []
    seen = set()
    for r in results[:MAX_ANSWER_CHUNKS]:
        meta = r.metadata
        key = (meta.get("id_reso"), meta.get("seccion"), meta.get("parrafo_index"), meta.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        fragment = (r.texto[:200] + "…") if len(r.texto) > 200 else r.texto
        cites.append({
            "id_reso": meta.get("id_reso"),
            "seccion": meta.get("seccion"),
            "parrafo_index": meta.get("parrafo_index"),
            "chunk_index": meta.get("chunk_index"),
            "anio": meta.get("anio"),
            "fecha": meta.get("fecha_legible") or meta.get("fecha_iso"),
            "tipo": meta.get("tipo"),
            "extracto": fragment
        })
    return cites

def _lmstudio_chat(messages: list, temperature: float = 0.2, max_tokens: int = 400) -> str:
    """
    Llama al servidor OpenAI-compatible de LM Studio.
    """
    if not LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL no está configurado. Define LLM_BASE_URL para usar LM Studio.")
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def generate_answer(query: str, contexts: List[RetrievalResult]) -> str:
   
    #sources
    items= []
    for c in contexts[:MAX_ANSWER_CHUNKS]:
        m = c.metadata
        items.append({
            "id_reso": m.get("id_reso"),
            "seccion": m.get("seccion"),
            "fecha": m.get("fecha_legible") or m.get("fecha_iso"),
            "texto": c.texto
        })
    
    #sources legibles
    src_lines = []
    for it in items:
        head= f"({it['id_reso']} - {it['seccion']} - {it['fecha']})"
        src_lines.append(f"{head}\n{it['texto']}\n")
    sources = "\n\n---\n\n".join(src_lines) if src_lines else "No hay fuentes disponibles."

    system_msg = (
        "Eres un asistente que responde preguntas sobre resoluciones del Consejo Universitario en español. "
        "Prioriza la información de las FUENTES. "
        "Si hay evidencia suficiente: responde conciso, claro y específico. "
        "Si la evidencia es parcial: responde lo que conste y señala explícitamente qué falta. "
        "Si no hay evidencia: dilo con claridad y sugiere qué buscar. "
        "Al final, si se usaron FUENTES, incluye una línea 'Citas: ' con los pares (id_reso; seccion; fecha) usados. "
        "No inventes citas. No cites si no hay fuentes."
    )

    user_msg = (
        f"Pregunta: {query}\n\n"
        f"FUENTES:\n{sources}"
    )

    return _lmstudio_chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ], temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    # Baseline sin LLM 
    joined = "\n\n".join([f"- {c.metadata.get('id_reso')} [{c.metadata.get('seccion')}]: {c.texto}" for c in contexts[:MAX_ANSWER_CHUNKS]])
    return f"Contexto relevante encontrado (modo local sin LLM):\n\n{joined}\n\nSugerencia: configura LLM_BASE_URL para respuestas redactadas."
