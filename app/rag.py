# rag.py
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
from collections import defaultdict

from settings import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION,
    TOP_K, MAX_ANSWER_CHUNKS, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY,
    CHROMA_TENANT, CHROMA_DATABASE, TEMPERATURE, MAX_TOKENS,
    K_LEX, RERANK_TOP, RRF_K, BM25_INDEX_DIR
)
from embeddins import embed_query
from lex import open_or_create as bm25_open_or_create, search as bm25_search


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
        available = [c.name for c in client.list_collections()]
        raise RuntimeError(
            f"La colección '{CHROMA_COLLECTION}' no existe. Colecciones disponibles: {available}. "
            "Ejecuta primero /ingest o crea la colección durante el ingest."
        )
    return client, coll


def _build_where(filtros: Dict[str, Any]) -> Dict[str, Any]:
    conditions = []
    if "anio" in filtros and filtros["anio"]:
        conditions.append({"anio": {"$eq": int(filtros["anio"])}})
    if "tipo" in filtros and filtros["tipo"]:
        conditions.append({"tipo": {"$eq": filtros["tipo"]}})
    if "id_reso" in filtros and filtros["id_reso"]:
        conditions.append({"id_reso": {"$eq": filtros["id_reso"]}})
    if not conditions:
        return {}
    return {"$and": conditions} if len(conditions) > 1 else conditions[0]


def _rrf_fuse(dense_ids: List[str], lex_ids: List[str], k: int = RRF_K) -> List[str]:
    """
    Reciprocal Rank Fusion para combinar listas de IDs (dense y lex).
    """
    scores = defaultdict(float)
    for rank, cid in enumerate(dense_ids):
        scores[cid] += 1.0 / (k + rank + 1)
    for rank, cid in enumerate(lex_ids):
        scores[cid] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked]


def _mmr(doc_vectors: List[List[float]], query_vec: List[float],
         lambda_mult: float = 0.5, top_n: int = 10) -> List[int]:
    """
    Maximal Marginal Relevance 
    """
    D = np.array(doc_vectors, dtype=float)
    q = np.array(query_vec, dtype=float)

    # similitud query-doc 
    qsim = D @ q

    selected: List[int] = []
    while len(selected) < min(top_n, len(D)):
        if not selected:
            i = int(np.argmax(qsim))
            selected.append(i)
            continue
        red = D @ D[selected].T      # doc-doc sim
        max_red = red.max(axis=1)
        score = lambda_mult * qsim - (1 - lambda_mult) * max_red
        for s in selected:
            score[s] = -1e9
        i = int(np.argmax(score))
        selected.append(i)
    return selected


def retrieve(query: str, filtros: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
    _, coll = build_client_collection()
    qvec = embed_query(query, normalize=True)

    res_dense = coll.query(
        query_embeddings=[qvec],
        n_results=TOP_K,
        where=_build_where(filtros) if filtros else None,
        include=["ids"]  
    )
    dense_ids = res_dense.get("ids", [[]])[0] if res_dense.get("ids") else []

    # lexico (BM25) – best-effort
    try:
        ix = bm25_open_or_create(BM25_INDEX_DIR)
        lex_hits = bm25_search(ix, query, filtros=filtros, limit=K_LEX)
        lex_ids = [cid for cid, _ in lex_hits]
    except Exception as e:
        print(f"[BM25] Error: {repr(e)}")
        lex_ids = []

    # Fusion rrf
    fused_ids = _rrf_fuse(dense_ids, lex_ids, k=RRF_K)
    fused_ids = fused_ids[:max(RERANK_TOP, MAX_ANSWER_CHUNKS)]
    if not fused_ids:
        return []

    # get payloads by ID
    got = coll.get(ids=fused_ids, include=["documents", "metadatas", "embeddings"])
    id_to_payload: Dict[str, Dict[str, Any]] = {}
    for i, cid in enumerate(got.get("ids", [])):
        id_to_payload[cid] = {
            "texto": got["documents"][i] if got.get("documents") else "",
            "meta": got["metadatas"][i] if got.get("metadatas") else {},
            "emb": got["embeddings"][i] if got.get("embeddings") else None
        }

    # final re-ranking MMR
    D: List[List[float]] = []
    id_list: List[str] = []
    for cid in fused_ids:
        p = id_to_payload.get(cid)
        if p and p["emb"] is not None:
            D.append(p["emb"])
            id_list.append(cid)

    if D:
        idxs = _mmr(D, qvec, lambda_mult=0.5, top_n=MAX_ANSWER_CHUNKS)
        ordered_ids = [id_list[i] for i in idxs]
    else:
        ordered_ids = fused_ids[:MAX_ANSWER_CHUNKS]

    # Build results
    out: List[RetrievalResult] = []
    for cid in ordered_ids:
        p = id_to_payload.get(cid)
        if not p:
            continue
        try:
            d = 1.0 - float(np.dot(np.array(p["emb"]), np.array(qvec))) if p["emb"] is not None else 0.0
        except Exception:
            d = 0.0
        out.append(RetrievalResult(
            id=cid,
            texto=p["texto"],
            metadata=p["meta"],
            distance=d
        ))
    return out


def format_citations(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    cites = []
    seen = set()
    for r in results[:MAX_ANSWER_CHUNKS]:
        meta = r.metadata or {}
        key = (meta.get("id_reso"), meta.get("seccion"),
               meta.get("parrafo_index"), meta.get("chunk_index"))
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
    from openai import OpenAI  # import local para evitar dependencia si no se usa
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


def generate_answer(query: str, contexts: List[RetrievalResult]) -> str:
    # SOURCES legibles
    items = []
    for c in contexts[:MAX_ANSWER_CHUNKS]:
        m = c.metadata or {}
        items.append({
            "id_reso": m.get("id_reso"),
            "seccion": m.get("seccion"),
            "fecha": m.get("fecha_legible") or m.get("fecha_iso"),
            "texto": c.texto
        })
    src_lines = []
    for it in items:
        head = f"({it['id_reso']}; {it['seccion']}; {it['fecha']})"
        src_lines.append(f"{head}\n{it['texto']}")
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
    user_msg = f"Pregunta:\n{query}\n\nFUENTES:\n{sources}"

    return _lmstudio_chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
