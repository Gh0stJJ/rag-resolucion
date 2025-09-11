from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from settings import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBED_MODEL,
    TOP_K, MAX_ANSWER_CHUNKS, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY,
    CHROMA_TENANT, CHROMA_DATABASE
)
from pydantic import BaseModel

from embeddins import embed_query
from openai import OpenAI


class RetrievalResult(BaseModel):
    id: str
    texto: str
    metadata: Dict[str, Any]
    distance: float

def build_client():
    # Si tu servidor NO usa multi-tenant/database, elimina tenant= y database=
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


    res = coll.query(
        query_embeddings=[qvec],
        n_results=TOP_K,
        where=where or None,
        include=["documents", "metadatas", "distances"]
    )

    out = []
    for i in range(len(res["ids"][0])):
        out.append(RetrievalResult(
            id=res["ids"][0][i],
            texto=res["documents"][0][i],
            metadata=res["metadatas"][0][i],
            distance=res["distances"][0][i],
        ))
    out.sort(key=lambda r: r.distance)
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
   
    # Prepara contexto
    ctx = []
    for c in contexts[:MAX_ANSWER_CHUNKS]:
        m = c.metadata
        header = f"[{m.get('id_reso')} | {m.get('seccion')} | anio={m.get('anio')} | fecha={m.get('fecha_legible') or m.get('fecha_iso')}]"
        ctx.append(f"{header}\n{c.texto}")
    context_text = "\n\n---\n\n".join(ctx) if ctx else "(no hay fragmentos relevantes)"

    if LLM_BASE_URL:
        system_msg = (
            "Eres un asistente para responder preguntas sobre resoluciones del Consejo Universitario.\n"
            "Responde SOLO con la información proporcionada en los fragmentos.\n"
            "Si la evidencia no es suficiente, indica claramente que no se encontró evidencia suficiente.\n"
            "Incluye SIEMPRE citas al final en formato: (id_reso; seccion; fecha).\n"
            "Responde en español, conciso y factual."
        )
        user_msg = (
            f"Pregunta: {query}\n\n"
            f"Fragmentos:\n{context_text}"
        )
        try:
            return _lmstudio_chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
                max_tokens=450
            )
        except Exception as e:
            joined = "\n\n".join([f"- {c.metadata.get('id_reso')} [{c.metadata.get('seccion')}]: {c.texto}" for c in contexts[:MAX_ANSWER_CHUNKS]])
            return f"(Advertencia: LLM local no disponible: {e})\n\nContexto relevante:\n\n{joined}"

    # Baseline sin LLM 
    joined = "\n\n".join([f"- {c.metadata.get('id_reso')} [{c.metadata.get('seccion')}]: {c.texto}" for c in contexts[:MAX_ANSWER_CHUNKS]])
    return f"Contexto relevante encontrado (modo local sin LLM):\n\n{joined}\n\nSugerencia: configura LLM_BASE_URL para respuestas redactadas."
