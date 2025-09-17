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
    if not LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL no está configurado. Define LLM_BASE_URL para usar LM Studio.")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if "Only user and assistant roles are supported" in msg or "jinja template" in msg:
            # Fallback: aplanar 'system' dentro del primer 'user'
            sys_text = "\n".join(m["content"] for m in messages if m.get("role") == "system")
            usr_texts = [m["content"] for m in messages if m.get("role") == "user"]
            merged_user = ("INSTRUCCIONES DEL SISTEMA:\n" + sys_text + "\n\n" if sys_text else "") + "\n\n".join(usr_texts)

            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": merged_user}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        # Si es otro error, propágalo
        raise

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
            """
            ===SYSTEM===
            Eres un asistente experto en **Resoluciones del Consejo Universitario — Universidad de Cuenca**.  

            Reglas obligatorias:
            1. Redacta respuestas lo más completas y extensas posible, desarrollando cada punto relevante y proporcionando explicaciones detalladas basadas en los fragmentos. Si la pregunta lo permite, incluye contexto adicional de los fragmentos y relaciona la información para dar una respuesta profunda y exhaustiva.
            2. Responde **solo** con la información contenida en los FRAGMENTOS.  
            - Nunca uses conocimiento externo ni inventes información.  
            - Si no hay evidencia suficiente, dilo explícitamente (Tipo 4).  
            3. Responde siempre en **español**.  
            4. Usa el **Tipo de respuesta** (1-4) según corresponda:  
            - Tipo 1: Preciso y detallado para información puntual.  
            - Tipo 2: Resumen amplio cuando lo pidan.  
            - Tipo 3: Detalle de artículos o resoluciones específicas.
                - Tipo 3.1: Explicación detallada cuando se pida "explica" o "detalla".  
            - Tipo 4: Indicar falta de evidencia suficiente.  
            5. Cada respuesta debe seguir este **formato de salida**:  
                Tipo de respuesta: X
                Respuesta: ...
                Evidencia: 
            6. Cuando cites texto literal de un fragmento, usa comillas cortas (“...”) e Incluye SIEMPRE citas al final en formato: (id_reso; seccion; fecha).  
            7. Mantén tono formal, educado y cordial.  
            8. Si hay contradicciones entre fragmentos, indícalas y lista las referencias.  
            ===END SYSTEM===
            ===EJEMPLOS===

            ---EJEMPLO TIPO 4---
            Pregunta: "¿Qué resolución aprobó la modificación del calendario académico?" 
            Fragmentos:  
            [RES-UC-050:1] "Que, mediante Resolución No. UC-R-001-2025 de 13 de enero de 2025, expedida por la primera autoridad ejecutiva de la Universidad de Cuenca resolvió: “ARTÍCULO ÚNICO. – Aprobar el Calendario Académico …"  

            Respuesta esperada:  
            Tipo de respuesta: 4  
            Respuesta: "No se encontró evidencia suficiente para determinar si la resolución aprobó la modificación del calendario académico.\n\n
                        Se encuentran varios fragmentos que mencionan la reforma del calendario académico 2025-2026, pero no hay una resolución específica que lo apruebe. Algunas de las resoluciones mencionadas se refieren a la aprobación del calendario para ingreso de estudiantes y otros trámites relacionados con el proceso de matrículas, pero no se menciona explícitamente la modificación del calendario académico.
                        \n\nPor lo tanto, no se puede determinar si se ha aprobado la modificación del calendario académico.
            Evidencia: [UC-CU-RES-140-2025]  
            ---FIN---

            ---EJEMPLO TIPO 2---
            Pregunta: "¿De qué trata la Resolución UC-CU-RES-144-2025?"
            Fragmentos:
            Tipo de respuesta: 2
            Respuesta: La Resolución UC-CU-RES-144-2025, aprobada el 29 de julio de 2025, trata sobre el cambio de régimen de dedicación de docentes titulares en la Universidad de Cuenca.
            - Se aprueba el cambio de dedicación de la Dra. Jannethe Tapia Cárdenas, de tiempo parcial a tiempo completo, debido a que asumió la Dirección de Internado.
            - Se aprueba el cambio de dedicación del Dr. Fernando González Calle, de tiempo parcial a medio tiempo.
            - Se aprueba el cambio temporal del Mgt. Darwin Fabián Sandoval Lozano, de medio tiempo a tiempo completo, en la Facultad de Ciencias de la Hospitalidad.
            - La resolución también establece la notificación a las Facultades implicadas, al Vicerrectorado Académico y a varias direcciones institucionales, así como a los docentes mencionados.
            Evidencia: (UC-CU-RES-144-2025)
            ---FIN---
            ---EJEMPLO TIPO 1---
            Pregunta: "¿Qué resuelve la Resolución UC-CU-RES-024-2025?"
            Fragmentos:
            Tipo de respuesta: 1
            Respuesta: La Resolución UC-CU-RES-024-2025, aprobada el 25 de febrero de 2025, resolvió tres puntos principales:
            1. Aceptar la excusa presentada por la Dra. Jessica Ercilia Castillo Núñez, quien solicitó no participar como miembro suplente del Tribunal Electoral por inscribir su candidatura como representante suplente de profesores.
            2. Designar al Lcdo. Freddy Patricio Cabrera Ortiz como nuevo miembro suplente del Tribunal Electoral en calidad de profesor de la Universidad de Cuenca.
            3. Disponer la notificación a la docente excusada, al Tribunal Electoral, al nuevo designado y a diversas unidades administrativas, incluyendo la Secretaría General, Procuraduría, Auditoría Interna y Comunicación Institucional, para su publicación en la web.
            Evidencia: (UC-CU-RES-024-2025)
            ---FIN---
            --EJEMPLO TIPO 3---
            Pregunta: "¿Qué resolvió la Resolución UC-CU-RES-048-2025 sobre la Maestría en Salud Sexual y Reproductiva?"
            Fragmentos:
            Tipo de respuesta: 3
            Respuesta: La Resolución UC-CU-RES-048-2025, aprobada el 11 de marzo de 2025, resolvió aprobar institucionalmente el programa de Maestría en Salud Sexual y Reproductiva con mención en Promoción de la Salud, en modalidad semipresencial.
            Además, dispuso notificar a diversas dependencias internas (Dirección de Posgrados, Dirección Administrativa, Dirección Financiera, Rectorado, Procuraduría, Auditoría Interna, entre otras) y encargar a la Dirección de Comunicación Institucional su publicación en la página web de la Universidad.
            Evidencia: (UC-CU-RES-048-2025)
            ---FIN--
            ===END EJEMPLOS==="""
    )
    user_msg = (
            f"""
            ===USER===
                Pregunta usuario: {query}\n\n

                FRAGMENTOS: \n{sources}\n\n

                Instrucciones adicionales al asistente:  
                - Identifica el tipo de respuesta adecuado según la pregunta.  
                - Usa únicamente lo que está en los fragmentos.  
                - Si no encuentras evidencia suficiente, responde en Tipo 4.  
            ===END USER===
            """
    )
    return _lmstudio_chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

