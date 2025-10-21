# rag.py
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from chroma_utils import build_client as build_ch_client, get_or_create_collection_with_space
from pydantic import BaseModel
from collections import defaultdict
from openai import OpenAI 
from hyde import generate_hypothetical_document
from settings import (
    USE_MULTIQUERY, USE_HYDE, MULTIQUERY_N,
    TOP_K, MAX_ANSWER_CHUNKS, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY,
    CHROMA_TENANT, CHROMA_DATABASE, TEMPERATURE, MAX_TOKENS,
    K_LEX, RERANK_TOP, RRF_K, BM25_INDEX_DIR, RERANKER_ENABLED
)
from embeddins import embed_query
from lex import open_or_create as bm25_open_or_create, search as bm25_search
from multiquery import decompose_query_into_subqueries
from utils import normalize_name
from reranker import rerank

class RetrievalResult(BaseModel):
    id: str
    texto: str
    metadata: Dict[str, Any]
    distance: float

    @property
    def id_reso(self) -> str | None:
        return (self.metadata or {}).get("id_reso")


def build_client():
    return build_ch_client()

def build_client_collection():
    client, coll = get_or_create_collection_with_space()
    return client, coll
from typing import Any, Dict, List

# (Asumimos que _normalize_name existe en este scope)
# def _normalize_name(name: str) -> List[str]: ...

def _build_where_document(filtros: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Lógica Anidada: (A OR B OR (c1 AND c2 AND ...))
    """
    main_conditions = []

    # Artículos y otros_doc → Se añaden como condiciones OR individuales
    for key in ["articulos", "otros_doc"]:
        values = filtros.get(key)
        if values:
            # [{"$contains": "Art1"}, {"$contains": "Doc1"}]
            main_conditions.extend([{"$contains": str(v)} for v in values])

    # Nombres → Se agrupan en un bloque "$and"
    nombres_list = filtros.get("nombres")
    if nombres_list:
        
        # ["Juan Pérez"] se convierte en ["Juan", "Pérez"]
        name_parts = [
            part
            for name in nombres_list
            for part in normalize_name(name) # Usa la función
        ]

        if name_parts:
            # [{"$contains": "Juan"}, {"$contains": "Pérez"}]
            and_conditions = [{"$contains": n} for n in name_parts]
            
            # Se añade el bloque AND 
            main_conditions.append({"$and": and_conditions})

    # Si no hay nada, devuelve None
    if not main_conditions:
        return None

    # Si solo hay una condición,
    if len(main_conditions) == 1:
        return main_conditions[0]

    # Envuelve todo en "$or"
    #print(f"WHERE DOCUMENT: { {'$or': main_conditions} }")
    return {"$or": main_conditions}

def _build_where(filtros: Dict[str, Any]) -> Dict[str, Any]:
    conditions = []
    # Filtro por id_reso
    if "id_reso" in filtros and filtros["id_reso"]:
        conditions.append({"id_reso": {"$eq": filtros["id_reso"]}})

    # Filtro por tipo de sesión
    if "tipo_sesion" in filtros and filtros["tipo_sesion"]:
        conditions.append({"tipo": {"$eq": filtros["tipo_sesion"]}})

    # Filtro por Rango de fechas
    # Asume que las fechas vienen en formato "YYYY-MM-DD" y el metadata está en YYYYMMDD
    date_from = filtros.get("date_from")
    date_to = filtros.get("date_to")

    if date_from:
        try:
            fecha_yyyymmdd_from = int(date_from.replace("-", ""))
            conditions.append({"fecha_yyyymmdd": {"$gte": fecha_yyyymmdd_from}})
        except (ValueError, TypeError):
            print(f"Advertencia: Formato de fecha de inicio inválido: {date_from}")

    if date_to:
        try:
            fecha_yyyymmdd_to = int(date_to.replace("-", ""))
            conditions.append({"fecha_yyyymmdd": {"$lte": fecha_yyyymmdd_to}})
        except (ValueError, TypeError):
            print(f"Advertencia: Formato de fecha de fin inválido: {date_to}")
    return {"$and": conditions} if conditions else None


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

    extra_results = {}

    expanded_queries = [query]
    subqueries, pseudo_doc = [], None

    #gen subqueries TODO (Future MCP)
    if USE_MULTIQUERY:
        subqueries = decompose_query_into_subqueries(query, n=MULTIQUERY_N)
        expanded_queries.extend(subqueries)
        extra_results["subqueries"] = subqueries

    #gen hypothetical document (HyDE) TODO (Future MCP)
    if USE_HYDE:
        pseudo_doc = generate_hypothetical_document(query)
        expanded_queries.append(pseudo_doc)
        extra_results["hyde"] = pseudo_doc[:200]

    qvecs = [embed_query(q) for q in expanded_queries]

    dense_ids: List[str] = []

    # Construir el filtro WHERE y WHERE DOCUMENT
    where_metadata_filter = _build_where(filtros) if filtros else None
    where_document_filter = _build_where_document(filtros) if filtros else None

    # dense search in ChromaDB
    for qvec in qvecs:
        res_dense_q = coll.query(
            query_embeddings=[qvec],
            n_results=TOP_K,
            where=where_metadata_filter,
            where_document=where_document_filter,
            include=["distances"]  
        )
        if res_dense_q and res_dense_q.get("ids"):
            dense_ids.extend(res_dense_q["ids"][0] or [])

    seen = set()
    dense_ids = [x for x in dense_ids if not (x in seen or seen.add(x))]

    # lexico (BM25) – best-effort
    try:
        ix = bm25_open_or_create(BM25_INDEX_DIR)
        lex_hits = bm25_search(ix, query, filtros=filtros, limit=K_LEX)
        lex_ids = [cid for cid, _ in lex_hits]
    except Exception as e:
        print(f"[BM25] Error: {repr(e)}")
        lex_ids = []

    # Results Fusion rrf
    fused_ids = _rrf_fuse(dense_ids, lex_ids, k=RRF_K)
    # stick ids to docs
    if filtros and filtros.get("id_reso"):
        prefix = filtros["id_reso"] + "|"
        prim = [cid for cid in fused_ids if cid.startswith(prefix)]
        rest = [cid for cid in fused_ids if not cid.startswith(prefix)]
        fused_ids = prim + rest

    fused_ids = fused_ids[:max(RERANK_TOP, MAX_ANSWER_CHUNKS)]
    if not fused_ids:
        return [], {}

    # get payloads by ID
    got = coll.get(ids=fused_ids, include=["documents", "metadatas", "embeddings"])
    id_to_payload: Dict[str, Dict[str, Any]] = {}
    for i, cid in enumerate(got.get("ids", [])):
        id_to_payload[cid] = {
            "texto": got["documents"][i] if i < len(got["documents"]) else "",
            "meta": got["metadatas"][i] if i < len(got["metadatas"]) else {},
            "emb": got["embeddings"][i] if i < len(got["embeddings"]) else None
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

    if RERANKER_ENABLED:
        try:
            doc_pairs = [(cid, id_to_payload[cid]["texto"]) for cid in ordered_ids if cid in id_to_payload]
            reranked = rerank(query, doc_pairs)
            reranked_ids = [rid for rid, _ in reranked]
            # Keep only valid reranks
            ordered_ids = [cid for cid in reranked_ids if cid in ordered_ids]
            print(f"[Reranker] Re-ranked {len(ordered_ids)} documents.")
        except Exception as e:
            print(f"[Reranker] Error during reranking: {repr(e)}")

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
    return out, extra_results


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
            Eres un asistente experto en **Resoluciones del Consejo Universitario — Universidad de Cuenca** dentro del marco jurídico de la República del Ecuador.  
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



