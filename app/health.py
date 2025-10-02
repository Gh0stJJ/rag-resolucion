# app/health.py
from typing import Dict, Any
import httpx
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from settings import (
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, CHROMA_TENANT, CHROMA_DATABASE,
    EMBED_BASE_URL, EMBED_MODEL, EMBED_API_KEY,
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
)

TIMEOUT_S = 5.0

def check_chroma() -> Dict[str, Any]:
    info: Dict[str, Any] = {"ok": False, "collection_dimension": None}
    base = f"http://{CHROMA_HOST}:{CHROMA_PORT}"

    # Heartbeat v2
    try:
        r = httpx.get(f"{base}/api/v2/heartbeat", timeout=TIMEOUT_S)
        info["heartbeat_status"] = r.status_code
        info["heartbeat_json"] = r.json()
    except Exception as e:
        info["error"] = f"chroma heartbeat failed: {e}"
        return info
    
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=Settings(allow_reset=False),
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
        # Intenta obtener la colección. Si no existe, no es un error, solo no hay datos.
        try:
            coll = client.get_collection(CHROMA_COLLECTION)
            info["collection_found"] = True
            info["collection_count"] = coll.count()
            # Si hay datos, verifica la dimensión del primer embedding
            if coll.count() > 0:
                peek = coll.peek(limit=1)
                if peek.get("embeddings") and len(peek["embeddings"]) > 0:
                    info["collection_dimension"] = len(peek["embeddings"][0])
        except Exception:
            info["collection_found"] = False
        info["ok"] = True
    except Exception as e:
        info["error"] = f"chroma client/collection failed: {e}"

    return info

def check_embed() -> Dict[str, Any]:
    info: Dict[str, Any] = {"ok": False, "model": EMBED_MODEL, "base_url": EMBED_BASE_URL}
    if not EMBED_BASE_URL:
        info["error"] = "EMBED_BASE_URL not set"
        return info
    try:
        client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)
        resp = client.embeddings.create(model=EMBED_MODEL, input=["ping"])
        dim = len(resp.data[0].embedding)
        info.update({"ok": True, "dimension": dim})
    except Exception as e:
        info["error"] = f"embeddings failed: {e}"
    return info

def check_chat() -> Dict[str, Any]:
    info: Dict[str, Any] = {"ok": False, "model": LLM_MODEL, "base_url": LLM_BASE_URL}
    if not LLM_BASE_URL:
        info["error"] = "LLM_BASE_URL not set"
        return info
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Eres un asistente de prueba de salud."},
                {"role": "user", "content": "Di 'pong' en una sola palabra."}
            ],
            temperature=0.0,
            max_tokens=4,
        )
        content = resp.choices[0].message.content.strip().lower()
        info["response"] = content
        info["ok"] = ("pong" in content)
    except Exception as e:
        info["error"] = f"chat failed: {e}"
    return info

def full_health() -> Dict[str, Any]:
    chroma = check_chroma()
    embeddings = check_embed()
    chat = check_chat()
    
    all_ok = chroma.get("ok") and embeddings.get("ok") and chat.get("ok")

    return {
        "status": "ok" if all_ok else "degraded",
        "chroma": chroma,
        "embeddings": embeddings,
        "chat": chat,
    }
