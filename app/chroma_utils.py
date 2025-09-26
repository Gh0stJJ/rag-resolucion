# chroma_utils.py
import chromadb
from chromadb.config import Settings
from settings import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, CHROMA_TENANT, CHROMA_DATABASE, CHROMA_DISTANCE

def build_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(allow_reset=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

def get_or_create_collection_with_space(client=None):
    client = client or build_client()
    try:
        coll = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        coll = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": CHROMA_DISTANCE})
    return client, coll

    try:
        meta = getattr(coll, "metadata", {}) or {}
        space = (meta.get("hnsw:space") or "").lower()

        if space and space != CHROMA_DISTANCE:
            print(f"[WARN] Collection '{CHROMA_COLLECTION}' hnsw:space='{space}' != expected '{CHROMA_DISTANCE}'")
    except Exception:
        pass
    return client, coll