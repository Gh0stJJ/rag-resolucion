import json
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from settings import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBED_MODEL, CHROMA_TENANT, CHROMA_DATABASE
from utils import to_chunks_from_arrays
from embeddins import embed_texts


def _client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(allow_reset=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )


def _collection(client):
    # Siempre crear si no existe
    return client.get_or_create_collection(CHROMA_COLLECTION)



def _clean_meta(md: Dict) -> Dict:
    out = {}
    for k, v in md.items():
        if v in (None, "", [], {}):
            continue
        if k == "anio":
            out[k] = int(v) if not isinstance(v, int) else v
        elif k in ("parrafo_index", "chunk_index"):
            out[k] = int(v)
        else: 
            out[k] = v
    return out


def get_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(allow_reset=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )

def get_collection(client):
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except:
        return client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def upsert_chunks(coll, docs: List[Dict]):
    ids, texts, metas, embeds = [], [], [], []
    for d in docs:
        _id = f"{d['id_reso']}|{d['seccion']}|p{d['parrafo_index']}|c{d['chunk_index']}"
        ids.append(_id)
        texts.append(d["texto"])
        meta = {k: v for k, v in d.items() if k != "texto"}
        metas.append(_clean_meta(meta))
    embeddings = embed_texts(texts, normalize=True) # Nomic 1.5
    coll.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

def ingest_folder(json_dir: str = "/data/json"):
    client = _client()
    coll = get_collection(client)

    p = Path(json_dir)
    files = sorted(p.glob("*.json"))
    total_chunks = 0

    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        chunks = list(to_chunks_from_arrays(doc))
        if not chunks:
            continue
        upsert_chunks(coll, chunks)
        total_chunks += len(chunks)

    return {"files": len(files), "chunks": total_chunks}

# Clean up the collection (for testing purposes)
def reset_collection():
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except:
        pass
    client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    return {"status": "collection reset"}

if __name__ == "__main__":
    stats = ingest_folder()
    print(stats)
