# ingest.py
import json
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from settings import  CHROMA_COLLECTION, BM25_INDEX_DIR, CHROMA_DISTANCE
from utils import to_chunks_from_arrays
from embeddins import embed_texts
from math import ceil
from collections import Counter
from lex import open_or_create as bm25_open_or_create, add_chunks as bm25_add_chunks
from lex import reset as bm25_reset
from chroma_utils import build_client as build_ch_client, get_or_create_collection_with_space



# Obligatory fields in metadata
REQUI_FIELDS = {"id_reso", "seccion", "parrafo_index", "chunk_index", "texto"}

UPSERT_BATCH = 256


def _ensure_collection(client):
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except:
        return client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})




def _clean_meta(md: Dict) -> Dict:
    out = {}
    for k, v in md.items():
        if v in (None, "", [], {}):
            continue
        if k == "anio":
            try:
                out[k] = int(v) if not isinstance(v, int) else v
            except Exception:
                continue # skip if cannot convert Patch: Error 500 issue
        elif k in ("parrafo_index", "chunk_index"):
            try:
                out[k] = int(v)
            except Exception:
                continue # skip if cannot convert Patch: Error 500 issue
        else:
            out[k] = v
    return out



def _validate_chunk(d: Dict) -> Tuple[bool, str]:
    for f in REQUI_FIELDS:
        if f not in d:
            return False, f"Missing field {f}"
        
    if not isinstance(d["texto"], str) or not d["texto"].strip():
        return False, "Empty or invalid 'texto' field"
    for idx_field in ("parrafo_index", "chunk_index"):
        try:
            int(d[idx_field])
        except Exception:
            return False, f"Invalid integer field {idx_field}"
        
    # Val id_reso and seccion as non-empty strings
    if not isinstance(d["id_reso"], str) or not d["id_reso"].strip():
        return False, "Empty or invalid 'id_reso' field"
    if not isinstance(d["seccion"], str) or not d["seccion"].strip():
        return False, "Empty or invalid 'seccion' field"
    return True, ""

def _build_id(d: Dict) -> str:
    return f"{d['id_reso']}|{d['seccion']}|p{d['parrafo_index']}|c{d['chunk_index']}"

def _prepare_batch(valid_docs: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    ids, texts, metas = [], [], []
    for d in valid_docs:
        _id = _build_id(d)
        ids.append(_id)
        texts.append(d["texto"])
        meta = {k: v for k, v in d.items() if k != "texto"}
        metas.append(_clean_meta(meta))
    return ids, texts, metas

def _upsert_in_batches(coll, valid_docs: List[Dict], report: Dict):
    """
    Upsert documents in batches to avoid memory issues.
    """
    seen = set()
    deduped = []

    for d in valid_docs:
        _id = _build_id(d)
        if _id in seen:
            report["skipped_reason_counts"]["duplicate_id_in_batch"] += 1
            continue
        seen.add(_id)
        deduped.append(d)

    # make batches and upsert
    n = len(deduped)
    if n == 0:
        return
    
    batches = ceil(n / UPSERT_BATCH)
    for i in range(batches):

        start = i * UPSERT_BATCH
        end = min(n, start + UPSERT_BATCH)
        sub = deduped[start:end]

        ids, texts, metas = _prepare_batch(sub)

        try:
            embeddings = embed_texts(texts, normalize=True) # Nomic 1.5
            if len(embeddings) != len(texts):
                report["skipped_reason_counts"]["embeddings_length_mismatch"] += len(texts)
                continue
            coll.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
            report["chunks_ingested"] += len(texts)
        except Exception as e:
            # if all fails, we dowgrade to per item upsert and point the error
            report["batches_failed"] += 1
            for d in sub:
                try:
                    _ids, _texts, _metas = _prepare_batch([d])
                    _emb = embed_texts(_texts, normalize=True)
                    coll.upsert(ids=_ids, documents=_texts, embeddings=_emb, metadatas=_metas)
                    report["chunks_ingested"] += 1
                except Exception as e2:
                    report["chunks_skipped"] += 1
                    report["errors"].append({"id": _build_id(d), "error": repr(e2)})


def ingest_folder(json_dir: str = "/data/json") -> Dict:
    client = get_client()
    coll = get_collection(client)

    ix = bm25_open_or_create(BM25_INDEX_DIR)

    p = Path(json_dir)
    files = sorted(p.glob("*.json"))

    report = {
        "files_seen": len(files),
        "files_ok": 0,
        "files_failed": 0,
        "chunks_seen": 0,
        "chunks_valid": 0,
        "chunks_ingested": 0,
        "chunks_skipped": 0,
        "batches_failed": 0,
        "skipped_reason_counts": Counter(),
        "file_stats": [],   # per file
        "errors": []       
    }

    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            report["files_failed"] += 1
            report["errors"].append({"file": f.name, "error": f"invalid_json:{repr(e)}"})
            continue

        try:
            chunks = list(to_chunks_from_arrays(doc))
        except Exception as e:
            report["files_failed"] += 1
            report["errors"].append({"file": f.name, "error": f"chunking_failed:{repr(e)}"})
            continue

        if not chunks:
            # archivo sin chunks útiles
            report["file_stats"].append({"file": f.name, "chunks": 0, "valid": 0, "ingested": 0, "skipped": 0})
            report["files_ok"] += 1
            continue

        report["files_ok"] += 1
        report["chunks_seen"] += len(chunks)

        valid, skipped = [], 0
        for idx, d in enumerate(chunks):
            ok, reason = _validate_chunk(d)
            if ok:
                report["chunks_valid"] += 1
                valid.append(d)
            else:
                skipped += 1
                report["chunks_skipped"] += 1
                report["skipped_reason_counts"][reason] += 1
                report["errors"].append({
                    "file": f.name,
                    "chunk_index_in_file": idx,
                    "reason": reason
                })

        before_ingested = report["chunks_ingested"]
        _upsert_in_batches(coll, valid, report)
        ingested_now = report["chunks_ingested"] - before_ingested

        # Add to BM25 index
        try:
            bm25_add_chunks(ix, valid)
        except Exception as e:
            report["errors"].append({
                "file": f.name,
                "error": f"bm25_indexing_failed:{repr(e)}"
            })

        report["file_stats"].append({
            "file": f.name,
            "chunks": len(chunks),
            "valid": len(valid),
            "ingested": ingested_now,
            "skipped": skipped
        })

    report["skipped_reason_counts"] = dict(report["skipped_reason_counts"])
    return report



def get_client():
    return build_ch_client()

def get_collection(client):
    _ , coll = get_or_create_collection_with_space(client)
    return coll


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



# Clean up the collection (for testing purposes)
def reset_collection():
    client = build_ch_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except:
        pass
    client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": CHROMA_DISTANCE})
    return {"status": "collection reset"}

if __name__ == "__main__":
    stats = ingest_folder()
    print(stats)
