# embeddins.py
from typing import List
import math 
from openai import OpenAI
from settings import EMBED_BASE_URL, EMBED_MODEL, EMBED_API_KEY


def _client() -> OpenAI:
    if not EMBED_BASE_URL:
        raise RuntimeError("EMBED_BASE_URL not set")
    return OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)

def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]

def embed_texts(texts: List[str], normalize:bool = True, batch_size: int = 64) -> List[List[float]]:
    """
    Call v1/embeddings from LM Studio
    Returns a list of vectors (float32/ float64)
    """

    client = _client()
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=chunk
        )
        vecs = [e.embedding for e in resp.data]

        if normalize:
            vecs = [l2_normalize(v) for v in vecs]
        out.extend(vecs)
    return out


def embed_query(query: str, normalize: bool = True) -> List[float]:
    return embed_texts([query], normalize=normalize)[0]
