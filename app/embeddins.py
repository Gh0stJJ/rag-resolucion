import os
from typing import List
import httpx
from settings import EMBED_BASE_URL

#Cliente httpx para mejor performance
client = httpx.Client(timeout=60)

def embed_texts(texts: List[str], **kwargs) -> List[List[float]]:
    # Llama al servicio de embeddings 
    if not EMBED_BASE_URL:
        raise ValueError("EMBED_BASE_URL no está configurado.")
    url = f"{EMBED_BASE_URL}/embeddings"
    payload = {"input": texts}

    try:
        reponse = client.post(url, json=payload)
        reponse.raise_for_status()
        # El servicio devuelve un objeto EmbeddingResponse
        json_response = reponse.json()
        embeddings = [item["embedding"] for item in json_response.get("data", [])]
        return embeddings
    except httpx.HTTPError as e:
        print(f"Error HTTP al llamar al servicio de embeddings: {e.response.text if e.response else 'No response'}")
    except Exception as e:
        print(f"Error al llamar al servicio de embeddings: {e}")

def embed_query(query: str) -> List[float]:
    """
    Genera el embedding para una única consulta (query).
    """
    embeddings = embed_texts([query])
    
    # Si hubo un error, el resultado será [None]. Lo manejamos aquí.
    if not embeddings or embeddings[0] is None:
        # Esto previene el crash en ChromaDB.
        raise ValueError("No se pudo generar el embedding para la consulta.")
        
    return embeddings[0]