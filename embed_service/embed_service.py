#embed_service.py
import os
import math
from typing import List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI, APIError
import numpy as np

from settings_embed import EBED_MODEL, DOWNSTREAM_API_KEY, DOWNSTREAM_BASE_URL

app = FastAPI(title="Embedding Service", version="0.1.0")

# --- Clientes y Lógica ---
def get_downstream_client() -> OpenAI:
    #crea un cliente en OPENAI si la URL está configurada
    if not DOWNSTREAM_BASE_URL:
        return None
    return OpenAI(base_url=DOWNSTREAM_BASE_URL, api_key=DOWNSTREAM_API_KEY)

def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()  # [0.0, 0.0, ...]
    return (arr / norm).tolist()

def dummy_embed(texts: List[str]) -> List[List[float]]:
    """
    Genera embeddings 'dummy' deterministas y normalizados de 4 dimensiones.
    Perfecto para pruebas sin un modelo real.
    """
    embeddings = []
    for text in texts:
        # Crea un hash simple y úsalo para generar 4 números flotantes
        hash_val = hash(text)
        vec = [
            (hash_val % 1000) / 1000.0,
            (hash_val % 500) / 500.0,
            math.sin(hash_val),
            math.cos(hash_val),
        ]
        embeddings.append(l2_normalize(vec))
    return embeddings
# --- Modelos Pydantic para la API ---

class EmbedRequest(BaseModel):
    # Acepta 'input' (estándar de OpenAI) o 'text' para compatibilidad
    input: Optional[Union[str, List[str]]] = None
    text: Optional[Union[str, List[str]]] = None
    model: str = EBED_MODEL

    def get_texts(self) -> List[str]:
        texts = self.input or self.text
        if not texts:
            raise ValueError("Se debe proporcionar 'entrada' o 'texto'.")
        return [texts] if isinstance(texts, str) else texts

class Embediding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbedingResponse(BaseModel):
    object: str = "list"
    data: List[Embediding]
    model: str = EBED_MODEL

class QueryEmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str = EBED_MODEL

# --- Endpoints de la API ---
@app.get("/")
def health_check():
    # Indica que modelo esta usando el servicio
    is_downstream = bool(DOWNSTREAM_BASE_URL)
    model_info = {
        "status": "ok",
        "model": EBED_MODEL,
        "downstream_configured": is_downstream,
        "downstream_url": DOWNSTREAM_BASE_URL if is_downstream else None
    }
    return model_info

#No se usa la noralización L2 en este endpoint porque los embeddings downstream ya están normalizados
#En caso de que se necesite, usar l2_normalize en la lista resultante
@app.post("/embeddings", response_model=EmbedingResponse)
def create_embeddings(request: EmbedRequest):
    # Genera embeddings para la lista de textos
    client = get_downstream_client()
    texts = request.get_texts()

    if client:
        # Llama al servicio downstream
        try:
            response = client.embeddings.create(
                model=request.model,
                input=texts
            )
            embeddings = [e.embedding for e in response.data]
        except:
            return {"status": "error", "message": "Error al llamar al servicio downstream: {e}"}, 500
    else:
        #Modo Fallback: usa embeddings dummy
        embeddings = dummy_embed(texts)
    response_data = [
        Embediding(embedding=emb, index=i) for i, emb in enumerate(embeddings)
    ]
    return EmbedingResponse(data=response_data)
@app.post("/embed_query", response_model=QueryEmbeddingResponse)
def embed_query(request: EmbedRequest):
    # Genera un embedding para una sola consulta
    texts = request.get_texts()
    if len(texts) != 1:
        return {"status": "error", "message": "Se debe proporcionar exactamente un texto para embed_query."}, 400
    
    query = texts[0]

    embedding_response = create_embeddings(EmbedRequest(input=[query], model=request.model))
    if isinstance(embedding_response, tuple):  # Error handling
        raise Exception(embedding_response[0].get("message", "Error desconocido"))
    
    embedding = embedding_response.data[0].embedding
    return QueryEmbeddingResponse(embedding=embedding)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("EMBED_PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)