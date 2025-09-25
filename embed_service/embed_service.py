import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import numpy as np
from openai import OpenAI, APIConnectionError

# --- Configuración ---
# Se leen las variables de entorno directamente.
EMBED_PORT = int(os.getenv("EMBED_PORT", "8010"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "onnx-models/all-MiniLM-L6-v2-onnx")
DOWNSTREAM_BASE_URL = os.getenv("DOWNSTREAM_BASE_URL","http://tei-server:80")
DOWNSTREAM_API_KEY = os.getenv("DOWNSTREAM_API_KEY", "not-used")

app = FastAPI(title="Servicio de Embeddings", version="0.3.0")

# --- Lógica de Embeddings ---
def dummy_embed(text: str) -> List[float]:
    seed = sum(ord(c) for c in text)
    rng = np.random.default_rng(seed=seed)
    vec = rng.random(4, dtype=np.float32) - 0.5
    norm_vec = vec / np.linalg.norm(vec)
    return [round(float(x), 3) for x in norm_vec]

# --- Cliente OpenAI ---
downstream_client = None
if DOWNSTREAM_BASE_URL:
    downstream_client = OpenAI(
        base_url=DOWNSTREAM_BASE_URL,
        api_key=DOWNSTREAM_API_KEY,
        timeout=30.0,
        max_retries=1
    )

# --- Modelos Pydantic para la API ---
class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = EMBED_MODEL

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str = EMBED_MODEL

# --- Endpoints de la API ---
@app.get("/")
def health_check():
    info: Dict[str, Any] = {
        "estado": "ok",
        "modelo_configurado": EMBED_MODEL,
        "modo": "proxy" if downstream_client else "dummy",
        "downstream_url": DOWNSTREAM_BASE_URL,
    }
    if not downstream_client:
        info["downstream_estado"] = "no_configurado"
        return info
    try:
        respuesta_prueba = downstream_client.embeddings.create(model=EMBED_MODEL, input=["test"])
        dimension = len(respuesta_prueba.data[0].embedding)
        info["downstream_estado"] = "ok"
        info["downstream_dimension_real"] = dimension
    except APIConnectionError as e:
        info["downstream_estado"] = "error_conexion"
        info["downstream_error"] = f"No se pudo conectar: {e.__cause__}"
    except Exception as e:
        info["downstream_estado"] = "error"
        info["downstream_error"] = str(e)
    return info

@app.post("/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(request: EmbeddingsRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    if downstream_client:
        try:
            response = downstream_client.embeddings.create(model=request.model, input=texts)
            embeddings = [e.embedding for e in response.data]
        except Exception as e:
            # --- CAMBIO CLAVE: Manejo de errores correcto para FastAPI ---
            raise HTTPException(status_code=500, detail=f"Error en servicio downstream: {e}")
    else:
        embeddings = [dummy_embed(t) for t in texts]
    data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]
    return EmbeddingsResponse(data=data, model=request.model or EMBED_MODEL)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("EMBED_PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)