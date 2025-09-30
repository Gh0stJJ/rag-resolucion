# embed_service.py
# Este es un servidor de embeddings HÍBRIDO y autocontenido.
# Puede usar 'sentence-transformers' O 'FlagEmbedding' para cargar un modelo
# de Hugging Face y servirlo a través de una API compatible con OpenAI.
# -------------------------------------------------------------------------
# CAMBIAR DE MOTOR:
# En docker-compose.yml, ajusta la variable de entorno EMBED_LIBRARY:
# - EMBED_LIBRARY=sentence-transformers  (para la versión estable y general)
# - EMBED_LIBRARY=flagembedding          (para BGE-M3 con su motor optimizado)
# -------------------------------------------------------------------------

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import torch

# --- Configuración Dinámica del Motor de Embeddings ---
# Lee la librería a usar desde una variable de entorno. Por defecto, usa la más estable.
EMBED_LIBRARY = os.getenv("EMBED_LIBRARY", "flagembedding ")

# Importaciones condicionales basadas en la librería seleccionada
if EMBED_LIBRARY == "sentence-transformers":
    from sentence_transformers import SentenceTransformer
    print("Modo de librería seleccionado: sentence-transformers")
elif EMBED_LIBRARY == "flagembedding":
    from FlagEmbedding import BGEM3FlagModel
    print("Modo de librería seleccionado: flagembedding")
else:
    raise ValueError(f"Librería desconocida en EMBED_LIBRARY: '{EMBED_LIBRARY}'. Usa 'sentence-transformers' o 'flagembedding'.")


# --- Configuración General (sin cambios) ---
MODEL_ID = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Lógica de Detección de Dispositivo Mejorada
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Usando dispositivo: {DEVICE}")


# --- Carga del Modelo (Lógica Híbrida) ---
model = None
try:
    print(f"Cargando el modelo '{MODEL_ID}' en el dispositivo '{DEVICE}'...")
    if TOKEN:
        print("Token de Hugging Face detectado. Se usará para autenticación.")

    # ========================================================================
    # SECCIÓN DE CARGA DE MODELO: SENTENCE-TRANSFORMERS
    # ========================================================================
    if EMBED_LIBRARY == "sentence-transformers":
        model = SentenceTransformer(MODEL_ID, device=DEVICE, token=TOKEN)

    # ========================================================================
    # SECCIÓN DE CARGA DE MODELO: FLAGEMBEDDING (para BGE-M3)
    # ========================================================================
    elif EMBED_LIBRARY == "flagembedding":
        # use_fp16=False es más seguro para CPU/MPS. En GPU NVIDIA, poner a True.
        model = BGEM3FlagModel(MODEL_ID, device=DEVICE, use_fp16=False)
        # Nota: FlagEmbedding usa el token de HF automáticamente si la variable de entorno está presente.

    print("Modelo cargado exitosamente.")

except Exception as e:
    print(f"Error fatal: No se pudo cargar el modelo '{MODEL_ID}' con la librería '{EMBED_LIBRARY}'.")
    print(f"Detalle del error: {e}")
    exit(1)


app = FastAPI(title=f"Servidor de Embeddings ({EMBED_LIBRARY})", version="2.0.0")


# --- Modelos Pydantic para la API (sin cambios) ---
class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_ID

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str = MODEL_ID


# --- Endpoints de la API ---
@app.get("/")
def health_check() -> Dict[str, Any]:
    return {
        "estado": "ok",
        "modelo_cargado": MODEL_ID,
        "dispositivo": DEVICE,
        "libreria_activa": EMBED_LIBRARY
    }

@app.post("/embeddings", response_model=EmbeddingsResponse)
def create_embeddings(request: EmbeddingsRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    embeddings_list = []
    try:
        # ========================================================================
        # SECCIÓN DE ENCODING: SENTENCE-TRANSFORMERS
        # ========================================================================
        if EMBED_LIBRARY == "sentence-transformers":
            embeddings = model.encode(
                texts,
                normalize_embeddings=True, # Indispensable para sentence-transformers
                show_progress_bar=False
            )
            embeddings_list = [emb.tolist() for emb in embeddings]

        # ========================================================================
        # SECCIÓN DE ENCODING: FLAGEMBEDDING (para BGE-M3)
        # ========================================================================
        elif EMBED_LIBRARY == "flagembedding":
            # Por ahora, solo pedimos el vector denso para compatibilidad.
            output = model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            # La normalización es automática en FlagEmbedding.
            embeddings_list = [emb.tolist() for emb in output['dense_vecs']]

        # --- Lógica Común de Respuesta (sin cambios) ---
        response_data = [
            EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings_list)
        ]
        return EmbeddingsResponse(data=response_data, model=MODEL_ID)

    except Exception as e:
        print(f"Error al generar embeddings con '{EMBED_LIBRARY}': {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al generar embeddings: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("EMBED_PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)

