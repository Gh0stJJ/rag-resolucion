#main_agent.py
from fastapi import FastAPI
from openai import OpenAI
from settings import OPENAI_API_BASE, API_KEY

# Importar la lógica y el modelo de datos del agente de filtros
from filter_agent.filter_agent import PromtRequest, get_structured_filters

# --- Configuración de la Aplicación y Cliente ---

#os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

# Inicializa el cliente de OpenAI una sola vez cuando la aplicación arranca
client = OpenAI(base_url=OPENAI_API_BASE, api_key=API_KEY)

# Inicializa la aplicación FastAPI
app = FastAPI(
    title="Servidor de Agentes",
    description="API para interactuar con diferentes agentes de IA.",
    version="1.0.0"
)

# --- Endpoints de la API ---

@app.post("/query-filters", tags=["Agentes de Búsqueda"])
async def query_filters_endpoint(request: PromtRequest):
    """
    Recibe un prompt de usuario y utiliza el 'Filter Agent' para extraer
    filtros estructurados en formato JSON.
    """
    response = await get_structured_filters(request, client)
    return response

# --- Punto de Entrada para Desarrollo Local ---
if __name__ == "__main__":
    import uvicorn
    # Corre el servidor en el puerto 8020, ideal para el entorno Docker
    uvicorn.run("main_agent:app", host="0.0.0.0", port=8020,reload=True)
