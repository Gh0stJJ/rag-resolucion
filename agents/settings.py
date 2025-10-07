import os

# USAR SERVICIO DE 
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://host.docker.internal:1234/v1")
API_KEY = "not-needed" # Requerido por la librería, aunque no se use
