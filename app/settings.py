import os

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))  # mapeado en compose
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "resoluciones_cu")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")


TOP_K = int(os.getenv("TOP_K", "12"))  
MAX_ANSWER_CHUNKS = int(os.getenv("MAX_ANSWER_CHUNKS", "6"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL") 
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama-3.1-8b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "notused")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")