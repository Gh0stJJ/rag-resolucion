# settings.py
import os

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))  # mapeado en compose
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "resoluciones_cu")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")
CHROMA_DISTANCE = os.getenv("CHROMA_DISTANCE", "cosine")

TOP_K = int(os.getenv("TOP_K", "50"))  
MAX_ANSWER_CHUNKS = int(os.getenv("MAX_ANSWER_CHUNKS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))

CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "1000"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "0"))
CHUNK_USE_SPACY = os.getenv("CHUNK_USE_SPACY", "1") == "1"
SPACY_MODEL = os.getenv("SPACY_MODEL", "es_core_news_sm")
SPACY_MAX_LENGTH = int(os.getenv("SPACY_MAX_LENGTH", "5000000")) #jic 

LLM_BASE_URL = os.getenv("LLM_BASE_URL") 
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5@f32")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "notused")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")

#BM25 Configuration
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "/data/whoosh_index")
K_LEX = int(os.getenv("K_LEX", "200"))           
RERANK_TOP = int(os.getenv("RERANK_TOP", "24"))   
RRF_K = int(os.getenv("RRF_K", "60"))
