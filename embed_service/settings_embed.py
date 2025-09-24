import os

EBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
DOWNSTREAM_BASE_URL = os.getenv("DOWNSTREAM_BASE_URL", "http://host.docker.internal:1234/v1")
DOWNSTREAM_API_KEY = os.getenv("DOWNSTREAM_API_KEY", "notused")