import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Pinecone Configuration
# ==============================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "qwen8b-rag"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "qwen8b"
)

PINECONE_DIMENSION = int(
    os.getenv(
        "PINECONE_DIMENSION",
        "4096"
    )
)

PINECONE_METRIC = os.getenv(
    "PINECONE_METRIC",
    "cosine"
)

PINECONE_CLOUD = os.getenv(
    "PINECONE_CLOUD",
    "aws"
)

PINECONE_REGION = os.getenv(
    "PINECONE_REGION",
    "us-east-1"
)


# ==============================
# Ollama Configuration
# ==============================

BASE_URL = os.getenv(
    "BASE_URL",
    "http://localhost:11434"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding:8b"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:8b"
)

# ==============================
# RAG Configuration
# ==============================

TOP_K = int(
    os.getenv(
        "TOP_K",
        "5"
    )
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200"
    )
)