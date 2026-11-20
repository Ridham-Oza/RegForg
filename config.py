"""
Central configuration for RegForg.

Every tunable lives here and can be overridden from `.env`.
No other module hard-codes model names, URLs or RAG parameters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", ROOT_DIR / "Data" / "documents"))
MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", ROOT_DIR / "Data" / "index_manifest.json"))
QUERY_LOG_PATH = Path(os.getenv("QUERY_LOG_PATH", ROOT_DIR / "logs" / "queries.jsonl"))
EVAL_DATASET_PATH = ROOT_DIR / "evaluation" / "dataset.json"
EVAL_RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

# ------------------------------------------------------------------
# Pinecone
# ------------------------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "regforg")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "regforg")
PINECONE_DIMENSION = _int("PINECONE_DIMENSION", 4096)
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ------------------------------------------------------------------
# Ollama
# ------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("BASE_URL", "http://localhost:11434"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0)
# Qwen3 "thinking" mode adds ~13 s per answer with no measurable gain for
# short grounded answers, so it is off by default.
LLM_THINKING = _bool("LLM_THINKING", False)
LLM_NUM_CTX = _int("LLM_NUM_CTX", 8192)
# Chunks are ~250 tokens; the model default (32k) reserves ~10 GB for nothing.
EMBEDDING_NUM_CTX = _int("EMBEDDING_NUM_CTX", 2048)
# Keep both models in memory between questions. Ollama's default (5 min)
# makes the first question after a pause pay a 5-10 s reload.
OLLAMA_KEEP_ALIVE = _int("OLLAMA_KEEP_ALIVE_SECONDS", 1800)

# ------------------------------------------------------------------
# RAG
# ------------------------------------------------------------------

TOP_K = _int("TOP_K", 5)
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 32)
# Cosine score below which a chunk is treated as irrelevant. If no chunk
# clears it the pipeline abstains instead of calling the LLM.
# Calibrated on the eval set: in-domain top scores >= 0.70,
# out-of-domain top scores <= 0.35.
RELEVANCE_THRESHOLD = _float("RELEVANCE_THRESHOLD", 0.50)

NO_ANSWER = "I could not find the answer in the indexed documents."

# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = _int("GRADIO_SERVER_PORT", 7860)
