"""Qwen3 embedding client served by a local Ollama instance."""

from functools import lru_cache
from typing import List

from langchain_ollama import OllamaEmbeddings

from config import EMBEDDING_MODEL, EMBEDDING_NUM_CTX, OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE


class QwenEmbeddings:

    def __init__(self, model_name: str = EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self._client = OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
            num_ctx=EMBEDDING_NUM_CTX,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        # Repeated questions (eval runs, retries, the inspector tab) skip the model.
        self.embed_query = lru_cache(maxsize=512)(self._embed_query)

    def _embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        return self._client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._client.embed_documents(texts)


@lru_cache(maxsize=1)
def get_embeddings() -> QwenEmbeddings:
    return QwenEmbeddings()


if __name__ == "__main__":
    vector = get_embeddings().embed_query("What is machine learning?")
    print(f"Model: {EMBEDDING_MODEL}  Dimension: {len(vector)}")
