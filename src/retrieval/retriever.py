"""
Dense retrieval from Pinecone.

The retriever is created once per process (`get_retriever`). The first version
built a new Pinecone client and embedding client on every question.
"""

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config import PINECONE_DIMENSION, PINECONE_NAMESPACE, RELEVANCE_THRESHOLD, TOP_K
from src.embeddings.ollama_embeddings import get_embeddings
from src.ingestion.indexer import get_index
from src.utils.logger import logger


class PineconeRetriever:

    def __init__(self, namespace: str = PINECONE_NAMESPACE):
        self.namespace = namespace
        self.index = get_index()
        self.embeddings = get_embeddings()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: Optional[float] = RELEVANCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Returns:
            matches         all top-k hits, best first
            relevant        hits with score >= min_score (what goes to the LLM)
            top_score       best cosine score (0 if nothing came back)
            embedding_time, search_time, retrieval_time   seconds
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        start = time.perf_counter()
        vector = self.embeddings.embed_query(query.strip())
        embedding_time = time.perf_counter() - start

        if len(vector) != PINECONE_DIMENSION:
            raise ValueError(
                f"Embedding dimension {len(vector)} != PINECONE_DIMENSION {PINECONE_DIMENSION}"
            )

        search_start = time.perf_counter()
        response = self.index.query(
            vector=vector,
            top_k=int(top_k),
            namespace=self.namespace,
            include_metadata=True,
        )
        search_time = time.perf_counter() - search_start

        matches: List[Dict[str, Any]] = []
        for rank, match in enumerate(getattr(response, "matches", None) or [], start=1):
            metadata = match.metadata or {}
            matches.append(
                {
                    "rank": rank,
                    "id": match.id,
                    "score": float(match.score),
                    "text": str(metadata.get("text", "")),
                    "file_name": str(metadata.get("file_name", "unknown")),
                    "page": int(metadata.get("page", 0)),
                }
            )

        threshold = min_score if min_score is not None else float("-inf")
        relevant = [m for m in matches if m["score"] >= threshold]

        result = {
            "query": query,
            "matches": matches,
            "relevant": relevant,
            "top_score": matches[0]["score"] if matches else 0.0,
            "embedding_time": embedding_time,
            "search_time": search_time,
            "retrieval_time": time.perf_counter() - start,
        }

        logger.info(
            f"Retrieved {len(matches)} ({len(relevant)} relevant, top={result['top_score']:.3f}) "
            f"in {result['retrieval_time']:.3f}s"
        )
        return result


@lru_cache(maxsize=1)
def get_retriever() -> PineconeRetriever:
    return PineconeRetriever()


if __name__ == "__main__":
    result = get_retriever().retrieve("What is machine learning?")
    for m in result["matches"]:
        print(f"[{m['rank']}] {m['score']:.3f}  p.{m['page']}  {m['text'][:100]}")
