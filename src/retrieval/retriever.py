import time
from typing import Any, Dict, List, Optional

from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_DIMENSION,
)

from src.embeddings.ollama_embeddings import QwenEmbeddings
from src.utils.logger import logger


class PineconeRetriever:

    def __init__(
        self,
        top_k: int = 5,
        namespace: str = PINECONE_NAMESPACE,
    ):
        self.top_k = int(top_k)
        self.namespace = namespace

        if not PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is missing."
            )

        self.pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        self.index = self.pc.Index(
            PINECONE_INDEX_NAME
        )

        self.embeddings = QwenEmbeddings()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the most relevant chunks from Pinecone.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        search_start = time.perf_counter()

        # --------------------------------------------------
        # 1. Generate query embedding
        # --------------------------------------------------

        embedding_start = time.perf_counter()

        query_vector = self.embeddings.embed_query(
            query
        )

        embedding_time = (
            time.perf_counter()
            - embedding_start
        )

        # --------------------------------------------------
        # 2. Validate embedding dimension
        # --------------------------------------------------

        if len(query_vector) != PINECONE_DIMENSION:

            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {PINECONE_DIMENSION}, "
                f"got {len(query_vector)}"
            )

        # --------------------------------------------------
        # 3. Make sure top_k is always an integer
        # --------------------------------------------------

        effective_top_k = (
            self.top_k
            if top_k is None
            else int(top_k)
        )

        # --------------------------------------------------
        # 4. Query Pinecone
        # --------------------------------------------------

        pinecone_start = time.perf_counter()

        results = self.index.query(
            vector=query_vector,
            top_k=effective_top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        pinecone_time = (
            time.perf_counter()
            - pinecone_start
        )

        retrieval_time = (
            time.perf_counter()
            - search_start
        )

        # --------------------------------------------------
        # 5. Convert Pinecone results
        # --------------------------------------------------

        matches: List[Dict[str, Any]] = []

        # getattr prevents Pylance from complaining
        # about Pinecone's dynamically typed response.

        raw_matches = getattr(
            results,
            "matches",
            []
        )

        if raw_matches is None:
            raw_matches = []

        for match in raw_matches:

            metadata = getattr(
                match,
                "metadata",
                None
            ) or {}

            matches.append(
                {
                    "id": str(
                        getattr(
                            match,
                            "id",
                            ""
                        )
                    ),

                    "score": float(
                        getattr(
                            match,
                            "score",
                            0.0
                        )
                    ),

                    "text": str(
                        metadata.get(
                            "text",
                            ""
                        )
                    ),

                    "source": str(
                        metadata.get(
                            "source",
                            "unknown"
                        )
                    ),

                    "file_name": str(
                        metadata.get(
                            "file_name",
                            "unknown"
                        )
                    ),

                    "page": int(
                        metadata.get(
                            "page",
                            0
                        )
                    ),
                }
            )

        logger.info(
            f"Retrieved {len(matches)} chunks "
            f"in {retrieval_time:.4f} seconds"
        )

        # --------------------------------------------------
        # 6. Return retrieval result
        # --------------------------------------------------

        return {
            "query": query,

            "matches": matches,

            "retrieval_time": retrieval_time,

            "embedding_time": embedding_time,

            "pinecone_time": pinecone_time,

            "top_k": effective_top_k,
        }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PINECONE RETRIEVER TEST")
    print("=" * 60)

    try:

        retriever = PineconeRetriever(
            top_k=5
        )

        query = (
            "What is machine learning and how is it defined?"
        )

        print(
            f"\nQuery: {query}"
        )

        result = retriever.retrieve(
            query
        )

        print(
            "\nRetrieved chunks:"
        )

        print(
            f"Count: {len(result['matches'])}"
        )

        print(
            f"Embedding time: "
            f"{result['embedding_time']:.4f} seconds"
        )

        print(
            f"Pinecone time: "
            f"{result['pinecone_time']:.4f} seconds"
        )

        print(
            f"Total retrieval time: "
            f"{result['retrieval_time']:.4f} seconds"
        )

        print("\nResults:")
        print("-" * 60)

        for i, match in enumerate(
            result["matches"],
            start=1
        ):

            print(
                f"\n[{i}] "
                f"Score: {match['score']:.4f}"
            )

            print(
                f"Source: "
                f"{match['file_name']}"
            )

            print(
                f"Page: "
                f"{match['page']}"
            )

            print(
                f"Text: "
                f"{match['text'][:300]}"
            )

        print("\n" + "=" * 60)
        print(
            "✅ RETRIEVER TEST SUCCESSFUL"
        )
        print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print(
            "❌ RETRIEVER TEST FAILED"
        )
        print("=" * 60)

        print(
            f"Error: {e}"
        )

        raise