import time
from typing import List, Dict, Any

from src.ingestion.indexer import (
    get_pinecone_client,
    create_index_if_not_exists
)

from src.embeddings.ollama_embeddings import (
    get_embeddings
)

from config import (
    PINECONE_NAMESPACE,
    TOP_K
)

from src.utils.logger import logger


def retrieve_documents(
    query: str,
    top_k: int = TOP_K
) -> Dict[str, Any]:

    start_time = time.perf_counter()

    index = create_index_if_not_exists()

    embeddings = get_embeddings()

    logger.info(
        f"Retrieving documents for: {query}"
    )

    query_vector = embeddings.embed_query(
        query
    )

    results = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    retrieval_time = (
        time.perf_counter()
        - start_time
    )

    documents = []

    for match in results.matches:

        metadata = match.metadata or {}

        documents.append(
            {
                "id": match.id,
                "score": float(
                    match.score
                ),
                "text": metadata.get(
                    "text",
                    ""
                ),
                "source": metadata.get(
                    "source",
                    "unknown"
                ),
                "file_name": metadata.get(
                    "file_name",
                    "unknown"
                ),
                "page": metadata.get(
                    "page",
                    0
                )
            }
        )

    logger.info(
        f"Retrieved {len(documents)} documents "
        f"in {retrieval_time:.2f}s"
    )

    return {
        "documents": documents,
        "retrieval_time": retrieval_time
    }