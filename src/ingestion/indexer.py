import time
import uuid
from typing import List

from pinecone import Pinecone, ServerlessSpec 
from langchain_core.documents import Document

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_DIMENSION,
    PINECONE_CLOUD,
    PINECONE_REGION,
)

from src.embeddings.ollama_embeddings import QwenEmbeddings
from src.utils.logger import logger


def get_pinecone_client():
    """Create and return Pinecone client."""

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing. "
            "Add it to your .env file."
        )

    return Pinecone(api_key=PINECONE_API_KEY)


def create_index_if_not_exists():
    """Create Pinecone index if it does not already exist."""

    pc = get_pinecone_client()

    existing_indexes = [
        index.name
        for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME in existing_indexes:

        logger.info(
            f"Pinecone index already exists: "
            f"{PINECONE_INDEX_NAME}"
        )

        return pc.Index(PINECONE_INDEX_NAME)

    logger.info(
        f"Creating Pinecone index: "
        f"{PINECONE_INDEX_NAME}"
    )

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    ) 

    logger.info(
        "Pinecone index created successfully."
    )

    return pc.Index(PINECONE_INDEX_NAME)


def index_documents(
    documents: List[Document],
    batch_size: int = 32,
):
    """Generate embeddings and upload document chunks to Pinecone."""

    start_time = time.perf_counter()

    index = create_index_if_not_exists()

    embeddings = QwenEmbeddings()

    vectors = []

    chunks_indexed = 0

    total_documents = len(documents)

    logger.info(
        f"Starting indexing of "
        f"{total_documents} chunks..."
    )

    for i, document in enumerate(documents):

        text = document.page_content.strip()

        if not text:
            continue

        # Generate Qwen embedding
        vector = embeddings.embed_query(text)

        # Verify embedding dimension
        if len(vector) != PINECONE_DIMENSION:

            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {PINECONE_DIMENSION}, "
                f"got {len(vector)}"
            )

        metadata = {
            "text": text,
            "source": str(
                document.metadata.get(
                    "source",
                    "unknown",
                )
            ),
            "file_name": str(
                document.metadata.get(
                    "file_name",
                    "unknown",
                )
            ),
            "page": int(
                document.metadata.get(
                    "page",
                    0,
                )
            ),
        }

        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": metadata,
            }
        )

        chunks_indexed += 1

        # Upload batch
        if (
            len(vectors) >= batch_size
            or i == total_documents - 1
        ):

            if vectors:

                response = index.upsert(
                    vectors=vectors,
                    namespace=PINECONE_NAMESPACE,
                )

                logger.info(
                    f"Uploaded batch: "
                    f"{response.upserted_count} vectors"
                )

                vectors = []

    elapsed = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        f"Indexing completed in "
        f"{elapsed:.2f} seconds"
    )

    logger.info(
        f"Total chunks indexed: "
        f"{chunks_indexed}"
    )

    return {
        "chunks_indexed": chunks_indexed,
        "indexing_time": elapsed,
    }


if __name__ == "__main__":

    print("=" * 60)
    print("PINECONE RAG INDEXING")
    print("=" * 60)

    try:

        from src.ingestion.loader import load_document
        from src.ingestion.chunker import chunk_documents

        # --------------------------------------------------
        # 1. Load PDF
        # --------------------------------------------------

        print("\n[1/4] Loading PDF...")

        documents = load_document(
            "Data/documents/MachineLearning-3.pdf"
        )

        print(
            f"Pages loaded: {len(documents)}"
        )

        # --------------------------------------------------
        # 2. Create chunks
        # --------------------------------------------------

        print("\n[2/4] Creating chunks...")

        chunks = chunk_documents(documents)

        print(
            f"Chunks created: {len(chunks)}"
        )

        # --------------------------------------------------
        # 3. Generate embeddings + upload
        # --------------------------------------------------

        print(
            "\n[3/4] Generating Qwen embeddings "
            "and uploading to Pinecone..."
        )

        result = index_documents(
            chunks,
            batch_size=16,
        )

        print(
            f"\nChunks indexed: "
            f"{result['chunks_indexed']}"
        )

        print(
            f"Indexing time: "
            f"{result['indexing_time']:.2f} seconds"
        )

        # --------------------------------------------------
        # 4. Verify Pinecone
        # --------------------------------------------------

        print("\n[4/4] Verifying Pinecone...")

        index = create_index_if_not_exists()

        stats = index.describe_index_stats()

        print("\nPinecone statistics:")
        print(stats)

        print("\n" + "=" * 60)
        print("RAG INDEXING COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print("RAG INDEXING FAILED")
        print("=" * 60)

        print(f"Error: {e}")

        raise