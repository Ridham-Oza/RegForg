import time
import uuid
from typing import List

from pinecone import Pinecone

from langchain_core.documents import Document

import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_DIMENSION,
    PINECONE_CLOUD,
    PINECONE_REGION
)

from src.embeddings.ollama_embeddings import (
       QwenEmbeddings
    
)

from src.utils.logger import logger


def get_pinecone_client():

    if not PINECONE_API_KEY:

        raise ValueError(
            "PINECONE_API_KEY is missing. "
            "Add it to your .env file."
        )

    return Pinecone(
        api_key=PINECONE_API_KEY
    )


def create_index_if_not_exists():

    pc = get_pinecone_client()

    existing_indexes = [
        index.name
        for index in pc.indexes.list()
    ]

    if PINECONE_INDEX_NAME in existing_indexes:

        logger.info(
            f"Pinecone index already exists: "
            f"{PINECONE_INDEX_NAME}"
        )

        return pc.index(
            PINECONE_INDEX_NAME
        )

    logger.info(
        f"Creating Pinecone index: "
        f"{PINECONE_INDEX_NAME}"
    )

    pc.indexes.create(
        name=PINECONE_INDEX_NAME,
        schema={
            "fields": {
                "embedding": {
                    "type": "dense_vector",
                    "dimension": PINECONE_DIMENSION,
                    "metric": "cosine"
                }
            }
        },
        deployment={
            "deployment_type": "managed",
            "cloud": PINECONE_CLOUD,
            "region": PINECONE_REGION
        }
    )

    logger.info(
        "Pinecone index created successfully"
    )

    return pc.index(
        PINECONE_INDEX_NAME
    )


def index_documents(
    documents: List[Document],
    batch_size: int = 32
):

    start_time = time.perf_counter()

    index = create_index_if_not_exists()

    embeddings = QwenEmbeddings()

    vectors = []

    total_documents = len(
        documents
    )

    logger.info(
        f"Indexing {total_documents} chunks"
    )

    for i, document in enumerate(
        documents
    ):

        text = document.page_content.strip()

        if not text:
            continue

        vector = embeddings.embed_query(
            text
        )

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
                    "unknown"
                )
            ),
            "file_name": str(
                document.metadata.get(
                    "file_name",
                    "unknown"
                )
            ),
            "page": int(
                document.metadata.get(
                    "page",
                    0
                )
            )
        }

        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": metadata
            }
        )

        if (
            len(vectors) >= batch_size
            or i == total_documents - 1
        ):

            index.upsert(
                vectors=vectors,
                namespace=PINECONE_NAMESPACE
            )

            logger.info(
                f"Uploaded "
                f"{min(i + 1, total_documents)}"
                f"/{total_documents} chunks"
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

    return {
        "chunks_indexed": total_documents,
        "indexing_time": elapsed
    }