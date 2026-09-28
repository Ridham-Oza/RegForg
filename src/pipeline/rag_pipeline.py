import time
from typing import Dict, Any

from src.retrieval.retriever import (
    retrieve_documents
)

from src.generation.llm import (
    generate_answer
)

from src.utils.logger import logger


def build_context(
    documents
):

    context_parts = []

    for i, document in enumerate(
        documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

File:
{document["file_name"]}

Page:
{document["page"]}

Relevance Score:
{document["score"]}

Content:
{document["text"]}
"""
        )

    return "\n\n".join(
        context_parts
    )


def run_rag(
    question: str,
    top_k: int = 5
) -> Dict[str, Any]:

    total_start = time.perf_counter()

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    retrieval_result = retrieve_documents(
        question,
        top_k=top_k
    )

    documents = retrieval_result[
        "documents"
    ]

    retrieval_time = retrieval_result[
        "retrieval_time"
    ]

    # -----------------------------------------------------
    # Context
    # -----------------------------------------------------

    context = build_context(
        documents
    )

    # -----------------------------------------------------
    # Generation
    # -----------------------------------------------------

    generation_start = time.perf_counter()

    answer = generate_answer(
        question,
        context
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    # -----------------------------------------------------
    # Total time
    # -----------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    logger.info(
        f"Retrieval time: "
        f"{retrieval_time:.3f}s"
    )

    logger.info(
        f"Generation time: "
        f"{generation_time:.3f}s"
    )

    logger.info(
        f"Total time: "
        f"{total_time:.3f}s"
    )

    return {
        "answer": answer,
        "documents": documents,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "num_chunks": len(documents)
    }