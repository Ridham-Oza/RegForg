import time
from typing import Dict, Any

from src.retrieval.retriever import PineconeRetriever
from src.generation.llm import generate_answer
from src.utils.logger import logger


# ============================================================
# RAG PIPELINE
# ============================================================

def run_rag(
    question: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Complete RAG pipeline.

    Flow:

    Question
        ↓
    Qwen Embedding
        ↓
    Pinecone Retrieval
        ↓
    Retrieved Documents
        ↓
    Context
        ↓
    Qwen3 8B
        ↓
    Final Answer
    """

    total_start = time.perf_counter()

    # ========================================================
    # 1. RETRIEVAL
    # ========================================================

    logger.info(
        f"Starting RAG pipeline for: {question}"
    )

    retriever = PineconeRetriever(
        top_k=top_k
    )

    retrieval_result = retriever.retrieve(
        question
    )

    documents = retrieval_result[
        "matches"
    ]

    # ========================================================
    # 2. GENERATION
    # ========================================================

    generation_result = generate_answer(
        query=question,
        documents=documents
    )

    # ========================================================
    # 3. TIMING
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # 4. LOGGING
    # ========================================================

    logger.info(
        f"Retrieval time: "
        f"{retrieval_result['retrieval_time']:.3f}s"
    )

    logger.info(
        f"Generation time: "
        f"{generation_result['generation_time']:.3f}s"
    )

    logger.info(
        f"Total RAG time: "
        f"{total_time:.3f}s"
    )

    # ========================================================
    # 5. FINAL RESULT
    # ========================================================

    return {
        "query": question,

        "answer": generation_result[
            "answer"
        ],

        "documents": documents,

        "num_chunks": len(
            documents
        ),

        "embedding_time": retrieval_result[
            "embedding_time"
        ],

        "pinecone_time": retrieval_result[
            "pinecone_time"
        ],

        "retrieval_time": retrieval_result[
            "retrieval_time"
        ],

        "generation_time": generation_result[
            "generation_time"
        ],

        "total_time": total_time,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("COMPLETE RAG PIPELINE TEST")
    print("=" * 70)

    question = "What is machine learning?"

    result = run_rag(
        question=question,
        top_k=5
    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    print("\nQUESTION:")
    print(result["query"])

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAG METRICS")
    print("=" * 70)

    print(
        f"Embedding time : "
        f"{result['embedding_time']:.3f} sec"
    )

    print(
        f"Pinecone time  : "
        f"{result['pinecone_time']:.3f} sec"
    )

    print(
        f"Retrieval time : "
        f"{result['retrieval_time']:.3f} sec"
    )

    print(
        f"Generation time: "
        f"{result['generation_time']:.3f} sec"
    )

    print(
        f"Total time     : "
        f"{result['total_time']:.3f} sec"
    )

    print(
        f"Chunks used    : "
        f"{result['num_chunks']}"
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for i, document in enumerate(
        result["documents"],
        start=1
    ):

        print(
            f"\n[{i}] "
            f"Score: {document['score']:.6f} | "
            f"Page: {document['page']} | "
            f"File: {document['file_name']}"
        )

        print(
            document["text"][:300]
            .replace("\n", " ")
        )

    print("\n" + "=" * 70)
    print("COMPLETE RAG PIPELINE SUCCESSFUL")
    print("=" * 70)
    
