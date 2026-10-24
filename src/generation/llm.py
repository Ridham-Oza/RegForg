import time
from typing import List, Dict, Any

from langchain_ollama import ChatOllama

from src.retrieval.retriever import PineconeRetriever
from src.utils.logger import logger


# ============================================================
# LLM CONFIGURATION
# ============================================================

LLM_MODEL = "qwen3:8b"


# ============================================================
# CREATE LLM
# ============================================================

def get_llm():
    """
    Create and return the local Qwen3 8B LLM.
    """

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    documents: List[Dict[str, Any]]
) -> str:
    """
    Combine retrieved documents into a context string.
    """

    context_parts = []

    for i, document in enumerate(documents):

        text = document.get("text", "").strip()

        if not text:
            continue

        page = document.get("page", "unknown")
        file_name = document.get(
            "file_name",
            "unknown"
        )

        context_parts.append(
            f"""
SOURCE {i + 1}
File: {file_name}
Page: {page}

{text}
""".strip()
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate an answer using retrieved context.
    """

    start_time = time.perf_counter()

    context = build_context(documents)

    if not context:
        return {
            "answer": (
                "I could not find relevant information "
                "in the provided document."
            ),
            "generation_time": 0.0,
        }

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I could not find the answer in the provided document."

Keep the answer clear, accurate, and concise.

Always base the answer on the retrieved sources.

================ CONTEXT ================

{context}

================ QUESTION ================

{query}

================ ANSWER ================
"""

    logger.info(
        f"Generating answer using {LLM_MODEL}"
    )

    llm = get_llm()

    response = llm.invoke(prompt)

    answer = response.content

    generation_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        f"Answer generated in "
        f"{generation_time:.2f}s"
    )

    return {
        "answer": answer,
        "generation_time": generation_time,
    }


# ============================================================
# COMPLETE RAG GENERATION TEST
# ============================================================

def answer_question(
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Complete retrieval + generation flow.
    """

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. RETRIEVAL
    # --------------------------------------------------------

    logger.info(
        f"Starting RAG question: {query}"
    )

    retriever = PineconeRetriever(
        top_k=top_k
    )

    retrieval_result = retriever.retrieve(
        query
    )

    documents = retrieval_result[
        "matches"
    ]

    # --------------------------------------------------------
    # 2. GENERATION
    # --------------------------------------------------------

    generation_result = generate_answer(
        query=query,
        documents=documents
    )

    # --------------------------------------------------------
    # 3. TOTAL TIME
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    logger.info(
        f"Total RAG time: "
        f"{total_time:.2f}s"
    )

    return {
        "query": query,

        "answer": generation_result[
            "answer"
        ],

        "matches": documents,

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
    print("RAG GENERATION TEST")
    print("=" * 70)

    question = "What is machine learning?"

    result = answer_question(
        question,
        top_k=5
    )

    print("\nQUESTION:")
    print(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("TIMING")
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

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for i, match in enumerate(
        result["matches"]
    ):

        print(
            f"\n[{i + 1}] "
            f"Score: {match['score']:.6f} | "
            f"Page: {match['page']} | "
            f"File: {match['file_name']}"
        )

    print("\n" + "=" * 70)
    print("RAG TEST COMPLETED")
    print("=" * 70)