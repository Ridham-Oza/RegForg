"""
The single RAG pipeline used by the CLI, the dashboard and the evaluation.

    question -> embed -> Pinecone top-k -> relevance gate -> Qwen3 -> cited answer

If no retrieved chunk clears RELEVANCE_THRESHOLD the pipeline abstains and
skips the LLM call: off-topic questions are refused instantly instead of
being answered from irrelevant context.
"""

import re
import time
from typing import Any, Dict, Iterator, Optional

from config import NO_ANSWER, RELEVANCE_THRESHOLD, TOP_K
from src.generation.llm import stream_answer
from src.retrieval.retriever import get_retriever
from src.utils.query_log import log_query

CITATION = re.compile(r"\[(\d+)\]")


def cited_sources(answer: str, n_sources: int) -> list:
    """Source numbers the answer actually cites (only valid ones)."""
    return sorted({int(n) for n in CITATION.findall(answer) if 1 <= int(n) <= n_sources})


def stream_rag(
    question: str,
    top_k: int = TOP_K,
    min_score: Optional[float] = RELEVANCE_THRESHOLD,
    log: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Yield progress events:
        {"type": "retrieval", ...retrieval result}
        {"type": "token", "text": "..."}            (zero or more)
        {"type": "done", ...final result}
    """

    total_start = time.perf_counter()
    retrieval = get_retriever().retrieve(question, top_k=top_k, min_score=min_score)
    yield {"type": "retrieval", **retrieval}

    sources = retrieval["relevant"]
    answer, generation_time, first_token_time = "", 0.0, None

    if sources:
        generation_start = time.perf_counter()
        for token in stream_answer(question, sources):
            if first_token_time is None:
                first_token_time = time.perf_counter() - total_start
            answer += token
            yield {"type": "token", "text": token}
        answer = answer.strip()
        generation_time = time.perf_counter() - generation_start
    else:
        answer = NO_ANSWER
        yield {"type": "token", "text": answer}

    result = {
        "type": "done",
        "query": question,
        "answer": answer,
        "sources": sources,
        "matches": retrieval["matches"],
        "abstained": not sources or NO_ANSWER.lower() in answer.lower(),
        "gated": not sources,
        "cited": cited_sources(answer, len(sources)),
        "top_score": retrieval["top_score"],
        "embedding_time": retrieval["embedding_time"],
        "search_time": retrieval["search_time"],
        "retrieval_time": retrieval["retrieval_time"],
        "time_to_first_token": first_token_time,
        "generation_time": generation_time,
        "total_time": time.perf_counter() - total_start,
        "top_k": top_k,
        "min_score": min_score,
    }

    if log:
        log_query(result)

    yield result


def run_rag(question: str, top_k: int = TOP_K, min_score: Optional[float] = RELEVANCE_THRESHOLD, log: bool = True) -> Dict[str, Any]:
    """Blocking version of stream_rag: returns the final result."""

    result: Dict[str, Any] = {}
    for event in stream_rag(question, top_k=top_k, min_score=min_score, log=log):
        result = event
    return result


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is machine learning?"
    for event in stream_rag(question, log=False):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            print("\n")
            for i, s in enumerate(event["sources"], start=1):
                print(f"[{i}] p.{s['page']}  score {s['score']:.3f}")
            print(
                f"\nretrieval {event['retrieval_time']:.2f}s | "
                f"generation {event['generation_time']:.2f}s | total {event['total_time']:.2f}s"
            )
