"""
Grounded answer generation with Qwen3 via Ollama.

Changes from the first version:
  * model, URL and temperature come from config (it was hard-coded);
  * Qwen3 thinking mode is disabled by default: measured 15 s -> 2.3 s per
    answer with the same output length;
  * the model is told to cite sources as [1], [2] … so every claim can be
    traced back to a page;
  * answers can be streamed token by token.
"""

from functools import lru_cache
from typing import Any, Dict, Iterator, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import (
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    LLM_THINKING,
    NO_ANSWER,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
)

SYSTEM_PROMPT = f"""You are RegForg, a question-answering assistant for a private document collection.

Rules:
1. Answer ONLY from the numbered sources provided. Never use outside knowledge.
2. Cite the sources you used with their numbers in square brackets, e.g. [1] or [2][3], right after the sentence they support.
3. If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
4. Be clear and concise. Use short paragraphs or bullet points. Keep any formulas from the sources."""


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
        reasoning=LLM_THINKING,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )


def build_context(documents: List[Dict[str, Any]]) -> str:
    """Number each source so the model can cite it."""

    return "\n\n".join(
        f"[{i}] (file: {d['file_name']}, page {d['page']})\n{d['text'].strip()}"
        for i, d in enumerate(documents, start=1)
        if d.get("text", "").strip()
    )


def build_messages(question: str, documents: List[Dict[str, Any]]):
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"SOURCES:\n\n{build_context(documents)}\n\nQUESTION: {question}"),
    ]


def stream_answer(question: str, documents: List[Dict[str, Any]]) -> Iterator[str]:
    """Yield answer tokens as they are generated."""

    for chunk in get_llm().stream(build_messages(question, documents)):
        if chunk.content:
            yield str(chunk.content)


def generate_answer(question: str, documents: List[Dict[str, Any]]) -> str:
    return "".join(stream_answer(question, documents)).strip()
