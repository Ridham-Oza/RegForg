"""Pipeline behaviour with retrieval and the LLM stubbed out (no network)."""

from config import NO_ANSWER
from src.pipeline import rag_pipeline


class FakeRetriever:
    def __init__(self, score):
        self.score = score

    def retrieve(self, query, top_k, min_score):
        matches = [{"rank": 1, "id": "x", "score": self.score, "text": "SVMs find a hyperplane.", "file_name": "b.pdf", "page": 38}]
        relevant = [m for m in matches if m["score"] >= min_score]
        return {"query": query, "matches": matches, "relevant": relevant, "top_score": self.score,
                "embedding_time": 0.01, "search_time": 0.02, "retrieval_time": 0.03}


def run(monkeypatch, score, answer_tokens=("An SVM finds a hyperplane ", "[1]", ".")):
    calls = []

    def fake_stream(question, sources):
        calls.append(sources)
        yield from answer_tokens

    monkeypatch.setattr(rag_pipeline, "get_retriever", lambda: FakeRetriever(score))
    monkeypatch.setattr(rag_pipeline, "stream_answer", fake_stream)
    return rag_pipeline.run_rag("What is an SVM?", top_k=5, min_score=0.5, log=False), calls


def test_off_topic_question_is_refused_without_calling_the_llm(monkeypatch):
    result, calls = run(monkeypatch, score=0.2)
    assert result["answer"] == NO_ANSWER
    assert result["abstained"] and result["gated"]
    assert calls == []


def test_relevant_question_is_answered_with_citations(monkeypatch):
    result, calls = run(monkeypatch, score=0.8)
    assert len(calls) == 1
    assert result["answer"] == "An SVM finds a hyperplane [1]."
    assert result["cited"] == [1]
    assert not result["abstained"]


def test_model_refusal_is_detected(monkeypatch):
    result, _ = run(monkeypatch, score=0.8, answer_tokens=(NO_ANSWER,))
    assert result["abstained"] and not result["gated"]


def test_invalid_citation_numbers_are_ignored():
    assert rag_pipeline.cited_sources("see [1], [3] and [9]", n_sources=3) == [1, 3]
