"""
Evaluate RegForg on evaluation/dataset.json.

    python -m evaluation.evaluate                 # retrieval + relevance gate (fast, no LLM)
    python -m evaluation.evaluate --generate      # also generate answers (citations, refusals)
    python -m evaluation.evaluate --top-k 8 --threshold 0.55

Results are written to evaluation/results/latest.json and shown in the
dashboard's Evaluation tab.
"""

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from config import EVAL_DATASET_PATH, EVAL_RESULTS_DIR, LLM_MODEL, EMBEDDING_MODEL, RELEVANCE_THRESHOLD, TOP_K
from evaluation.metrics import hit_at_k, mean, percentile, recall_at_k, reciprocal_rank
from src.pipeline.rag_pipeline import run_rag
from src.retrieval.retriever import get_retriever

Progress = Optional[Callable[[float, str], None]]


def load_dataset() -> List[Dict[str, Any]]:
    return json.loads(EVAL_DATASET_PATH.read_text())["items"]


def run_evaluation(
    top_k: int = TOP_K,
    min_score: float = RELEVANCE_THRESHOLD,
    generate: bool = False,
    progress: Progress = None,
    save: bool = True,
) -> Dict[str, Any]:

    items = load_dataset()
    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(items, start=1):
        question = item["question"]
        if progress:
            progress((i - 1) / len(items), f"[{i}/{len(items)}] {question}")

        if generate:
            result = run_rag(question, top_k=top_k, min_score=min_score, log=False)
            refused = result["abstained"]
        else:
            result = get_retriever().retrieve(question, top_k=top_k, min_score=min_score)
            result["generation_time"] = None
            refused = not result["relevant"]

        pages = [m["page"] for m in result["matches"]]
        relevant = item["relevant_pages"]

        row = {
            "question": question,
            "answerable": item["answerable"],
            "relevant_pages": relevant,
            "retrieved_pages": pages,
            "top_score": round(result["top_score"], 4),
            "refused": refused,
            "retrieval_time": result["retrieval_time"],
            "generation_time": result.get("generation_time"),
            "total_time": result.get("total_time", result["retrieval_time"]),
        }

        if item["answerable"]:
            row.update(
                hit=hit_at_k(pages, relevant),
                mrr=reciprocal_rank(pages, relevant),
                recall=recall_at_k(pages, relevant),
            )
        if generate:
            row["answer"] = result["answer"]
            row["cited"] = bool(result["cited"])

        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]

    summary: Dict[str, Any] = {
        "hit_at_k": mean(r["hit"] for r in answerable),
        "mrr": mean(r["mrr"] for r in answerable),
        "recall_at_k": mean(r["recall"] for r in answerable),
        "refusal_rate": mean(float(r["refused"]) for r in unanswerable),
        "false_refusal_rate": mean(float(r["refused"]) for r in answerable),
        "retrieval_p50": percentile([r["retrieval_time"] for r in rows], 50),
        "retrieval_p95": percentile([r["retrieval_time"] for r in rows], 95),
    }
    if generate:
        summary.update(
            citation_rate=mean(float(r["cited"]) for r in answered),
            generation_p50=percentile([r["generation_time"] for r in answered], 50),
            generation_p95=percentile([r["generation_time"] for r in answered], 95),
            total_p50=percentile([r["total_time"] for r in rows], 50),
            total_p95=percentile([r["total_time"] for r in rows], 95),
        )

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "top_k": top_k,
            "min_score": min_score,
            "generate": generate,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
        },
        "summary": summary,
        "rows": rows,
    }

    if save:
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (EVAL_RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2))

    if progress:
        progress(1.0, "Evaluation complete")
    return report


def load_latest() -> Optional[Dict[str, Any]]:
    path = EVAL_RESULTS_DIR / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None


def print_report(report: Dict[str, Any]) -> None:
    cfg, s = report["config"], report["summary"]
    print(f"\nRegForg evaluation  top_k={cfg['top_k']}  threshold={cfg['min_score']}  "
          f"({cfg['n_answerable']} answerable, {cfg['n_unanswerable']} out-of-scope)\n")
    labels = {
        "hit_at_k": "Hit@K", "mrr": "MRR", "recall_at_k": "Recall@K",
        "refusal_rate": "Out-of-scope refused", "false_refusal_rate": "False refusals",
        "citation_rate": "Answers with citations",
        "retrieval_p50": "Retrieval p50 (s)", "retrieval_p95": "Retrieval p95 (s)",
        "generation_p50": "Generation p50 (s)", "generation_p95": "Generation p95 (s)",
        "total_p50": "End-to-end p50 (s)", "total_p95": "End-to-end p95 (s)",
    }
    for key, label in labels.items():
        if key in s and s[key] is not None:
            print(f"  {label:<26} {s[key]:.3f}")

    misses = [r for r in report["rows"] if r["answerable"] and not r["hit"]]
    if misses:
        print("\nMisses:")
        for r in misses:
            print(f"  - {r['question']}  expected {r['relevant_pages']} got {r['retrieved_pages']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD)
    parser.add_argument("--generate", action="store_true", help="also run the LLM")
    args = parser.parse_args()

    print_report(run_evaluation(args.top_k, args.threshold, args.generate,
                                progress=lambda _, msg: print(msg)))
