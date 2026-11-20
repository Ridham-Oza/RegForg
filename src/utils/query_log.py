"""Append-only JSONL log of every question, used by the Monitoring tab."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from config import QUERY_LOG_PATH

LOGGED_FIELDS = (
    "query", "abstained", "gated", "top_score", "top_k", "min_score",
    "embedding_time", "search_time", "retrieval_time",
    "time_to_first_token", "generation_time", "total_time",
)


def log_query(result: Dict[str, Any]) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record.update({k: result.get(k) for k in LOGGED_FIELDS})
    record["n_sources"] = len(result.get("sources", []))
    record["n_cited"] = len(result.get("cited", []))
    record["pages"] = [s["page"] for s in result.get("sources", [])]

    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUERY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_queries(limit: int = 500) -> List[Dict[str, Any]]:
    if not QUERY_LOG_PATH.exists():
        return []
    lines = QUERY_LOG_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(line) for line in lines if line.strip()]


def clear_queries() -> None:
    if QUERY_LOG_PATH.exists():
        QUERY_LOG_PATH.unlink()
