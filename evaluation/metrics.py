"""
Evaluation metrics. Pure functions, unit-tested in tests/test_metrics.py.

Kept:
  Hit@K          did any relevant page appear in the top K?
  MRR            how high is the first relevant hit?
  Recall@K       share of the relevant pages that were retrieved
  Refusal rate   out-of-scope questions correctly refused
  False refusal  in-scope questions wrongly refused
  Citation rate  answered questions whose answer cites a valid source
  p50 / p95      latency per stage

Removed from the first version:
  Average similarity     raw cosine depends on the model and query phrasing,
                         not on answer quality, so it cannot be compared
                         between configurations.
  End-to-end retrieval   measured the same span as retrieval latency
  latency                (1.7317 s vs 1.7314 s); it added nothing.
  Chunks used            always equal to top_k.
"""

import math
from typing import Iterable, List, Optional, Sequence


def hit_at_k(retrieved: Sequence[int], relevant: Iterable[int]) -> float:
    return 1.0 if set(retrieved) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[int], relevant: Iterable[int]) -> float:
    relevant = set(relevant)
    for rank, page in enumerate(retrieved, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: Sequence[int], relevant: Iterable[int]) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile, q in [0, 100]."""

    data: List[float] = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return data[low] + (data[high] - data[low]) * (position - low)
