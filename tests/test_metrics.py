import pytest

from evaluation.metrics import hit_at_k, percentile, recall_at_k, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k([3, 4, 5], [5]) == 1.0
    assert hit_at_k([3, 4], [9]) == 0.0


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank([1, 2, 7, 7], [7]) == pytest.approx(1 / 3)
    assert reciprocal_rank([1, 2], [7]) == 0.0


def test_recall_counts_unique_pages():
    assert recall_at_k([7, 7, 8], [7, 8, 9, 10]) == 0.5
    assert recall_at_k([1], []) == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([], 50) is None
    assert percentile([2.0, None], 95) == 2.0
