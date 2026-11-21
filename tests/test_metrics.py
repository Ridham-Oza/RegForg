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


def test_same_page_in_another_document_is_not_a_hit():
    from evaluation.evaluate import match_keys

    item = {"document": "Learning_Python-2.pdf", "relevant_pages": [59]}
    matches = [{"file_name": "MachineLearning-3.pdf", "page": 59},
               {"file_name": "Learning_Python-2.pdf", "page": 60}]
    retrieved, relevant = match_keys(item, matches)
    assert hit_at_k(retrieved, relevant) == 0.0

    matches.append({"file_name": "Learning_Python-2.pdf", "page": 59})
    retrieved, relevant = match_keys(item, matches)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_items_without_document_match_on_page_only():
    from evaluation.evaluate import match_keys

    retrieved, relevant = match_keys({"relevant_pages": [7]}, [{"file_name": "x.pdf", "page": 7}])
    assert hit_at_k(retrieved, relevant) == 1.0
