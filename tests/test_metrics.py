import pytest

from skrag.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank

A, B, C, D = "a", "b", "c", "d"


def test_hit_at_k_looks_only_at_the_first_k_chunks():
    retrieved = [C, D, A]
    assert not hit_at_k(retrieved, [A], 1)
    assert not hit_at_k(retrieved, [A], 2)
    assert hit_at_k(retrieved, [A], 3)


def test_any_expected_source_gives_a_hit():
    assert hit_at_k([C, B], [A, B], 5)


def test_no_retrieved_chunks_is_a_miss():
    assert not hit_at_k([], [A], 5)
    assert reciprocal_rank([], [A]) == 0.0


def test_reciprocal_rank_is_one_over_the_first_matching_chunk():
    assert reciprocal_rank([A, C], [A]) == 1.0
    assert reciprocal_rank([C, D, B], [A, B]) == pytest.approx(1 / 3)
    assert reciprocal_rank([C, D], [A]) == 0.0


def test_a_repeated_document_still_uses_chunk_rank():
    assert reciprocal_rank([C, C, A], [A]) == pytest.approx(1 / 3)


def test_alternatives_give_full_recall_from_any_one_source():
    assert recall_at_k([A, C], [A, B], 5, need_all=False) == 1.0
    assert recall_at_k([C, D], [A, B], 5, need_all=False) == 0.0


def test_need_all_recall_is_the_share_of_sources_found():
    assert recall_at_k([A, C], [A, B], 5, need_all=True) == 0.5
    assert recall_at_k([A, B], [A, B], 5, need_all=True) == 1.0
    assert recall_at_k([A, A, A], [A, B], 5, need_all=True) == 0.5


def test_recall_ignores_chunks_beyond_k():
    assert recall_at_k([C, D, A], [A], 2, need_all=False) == 0.0


def test_mean_of_nothing_is_zero():
    assert mean([]) == 0.0
    assert mean([1, 0, 1, 1]) == 0.75
