from app.eval.retrieval_metrics import compute_page_ranks, hit_at_k, reciprocal_rank, recall_at_k


def test_compute_page_ranks_finds_earliest_occurrence():
    ranks = compute_page_ranks([5, 2, 1, 2, 3], expected_pages=[2, 3])
    assert ranks == {2: 2, 3: 5}


def test_compute_page_ranks_missing_expected_page_is_absent():
    assert compute_page_ranks([1, 2, 3], expected_pages=[9]) == {}


def test_compute_page_ranks_empty_hits():
    assert compute_page_ranks([], expected_pages=[1]) == {}


def test_hit_at_k_true_and_false():
    ranks = {2: 4}
    assert hit_at_k(ranks, 5) is True
    assert hit_at_k(ranks, 3) is False
    assert hit_at_k({}, 5) is False


def test_reciprocal_rank_uses_best_rank_among_multiple_expected():
    assert reciprocal_rank({2: 5, 3: 1}) == 1.0
    assert reciprocal_rank({2: 4}) == 0.25
    assert reciprocal_rank({}) == 0.0


def test_recall_at_k_fraction_of_expected_found():
    assert recall_at_k({1: 1, 2: 2}, expected_pages=[1, 2], k=5) == 1.0
    assert recall_at_k({1: 1}, expected_pages=[1, 2], k=5) == 0.5
    assert recall_at_k({1: 9}, expected_pages=[1, 2], k=5) == 0.0  # found but beyond k
    assert recall_at_k({}, expected_pages=[], k=5) == 0.0  # unanswerable: scored elsewhere, not here


def test_recall_at_k_respects_k_boundary():
    assert recall_at_k({1: 5}, expected_pages=[1], k=5) == 1.0
    assert recall_at_k({1: 6}, expected_pages=[1], k=5) == 0.0
