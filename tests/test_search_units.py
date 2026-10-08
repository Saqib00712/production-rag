import numpy as np

from app.services.fusion import reciprocal_rank_fusion
from app.services.search import build_match_query
from app.services.vector_math import cosine_top_k


def test_match_query_quotes_terms_and_drops_stopwords():
    assert build_match_query("What is the refund policy?") == '"refund" OR "policy"'


def test_match_query_neutralizes_fts_syntax():
    q = build_match_query('"; DROP TABLE chunks; -- OR NEAR( a:b')
    assert '"' in q and ";" not in q and "--" not in q and "(" not in q and ":" not in q
    assert build_match_query("muhammad-saqib") == '"muhammad" OR "saqib"'


def test_match_query_empty_cases():
    assert build_match_query("") == ""
    assert build_match_query("what is it?") == ""
    assert build_match_query("!!! ???") == ""


def test_match_query_dedupes_and_caps_terms():
    assert build_match_query("refund refund REFUND") == '"refund"'
    assert build_match_query(" ".join(f"w{i}" for i in range(100))).count("OR") == 31


def test_rrf_rewards_agreement_between_retrievers():
    fused = reciprocal_rank_fusion({"kw": ["a", "b", "c"], "vec": ["b", "d", "a"]}, k=60)
    ids = [c for c, _ in fused]
    assert ids[0] == "b" and ids[1] == "a"          # found by both
    assert set(ids) == {"a", "b", "c", "d"}
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-12


def test_rrf_single_list_keeps_order_and_ties_are_deterministic():
    assert [c for c, _ in reciprocal_rank_fusion({"kw": ["x", "y", "z"]})] == ["x", "y", "z"]
    tie = reciprocal_rank_fusion({"a": ["p"], "b": ["q"]})
    assert [c for c, _ in tie] == ["p", "q"]


def test_cosine_top_k_exact_ordering():
    m = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    res = cosine_top_k([1, 0], ["a", "b", "c"], m, 3)
    assert [c for c, _ in res] == ["a", "c", "b"]
    assert abs(res[0][1] - 1.0) < 1e-6 and abs(res[2][1]) < 1e-6


def test_cosine_edge_cases():
    m = np.array([[0, 0], [1, 0]], dtype=np.float32)
    assert cosine_top_k([0, 0], ["a", "b"], m, 2) == []          # zero query
    assert cosine_top_k([1, 0], [], np.empty((0, 0), dtype=np.float32), 5) == []
    res = cosine_top_k([1, 0], ["zero", "one"], m, 5)            # zero row: no NaN
    assert res[0][0] == "one" and res[1][1] == 0.0
