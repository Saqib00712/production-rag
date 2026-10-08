"""Unit tests for both VectorIndex backends. HnswIndex tests use small,
exact-recall-friendly parameters so results are deterministic enough to
assert on directly -- at this tiny scale HNSW should find the true nearest
neighbor essentially always, which is itself a useful sanity check that
the wiring (not just the library) is correct."""

import numpy as np
import pytest

from app.services.vector_index import BruteForceIndex, HnswIndex, build_index

hnswlib = pytest.importorskip("hnswlib")


def orthogonal_vectors():
    ids = ["a", "b", "c"]
    matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    return ids, matrix


def test_brute_force_index_matches_cosine_top_k_directly():
    ids, matrix = orthogonal_vectors()
    index = BruteForceIndex()
    index.build(ids, matrix)
    results = index.search([1, 0], k=3)
    assert [r[0] for r in results] == ["a", "c", "b"]  # exact ranking by cosine similarity
    assert abs(results[0][1] - 1.0) < 1e-6


def test_brute_force_index_empty():
    index = BruteForceIndex()
    index.build([], np.empty((0, 0), dtype=np.float32))
    assert index.search([1, 0], k=5) == []
    assert index.size == 0


def test_hnsw_index_finds_exact_match_at_small_scale():
    ids, matrix = orthogonal_vectors()
    index = HnswIndex(ef_construction=200, m=16, ef_search=50)
    index.build(ids, matrix)
    results = index.search([1, 0], k=1)
    assert results[0][0] == "a"
    assert results[0][1] > 0.99  # near-1.0 cosine similarity, allowing float slack


def test_hnsw_index_respects_k():
    ids, matrix = orthogonal_vectors()
    index = HnswIndex()
    index.build(ids, matrix)
    assert len(index.search([1, 0], k=2)) == 2


def test_hnsw_index_k_larger_than_corpus_is_clamped_not_an_error():
    ids, matrix = orthogonal_vectors()
    index = HnswIndex()
    index.build(ids, matrix)
    results = index.search([1, 0], k=100)
    assert len(results) == 3  # clamped to corpus size, no crash


def test_hnsw_index_empty_build_returns_no_results():
    index = HnswIndex()
    index.build([], np.empty((0, 2), dtype=np.float32))
    assert index.search([1, 0], k=5) == []
    assert index.size == 0


def test_hnsw_and_brute_force_agree_on_ranking_at_small_scale():
    rng = np.random.default_rng(42)
    ids = [f"v{i}" for i in range(200)]
    matrix = rng.normal(size=(200, 16)).astype(np.float32)
    query = rng.normal(size=16).tolist()

    bf = BruteForceIndex(); bf.build(ids, matrix)
    hnsw = HnswIndex(ef_construction=400, m=32, ef_search=200)  # generous settings -> near-exact recall
    hnsw.build(ids, matrix)

    bf_top10 = {r[0] for r in bf.search(query, k=10)}
    hnsw_top10 = {r[0] for r in hnsw.search(query, k=10)}
    overlap = len(bf_top10 & hnsw_top10) / 10
    assert overlap >= 0.8  # generous HNSW settings should recover most of the true top-10


def test_build_index_factory_brute_force():
    ids, matrix = orthogonal_vectors()
    index = build_index("brute_force", ids, matrix)
    assert isinstance(index, BruteForceIndex) and index.size == 3


def test_build_index_factory_hnsw():
    ids, matrix = orthogonal_vectors()
    index = build_index("hnsw", ids, matrix, ef_construction=200, m=16, ef_search=50)
    assert isinstance(index, HnswIndex) and index.size == 3


def test_build_index_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_index("quantum", [], np.empty((0, 0)))
