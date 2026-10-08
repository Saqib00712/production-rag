import sqlite3

import numpy as np

from app.models.chunk import Chunk
from app.repositories.chunk_repository import ChunkRepository
from app.services.search import build_match_query


def _c(i, text, doc="d1", tenant="default"):
    return Chunk(chunk_id=f"{doc}-{i}", document_id=doc, tenant_id=tenant, chunk_index=i, page_number=i + 1,
                 text=text, token_count=5, content_hash=f"{doc}h{i}")


def _repo(tmp_path):
    return ChunkRepository(str(tmp_path / "s.db"))


def test_bm25_ranks_exact_rare_term_first(tmp_path):
    repo = _repo(tmp_path)
    repo.save_indexed_document("d1", [
        _c(0, "general shipping information for all orders"),
        _c(1, "warranty claims need serial ERR4471 always"),
        _c(2, "shipping shipping shipping faster"),
    ], {}, "m")
    hits = repo.keyword_search(build_match_query("serial ERR4471"), 5, "default")
    assert hits[0][0] == "d1-1" and hits[0][1] > 0          # higher = better
    assert repo.keyword_search(build_match_query("nonexistentword"), 5, "default") == []


def test_stemming_matches_word_variants(tmp_path):
    repo = _repo(tmp_path)
    repo.save_indexed_document("d1", [_c(0, "Refunds are issued quickly")], {}, "m")
    assert repo.keyword_search(build_match_query("refund"), 5, "default")[0][0] == "d1-0"


def test_reindex_replaces_keyword_index(tmp_path):
    repo = _repo(tmp_path)
    repo.save_indexed_document("d1", [_c(0, "old zebra text")], {}, "m")
    repo.save_indexed_document("d1", [_c(0, "fresh giraffe text")], {}, "m")
    assert repo.keyword_search('"zebra"', 5, "default") == []
    assert repo.keyword_search('"giraffe"', 5, "default")


def test_document_filter(tmp_path):
    repo = _repo(tmp_path)
    repo.save_indexed_document("d1", [_c(0, "shared term here", "d1")], {}, "m")
    repo.save_indexed_document("d2", [_c(0, "shared term there", "d2")], {}, "m")
    assert len(repo.keyword_search('"shared"', 10, "default")) == 2
    assert [c for c, _ in repo.keyword_search('"shared"', 10, "default", "d2")] == ["d2-0"]


def test_tenant_filter_isolates_results_even_without_a_document_filter(tmp_path):
    """The actual Day 10 isolation guarantee: a search with NO document_id
    (search across all of 'my' documents) must still never cross tenants."""
    repo = _repo(tmp_path)
    repo.save_indexed_document("d1", [_c(0, "shared term here", "d1", tenant="tenant-a")], {}, "m")
    repo.save_indexed_document("d2", [_c(0, "shared term there", "d2", tenant="tenant-b")], {}, "m")
    assert [c for c, _ in repo.keyword_search('"shared"', 10, "tenant-a")] == ["d1-0"]
    assert [c for c, _ in repo.keyword_search('"shared"', 10, "tenant-b")] == ["d2-0"]
    assert repo.keyword_search('"shared"', 10, "tenant-c") == []  # a third, unrelated tenant sees nothing


def test_keyword_index_heals_when_missing(tmp_path):
    """Simulates the Day 2 -> Day 3 upgrade: chunks exist, FTS rows do not."""
    path = str(tmp_path / "s.db")
    repo = ChunkRepository(path)
    repo.save_indexed_document("d1", [_c(0, "legacy chunk about warranty")], {}, "m")
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM chunks_fts")
    conn.commit(); conn.close()
    healed = ChunkRepository(path)
    assert healed.keyword_search('"warranty"', 5, "default")


def test_tenant_column_migration_defaults_existing_rows(tmp_path):
    """Simulates a pre-Day-10 database: a `chunks` table with no tenant_id
    column at all. Opening it with the current ChunkRepository must ALTER
    TABLE to add the column, defaulting existing rows to 'default' rather
    than crashing or silently losing them."""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
        "chunk_index INTEGER NOT NULL, page_number INTEGER NOT NULL, text TEXT NOT NULL, "
        "token_count INTEGER NOT NULL, content_hash TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO chunks VALUES ('d1-0', 'd1', 0, 1, 'legacy text', 2, 'h0')")
    conn.commit(); conn.close()

    migrated = ChunkRepository(path)
    assert [c for c, _ in migrated.keyword_search('"legacy"', 5, "default")] == ["d1-0"]


def test_load_vectors_shape_and_filter(tmp_path):
    repo = _repo(tmp_path)
    repo.save_indexed_document("d1", [_c(0, "a"), _c(1, "b")], {"d1h0": [1.0, 0.0], "d1h1": [0.0, 1.0]}, "m")
    repo.save_indexed_document("d2", [_c(0, "c", "d2")], {"d2h0": [0.5, 0.5]}, "m")
    ids, matrix = repo.load_vectors("m", "default")
    assert len(ids) == 3 and matrix.shape == (3, 2) and matrix.dtype == np.float32
    ids, matrix = repo.load_vectors("m", "default", "d2")
    assert ids == ["d2-0"]
    assert repo.load_vectors("other-model", "default")[0] == []


def test_load_vectors_respects_tenant_isolation(tmp_path):
    repo = _repo(tmp_path)
    repo.save_indexed_document("d1", [_c(0, "a", "d1", tenant="tenant-a")], {"d1h0": [1.0, 0.0]}, "m")
    repo.save_indexed_document("d2", [_c(0, "b", "d2", tenant="tenant-b")], {"d2h0": [0.0, 1.0]}, "m")
    ids, _ = repo.load_vectors("m", "tenant-a")
    assert ids == ["d1-0"]


def test_query_cache_roundtrip_and_prune(tmp_path):
    repo = _repo(tmp_path)
    for i in range(5):
        repo.save_query_embedding(f"q{i}", "m", [float(i), 0.5], max_rows=3)
    assert repo.get_query_embedding("q4", "m") == [4.0, 0.5]
    assert repo.get_query_embedding("q0", "m") is None      # oldest pruned
    assert repo.get_query_embedding("q4", "other") is None
