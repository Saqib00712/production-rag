from app.models.chunk import Chunk
from app.repositories.chunk_repository import ChunkRepository


def _chunk(i, doc="d1", h=None):
    return Chunk(chunk_id=f"{doc}-{i}", document_id=doc, chunk_index=i, page_number=1,
                 text=f"text {i}", token_count=2, content_hash=h or f"h{i}")


def test_save_list_and_embedding_flags(tmp_path):
    repo = ChunkRepository(str(tmp_path / "t.db"))
    repo.save_indexed_document("d1", [_chunk(0), _chunk(1)], {"h0": [0.5, 0.25]}, "m")
    total, chunks = repo.list_chunks("d1", "m", 10, 0, "default")
    assert total == 2
    assert chunks[0].has_embedding and chunks[0].embedding_dim == 2
    assert not chunks[1].has_embedding


def test_embedding_roundtrip_and_model_isolation(tmp_path):
    repo = ChunkRepository(str(tmp_path / "t.db"))
    repo.save_indexed_document("d1", [_chunk(0)], {"h0": [0.5, 0.25]}, "m1")
    assert repo.get_cached_embeddings(["h0"], "m1") == {"h0": [0.5, 0.25]}
    assert repo.get_cached_embeddings(["h0"], "m2") == {}


def test_reindex_replaces_chunks(tmp_path):
    repo = ChunkRepository(str(tmp_path / "t.db"))
    repo.save_indexed_document("d1", [_chunk(0), _chunk(1)], {}, "m")
    repo.save_indexed_document("d1", [_chunk(0)], {}, "m")
    assert repo.list_chunks("d1", "m", 10, 0, "default")[0] == 1
