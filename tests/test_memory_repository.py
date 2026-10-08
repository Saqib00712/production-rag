import sqlite3

from app.repositories.memory_repository import MemoryRepository


def _repo(tmp_path) -> MemoryRepository:
    return MemoryRepository(str(tmp_path / "memories.db"))


def test_add_and_list_memories_round_trip(tmp_path):
    repo = _repo(tmp_path)
    repo.add_memory("tenant-a", "Prefers answers in metric units.")
    repo.add_memory("tenant-a", "Works in procurement.")
    items = repo.list_memories("tenant-a")
    assert [m.content for m in items] == ["Prefers answers in metric units.", "Works in procurement."]


def test_list_memories_is_scoped_per_tenant(tmp_path):
    repo = _repo(tmp_path)
    repo.add_memory("tenant-a", "fact about a")
    repo.add_memory("tenant-b", "fact about b")
    assert [m.content for m in repo.list_memories("tenant-a")] == ["fact about a"]
    assert [m.content for m in repo.list_memories("tenant-b")] == ["fact about b"]


def test_adding_the_same_fact_twice_does_not_duplicate_it(tmp_path):
    repo = _repo(tmp_path)
    first = repo.add_memory("tenant-a", "Prefers metric units.")
    second = repo.add_memory("tenant-a", "  prefers metric units.  ")  # case/whitespace-insensitive
    assert first == second
    assert len(repo.list_memories("tenant-a")) == 1


def test_get_owner_returns_none_for_unknown_memory(tmp_path):
    repo = _repo(tmp_path)
    assert repo.get_owner("does-not-exist") is None


def test_get_owner_returns_the_owning_tenant(tmp_path):
    repo = _repo(tmp_path)
    mid = repo.add_memory("tenant-a", "a fact")
    assert repo.get_owner(mid) == "tenant-a"


def test_delete_memory_removes_it(tmp_path):
    repo = _repo(tmp_path)
    mid = repo.add_memory("tenant-a", "a fact")
    repo.delete_memory(mid)
    assert repo.get_owner(mid) is None
    assert repo.list_memories("tenant-a") == []


def test_max_memories_evicts_the_oldest_fact_first(tmp_path):
    repo = _repo(tmp_path)
    repo.add_memory("tenant-a", "fact 1", max_memories=2)
    repo.add_memory("tenant-a", "fact 2", max_memories=2)
    repo.add_memory("tenant-a", "fact 3", max_memories=2)
    items = repo.list_memories("tenant-a")
    assert [m.content for m in items] == ["fact 2", "fact 3"]


def test_list_memories_limit_returns_the_most_recent(tmp_path):
    repo = _repo(tmp_path)
    for i in range(5):
        repo.add_memory("tenant-a", f"fact {i}")
    items = repo.list_memories("tenant-a", limit=2)
    assert [m.content for m in items] == ["fact 3", "fact 4"]


# --- Day 15: vector-indexed retrieval ---

def test_search_memories_ranks_by_cosine_similarity(tmp_path):
    repo = _repo(tmp_path)
    repo.add_memory("tenant-a", "metric fact", embedding=[1.0, 0.0])
    repo.add_memory("tenant-a", "unrelated fact", embedding=[0.0, 1.0])
    results = repo.search_memories("tenant-a", query_embedding=[1.0, 0.0], top_k=1)
    assert [m.content for m in results] == ["metric fact"]


def test_search_memories_excludes_facts_with_no_embedding(tmp_path):
    repo = _repo(tmp_path)
    repo.add_memory("tenant-a", "no embedding yet")  # e.g. embedding failed when it was stored
    repo.add_memory("tenant-a", "has an embedding", embedding=[1.0, 0.0])
    results = repo.search_memories("tenant-a", query_embedding=[1.0, 0.0], top_k=5)
    assert [m.content for m in results] == ["has an embedding"]


def test_search_memories_returns_empty_when_nothing_is_embedded(tmp_path):
    repo = _repo(tmp_path)
    repo.add_memory("tenant-a", "a fact")
    assert repo.search_memories("tenant-a", [1.0, 0.0], top_k=5) == []


def test_search_memories_respects_top_k(tmp_path):
    repo = _repo(tmp_path)
    for i in range(5):
        repo.add_memory("tenant-a", f"fact {i}", embedding=[float(i), 0.0])
    results = repo.search_memories("tenant-a", [4.0, 0.0], top_k=2)
    assert len(results) == 2


def test_search_memories_is_scoped_per_tenant(tmp_path):
    repo = _repo(tmp_path)
    repo.add_memory("tenant-a", "fact a", embedding=[1.0, 0.0])
    repo.add_memory("tenant-b", "fact b", embedding=[1.0, 0.0])
    results = repo.search_memories("tenant-a", [1.0, 0.0], top_k=5)
    assert [m.content for m in results] == ["fact a"]


def test_add_memory_backfills_an_embedding_onto_a_restated_fact(tmp_path):
    repo = _repo(tmp_path)
    mid1 = repo.add_memory("tenant-a", "a fact")  # no embedding initially
    mid2 = repo.add_memory("tenant-a", "a fact", embedding=[1.0, 0.0])  # restated, now with an embedding
    assert mid1 == mid2
    results = repo.search_memories("tenant-a", [1.0, 0.0], top_k=5)
    assert [m.memory_id for m in results] == [mid1]


def test_add_memory_never_overwrites_an_existing_embedding(tmp_path):
    repo = _repo(tmp_path)
    mid1 = repo.add_memory("tenant-a", "a fact", embedding=[1.0, 0.0])
    mid2 = repo.add_memory("tenant-a", "a fact", embedding=[0.0, 1.0])  # restated with a DIFFERENT vector
    assert mid1 == mid2
    # original embedding wins -- still ranks top for its original direction
    results = repo.search_memories("tenant-a", [1.0, 0.0], top_k=5)
    assert [m.memory_id for m in results] == [mid1]


def test_count_reflects_inserts_and_deletes(tmp_path):
    repo = _repo(tmp_path)
    assert repo.count("tenant-a") == 0
    mid = repo.add_memory("tenant-a", "a fact")
    assert repo.count("tenant-a") == 1
    repo.delete_memory(mid)
    assert repo.count("tenant-a") == 0


def test_migrates_an_old_schema_missing_the_embedding_column(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, "
        "content TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    conn.execute("INSERT INTO memories VALUES (?,?,?,?)", ("mid1", "tenant-a", "old fact", 0.0))
    conn.commit()
    conn.close()

    repo = MemoryRepository(path)  # must not raise -- self-healing migration adds `embedding`
    assert [m.content for m in repo.list_memories("tenant-a")] == ["old fact"]
    assert repo.search_memories("tenant-a", [1.0, 0.0], top_k=5) == []  # old row has no embedding yet
