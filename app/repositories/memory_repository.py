"""
SQLite persistence for long-term, cross-conversation memory (Day 14), with
vector-indexed retrieval (Day 15).

Distinct from ConversationRepository (Day 13) on purpose: a conversation
turn is scoped to ONE conversation and read back by that conversation's
`conversation_id`. A memory fact is scoped to a TENANT and is meant to be
read back in a conversation that has never happened before -- "remembers
the user prefers metric units" has to survive the user starting a brand
new conversation next week, which conversation-scoped history never could.

Same "return 404, not 403" ownership pattern as conversations and
documents: `get_owner()` lets a caller distinguish "doesn't exist" from
"exists but isn't yours" nowhere in the response.

Cap + FIFO eviction (`_enforce_cap`): an LLM extracting "durable facts"
after every single turn, forever, would grow without bound and eventually
flood the generation prompt with stale or redundant notes -- diluting the
useful ones and inflating every future request's token cost. Capping at
`max_memories` and evicting the OLDEST fact first keeps the store small and
, in practice, keeps it recent: a fact worth remembering tends to get
restated (and re-extracted, see services/memory_extractor.py's dedup
instruction) if it's still relevant, so quietly dropping a years-old one
rarely loses anything that still matters.

Day 15 adds an optional per-fact embedding (same `pack_vector`/
`unpack_vector` float-array packing ChunkRepository already uses for chunk
vectors -- reused rather than reinvented) and `search_memories()`, a
brute-force cosine ranking over just THIS tenant's facts via
`vector_math.cosine_top_k` -- the exact same function Day 3's document
search and Day 12's HNSW fallback both use. Brute force, not an ANN index:
a tenant's memory store is capped at `memory_max_facts` (tens of facts,
not tens of thousands of chunks), so an index with its own build/maintain
overhead buys nothing here that a single matrix-vector product doesn't
already give for free -- the same "exact is fine until scale says
otherwise" reasoning Day 12's docs laid out for documents, just never
reaching the "otherwise" for a store this small by design.
"""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from app.models.memory import MemoryItem
from app.repositories.chunk_repository import pack_vector, unpack_vector
from app.services.vector_math import cosine_top_k

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id   TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(tenant_id, created_at);
"""


class MemoryRepository:
    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_embedding_column(conn)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _migrate_embedding_column(conn: sqlite3.Connection) -> None:
        """Self-healing schema migration, same pattern as
        ChunkRepository._migrate_tenant_column: a `memories` table created
        before Day 15 has no `embedding` column. `CREATE TABLE IF NOT
        EXISTS` above does nothing for an existing table, so we check for
        the column explicitly and ALTER TABLE if it's missing. Existing
        rows get NULL (no embedding yet) -- they simply fall back to the
        Day 14 recency-based list until re-extracted or re-stated, rather
        than erroring or needing a backfill migration."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "embedding" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")

    def add_memory(
        self, tenant_id: str, content: str, max_memories: int | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Inserts a new fact, or returns the id of an existing one with the
        same (case-insensitive, whitespace-trimmed) text for this tenant --
        the model re-stating a fact it already extracted shouldn't duplicate
        it. If that existing fact has no embedding yet (e.g. it predates
        Day 15, or embedding failed when it was first stored) and one is
        given now, it's backfilled onto the existing row rather than
        discarded -- a restated fact is a free opportunity to fill the gap.
        When `max_memories` is given, evicts the oldest fact(s) past that
        cap afterwards."""
        content = content.strip()
        blob = pack_vector(embedding) if embedding is not None else None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT memory_id, embedding FROM memories WHERE tenant_id = ? AND lower(trim(content)) = ?",
                (tenant_id, content.lower()),
            ).fetchone()
            if row:
                memory_id, existing_embedding = row
                if existing_embedding is None and blob is not None:
                    conn.execute("UPDATE memories SET embedding = ? WHERE memory_id = ?", (blob, memory_id))
                return memory_id
            memory_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO memories (memory_id, tenant_id, content, created_at, embedding) VALUES (?,?,?,?,?)",
                (memory_id, tenant_id, content, time.time(), blob),
            )
            if max_memories is not None:
                self._enforce_cap(conn, tenant_id, max_memories)
        return memory_id

    def _enforce_cap(self, conn: sqlite3.Connection, tenant_id: str, max_memories: int) -> None:
        (count,) = conn.execute("SELECT COUNT(*) FROM memories WHERE tenant_id = ?", (tenant_id,)).fetchone()
        overflow = count - max_memories
        if overflow > 0:
            conn.execute(
                "DELETE FROM memories WHERE memory_id IN ("
                "  SELECT memory_id FROM memories WHERE tenant_id = ? ORDER BY created_at ASC LIMIT ?"
                ")",
                (tenant_id, overflow),
            )

    def get_owner(self, memory_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT tenant_id FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        return row[0] if row else None

    def count(self, tenant_id: str) -> int:
        """Lets callers skip embedding a question entirely when a tenant
        has no memories at all yet -- the same cost-discipline shape as
        Day 13's contextualizer skipping a conversation's first turn."""
        with self._conn() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM memories WHERE tenant_id = ?", (tenant_id,)).fetchone()
        return n

    def list_memories(self, tenant_id: str, limit: int | None = None) -> list[MemoryItem]:
        """All of a tenant's facts, most-recently-added last -- the Day 14
        behavior, kept as-is for GET /memories and as the fallback
        `search_memories` uses when no embedder is available or no fact has
        an embedding yet."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, content, created_at FROM memories WHERE tenant_id = ? ORDER BY created_at ASC",
                (tenant_id,),
            ).fetchall()
        items = [MemoryItem(memory_id=r[0], content=r[1], created_at=r[2]) for r in rows]
        if limit is not None and limit >= 0:
            items = items[-limit:]
        return items

    def search_memories(self, tenant_id: str, query_embedding: list[float], top_k: int) -> list[MemoryItem]:
        """The Day 15 retrieval path: ranks this tenant's EMBEDDED facts by
        cosine similarity to `query_embedding` and returns the top_k, most
        relevant first. Facts with no embedding yet (NULL `embedding`) are
        excluded here, not scored as zero -- callers fall back to
        `list_memories` when this returns fewer facts than they need (see
        routers/ask.py's `_fetch_memory_notes`), rather than this method
        silently padding its own result with irrelevant, unscored rows."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, content, created_at, embedding FROM memories "
                "WHERE tenant_id = ? AND embedding IS NOT NULL",
                (tenant_id,),
            ).fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        by_id = {r[0]: MemoryItem(memory_id=r[0], content=r[1], created_at=r[2]) for r in rows}
        matrix = np.array([unpack_vector(r[3]) for r in rows], dtype=np.float32)
        ranked = cosine_top_k(query_embedding, ids, matrix, top_k)
        return [by_id[mid] for mid, _score in ranked]

    def delete_memory(self, memory_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
