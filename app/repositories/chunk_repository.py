"""
SQLite persistence for chunks and embeddings.

Schema decision: embeddings are keyed by (content_hash, model), NOT by chunk.
  * Re-uploading the same document, or identical text in two documents,
    reuses the stored vector -> zero API cost (an embedding cache for free).
  * Changing the embedding model never overwrites old vectors.
  * This cache is intentionally SHARED across tenants (Day 10): the vector
    for a given piece of text is the same regardless of who uploaded it,
    and a content hash reveals nothing about tenant identity or document
    content by itself. Sharing it is a real cost saving with no isolation
    cost -- unlike CHUNKS, which are always tenant-scoped (see below).
Vectors are stored as packed float32 BLOBs.

Day 10 adds `tenant_id` to `chunks` (and the FTS index) and filters EVERY
chunk-reading query by it. This is the actual tenant isolation boundary:
even if a caller supplied no document_id filter (a cross-document search),
results can never include another tenant's chunks, because tenant_id is
never optional in these queries the way document_id is.

A new short-lived connection per call keeps this thread-safe when called
through FastAPI's threadpool.
"""

import sqlite3
import time
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from app.models.chunk import Chunk

_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    chunk_index  INTEGER NOT NULL,
    page_number  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    content_hash TEXT NOT NULL
);
"""

_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);
CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    PRIMARY KEY (content_hash, model)
);
CREATE TABLE IF NOT EXISTS query_embeddings (
    query_hash TEXT NOT NULL,
    model      TEXT NOT NULL,
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (query_hash, model)
);
"""

# BM25 keyword index (SQLite FTS5). Porter stemming: "refunds" matches "refund".
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    tenant_id UNINDEXED,
    tokenize = 'porter unicode61'
);
"""


def pack_vector(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


class ChunkRepository:
    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_CHUNKS_TABLE)
            self._migrate_tenant_column(conn)
            conn.executescript(_SCHEMA)
            try:
                conn.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError as exc:
                raise RuntimeError(
                    "This Python's SQLite was built without FTS5, which keyword search needs."
                ) from exc
            self._heal_fts(conn)

    @staticmethod
    def _migrate_tenant_column(conn: sqlite3.Connection) -> None:
        """Self-healing schema migration: a `chunks` table created before
        Day 10 has no `tenant_id` column. `CREATE TABLE IF NOT EXISTS`
        above does nothing for an existing table, so we check for the
        column explicitly and ALTER TABLE if it's missing -- existing rows
        get the 'default' tenant, matching DocumentMetadata's own backward-
        compat default (see app/models/document.py)."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "tenant_id" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")

    @staticmethod
    def _heal_fts(conn: sqlite3.Connection) -> None:
        """Self-healing: if the keyword index is out of sync with chunks
        (e.g. chunks created before FTS existed, or before tenant_id was
        added to the FTS schema), rebuild it from scratch."""
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if n_chunks != n_fts:
            conn.execute("DELETE FROM chunks_fts")
            conn.execute(
                "INSERT INTO chunks_fts(text, chunk_id, document_id, tenant_id) "
                "SELECT text, chunk_id, document_id, tenant_id FROM chunks"
            )

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

    def get_cached_embeddings(self, hashes: list[str], model: str) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._conn() as conn:
            for i in range(0, len(hashes), 500):  # stay under SQLite's variable limit
                part = hashes[i : i + 500]
                marks = ",".join("?" * len(part))
                rows = conn.execute(
                    f"SELECT content_hash, vector FROM embeddings "
                    f"WHERE model = ? AND content_hash IN ({marks})",
                    [model, *part],
                )
                for h, blob in rows:
                    found[h] = unpack_vector(blob)
        return found

    def save_indexed_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        new_embeddings: dict[str, list[float]],
        model: str,
    ) -> None:
        """Atomic: either the whole document is (re)indexed or nothing changes.
        Every chunk carries its own tenant_id (set by the caller from the
        document's owning tenant), so isolation holds even if this method
        were ever called with a mixed batch -- it never has to trust a
        single ambient tenant_id for the whole call."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
            conn.executemany(
                "INSERT INTO chunks_fts(text, chunk_id, document_id, tenant_id) VALUES (?,?,?,?)",
                [(c.text, c.chunk_id, c.document_id, c.tenant_id) for c in chunks],
            )
            conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                [
                    (c.chunk_id, c.document_id, c.tenant_id, c.chunk_index, c.page_number,
                     c.text, c.token_count, c.content_hash)
                    for c in chunks
                ],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                [(h, model, len(v), pack_vector(v)) for h, v in new_embeddings.items()],
            )

    def list_chunks(
        self, document_id: str, model: str, limit: int, offset: int, tenant_id: str
    ) -> tuple[int, list[Chunk]]:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ? AND tenant_id = ?", (document_id, tenant_id)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT c.chunk_id, c.document_id, c.tenant_id, c.chunk_index, c.page_number, c.text, "
                "c.token_count, c.content_hash, e.dim "
                "FROM chunks c LEFT JOIN embeddings e "
                "ON e.content_hash = c.content_hash AND e.model = ? "
                "WHERE c.document_id = ? AND c.tenant_id = ? ORDER BY c.chunk_index LIMIT ? OFFSET ?",
                (model, document_id, tenant_id, limit, offset),
            ).fetchall()
        chunks = [
            Chunk(chunk_id=r[0], document_id=r[1], tenant_id=r[2], chunk_index=r[3], page_number=r[4],
                  text=r[5], token_count=r[6], content_hash=r[7],
                  has_embedding=r[8] is not None, embedding_dim=r[8])
            for r in rows
        ]
        return total, chunks

    # ---------------- Day 3: search (Day 10: tenant-scoped) ----------------
    def keyword_search(
        self, match_query: str, limit: int, tenant_id: str, document_id: str | None = None
    ) -> list[tuple[str, float]]:
        """BM25 ranking. `match_query` MUST come from build_match_query()
        (quoted terms only) -- never pass raw user input to MATCH.
        FTS5's bm25() is negative (lower = better); we return -bm25 so
        higher = better. `tenant_id` is NOT optional: unlike document_id,
        which narrows an already tenant-scoped search, tenant_id is the
        isolation boundary itself and must always be applied."""
        if not match_query:
            return []
        sql = "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? AND tenant_id = ?"
        params: list = [match_query, tenant_id]
        if document_id:
            sql += " AND document_id = ?"
            params.append(document_id)
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return [(cid, -score) for cid, score in conn.execute(sql, params)]

    def load_vectors(
        self, model: str, tenant_id: str, document_id: str | None = None
    ) -> tuple[list[str], np.ndarray]:
        sql = (
            "SELECT c.chunk_id, e.vector FROM chunks c JOIN embeddings e "
            "ON e.content_hash = c.content_hash AND e.model = ? WHERE c.tenant_id = ?"
        )
        params: list = [model, tenant_id]
        if document_id:
            sql += " AND c.document_id = ?"
            params.append(document_id)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        ids = [r[0] for r in rows]
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return ids, matrix

    def get_chunks_by_ids(self, ids: list[str], tenant_id: str) -> dict[str, Chunk]:
        """tenant_id is still enforced here even though chunk_ids are
        already unguessable UUIDs scoped to a document: defense in depth
        against a future caller accidentally passing an id from the wrong
        tenant (e.g. a bug elsewhere), rather than relying solely on ids
        never leaking across tenants in the first place."""
        out: dict[str, Chunk] = {}
        with self._conn() as conn:
            for i in range(0, len(ids), 500):
                part = ids[i : i + 500]
                marks = ",".join("?" * len(part))
                for r in conn.execute(
                    "SELECT chunk_id, document_id, tenant_id, chunk_index, page_number, text, "
                    f"token_count, content_hash FROM chunks WHERE chunk_id IN ({marks}) AND tenant_id = ?",
                    [*part, tenant_id],
                ):
                    out[r[0]] = Chunk(
                        chunk_id=r[0], document_id=r[1], tenant_id=r[2], chunk_index=r[3], page_number=r[4],
                        text=r[5], token_count=r[6], content_hash=r[7],
                    )
        return out

    def get_query_embedding(self, query_hash: str, model: str) -> list[float] | None:
        # Deliberately NOT tenant-scoped: see module docstring on why the
        # embedding/query-embedding caches are safe and beneficial to share.
        with self._conn() as conn:
            row = conn.execute(
                "SELECT vector FROM query_embeddings WHERE query_hash = ? AND model = ?",
                (query_hash, model),
            ).fetchone()
        return unpack_vector(row[0]) if row else None

    def save_query_embedding(
        self, query_hash: str, model: str, vector: list[float], max_rows: int
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_embeddings VALUES (?,?,?,?)",
                (query_hash, model, pack_vector(vector), time.time()),
            )
            # bounded cache: user-controlled input must not grow storage forever
            conn.execute(
                "DELETE FROM query_embeddings WHERE rowid IN ("
                "SELECT rowid FROM query_embeddings ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                (max_rows,),
            )
