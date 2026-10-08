"""
SQLite persistence for API keys and per-tenant usage.

Separate from ChunkRepository (different concern: identity/billing vs.
document content) even though both live in the same SQLite file for this
project's scale -- a real deployment could split this into its own
database without changing the interface callers use.

Keys are stored as SHA-256 hashes, never plaintext -- the same principle as
password storage. The plaintext key is shown to the operator exactly once,
at creation time (scripts/manage_keys.py), and never again.
"""

import hashlib
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash    TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    label       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);

CREATE TABLE IF NOT EXISTS tenant_usage (
    tenant_id   TEXT NOT NULL,
    usage_date  TEXT NOT NULL,   -- 'YYYY-MM-DD', UTC
    cost_usd    REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, usage_date)
);
"""


def hash_key(plaintext_key: str) -> str:
    return hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()


@dataclass
class ApiKeyRecord:
    key_hash: str
    tenant_id: str
    label: str
    created_at: float
    revoked: bool


class TenantRepository:
    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

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

    # ---------------- API keys ----------------
    def create_key(self, plaintext_key: str, tenant_id: str, label: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO api_keys VALUES (?,?,?,?,0)",
                (hash_key(plaintext_key), tenant_id, label, time.time()),
            )

    def lookup(self, plaintext_key: str) -> ApiKeyRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT key_hash, tenant_id, label, created_at, revoked FROM api_keys WHERE key_hash = ?",
                (hash_key(plaintext_key),),
            ).fetchone()
        if row is None:
            return None
        return ApiKeyRecord(key_hash=row[0], tenant_id=row[1], label=row[2], created_at=row[3], revoked=bool(row[4]))

    def list_keys(self, tenant_id: str | None = None) -> list[ApiKeyRecord]:
        with self._conn() as conn:
            if tenant_id:
                rows = conn.execute(
                    "SELECT key_hash, tenant_id, label, created_at, revoked FROM api_keys WHERE tenant_id = ? "
                    "ORDER BY created_at", (tenant_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key_hash, tenant_id, label, created_at, revoked FROM api_keys ORDER BY created_at"
                ).fetchall()
        return [ApiKeyRecord(key_hash=r[0], tenant_id=r[1], label=r[2], created_at=r[3], revoked=bool(r[4])) for r in rows]

    def revoke(self, key_hash_prefix: str) -> int:
        """Revokes by hash PREFIX so an operator can act from the truncated
        hash shown in `list_keys` output without ever having the plaintext
        key again. Returns the number of keys revoked (0 or 1 in practice,
        but a short/ambiguous prefix could match more -- caller should
        check and warn)."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE key_hash LIKE ?", (key_hash_prefix + "%",)
            )
            return cur.rowcount

    def any_keys_exist(self) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM api_keys LIMIT 1").fetchone() is not None

    # ---------------- usage ledger ----------------
    def add_usage(self, tenant_id: str, usage_date: str, cost_usd: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tenant_usage VALUES (?,?,?) "
                "ON CONFLICT(tenant_id, usage_date) DO UPDATE SET cost_usd = cost_usd + excluded.cost_usd",
                (tenant_id, usage_date, cost_usd),
            )

    def get_usage(self, tenant_id: str, usage_date: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT cost_usd FROM tenant_usage WHERE tenant_id = ? AND usage_date = ?",
                (tenant_id, usage_date),
            ).fetchone()
        return row[0] if row else 0.0
