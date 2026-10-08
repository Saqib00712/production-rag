"""
SQLite persistence for multi-turn conversations.

Same reasoning as Day 10's tenant usage ledger: a conversation a user is in
the middle of must survive a server restart, so it lives in SQLite, not in
memory. Separate file/table from ChunkRepository and TenantRepository
(different concern again: conversation state vs. document content vs.
identity/billing) even though all three could share one SQLite file at this
project's scale.

Ownership: `get_owner()` exists specifically to feed the same "return 404,
not 403" isolation pattern used for documents (see routers/documents.py's
`_load_parsed`) -- a caller who doesn't own a conversation_id should never
be able to tell the difference between "doesn't exist" and "exists, but
isn't yours".
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.models.conversation import Turn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_tenant ON conversations(tenant_id);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id             TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL,
    turn_index          INTEGER NOT NULL,
    question            TEXT NOT NULL,
    search_query        TEXT NOT NULL,
    answer              TEXT NOT NULL,
    citations_json      TEXT NOT NULL,
    refusal_reason      TEXT,
    created_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_conversation ON conversation_turns(conversation_id, turn_index);
"""


class ConversationRepository:
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

    def create_conversation(self, tenant_id: str) -> str:
        conversation_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations VALUES (?,?,?)",
                (conversation_id, tenant_id, time.time()),
            )
        return conversation_id

    def get_owner(self, conversation_id: str) -> str | None:
        """Returns the owning tenant_id, or None if no such conversation
        exists. Callers use this for the 404-not-403 isolation check --
        never report "not yours" differently from "doesn't exist"."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT tenant_id FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        return row[0] if row else None

    def append_turn(
        self, conversation_id: str, *, question: str, search_query: str, answer: str,
        citations: list[dict], refusal_reason: str | None,
    ) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) FROM conversation_turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            next_index = row[0] + 1
            conn.execute(
                "INSERT INTO conversation_turns VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()), conversation_id, next_index, question, search_query, answer,
                    json.dumps(citations), refusal_reason, time.time(),
                ),
            )

    def get_turns(self, conversation_id: str, limit: int | None = None) -> list[Turn]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT turn_index, question, search_query, answer, citations_json, refusal_reason, created_at "
                "FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_index",
                (conversation_id,),
            ).fetchall()
        turns = [
            Turn(
                turn_index=r[0], question=r[1], search_query=r[2], answer=r[3],
                citations=json.loads(r[4]), refusal_reason=r[5], created_at=r[6],
            )
            for r in rows
        ]
        if limit is not None and limit >= 0:
            turns = turns[-limit:]
        return turns
