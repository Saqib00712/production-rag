"""Data contracts for multi-turn conversations (Day 13)."""

from pydantic import BaseModel


class Turn(BaseModel):
    turn_index: int
    question: str              # the user's actual wording for this turn
    search_query: str          # what was actually used for retrieval (may be contextualized -- see services/contextualizer.py)
    answer: str
    citations: list[dict] = []  # already-validated Citation data from that turn, stored as plain dicts
    refusal_reason: str | None = None
    created_at: float


class CreateConversationResponse(BaseModel):
    conversation_id: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    turns: list[Turn]
