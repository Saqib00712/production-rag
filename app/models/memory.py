"""Data contracts for long-term, cross-conversation memory (Day 14).

Deliberately NOT the same shape as a conversation Turn (models/conversation.py):
a memory is a single durable fact ("prefers metric units", "works in
procurement"), not a transcript entry -- it has no question/answer pair and
outlives any one conversation.
"""

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    memory_id: str
    content: str
    created_at: float


class CreateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=500)


class ListMemoriesResponse(BaseModel):
    memories: list[MemoryItem]
