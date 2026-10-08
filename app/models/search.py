"""Search request/response contracts."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

SearchMode = Literal["keyword", "vector", "hybrid"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: SearchMode = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    document_id: UUID | None = None  # restrict to one document

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    page_number: int  # the citation anchor
    text: str
    score: float  # meaning depends on score_type
    keyword_rank: int | None = None
    keyword_score: float | None = None  # BM25 (higher = better)
    vector_rank: int | None = None
    vector_score: float | None = None  # cosine similarity in [-1, 1]


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    score_type: Literal["bm25", "cosine", "rrf"]
    top_k: int
    document_id: str | None
    hits: list[SearchHit]
    degraded: bool = False  # hybrid fell back to keyword-only
    warnings: list[str] = []
    query_tokens: int = 0  # embedding tokens billed for this query (0 = cache hit or keyword)
    query_cost_usd: float = 0.0
    timings_ms: dict[str, float] = {}
