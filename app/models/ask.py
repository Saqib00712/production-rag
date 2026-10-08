from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    document_id: UUID | None = None
    top_k: int = Field(5, ge=1, le=10, description="Max source chunks given to the model")
    rerank: bool = True
    conversation_id: UUID | None = Field(
        default=None,
        description="Optional: created via POST /conversations. When given, recent turns are used to "
                     "contextualize retrieval and this turn is appended to the conversation's history.",
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    page_number: int
    snippet: str
    rerank_score: float | None = None


class CostBreakdown(BaseModel):
    embedding_tokens: int = 0
    rerank_prompt_tokens: int = 0
    rerank_completion_tokens: int = 0
    generation_prompt_tokens: int = 0
    generation_completion_tokens: int = 0
    # Day 16: these three were real costs since Day 13/14/15 introduced them
    # (query contextualization, memory fact extraction, memory embeddings)
    # but were never counted here or against the Day 10 per-tenant daily
    # budget until now -- see docs/day16.md.
    contextualizer_prompt_tokens: int = 0
    contextualizer_completion_tokens: int = 0
    memory_extraction_prompt_tokens: int = 0
    memory_extraction_completion_tokens: int = 0
    memory_embedding_tokens: int = 0  # the question embedding (Day 15 retrieval) + any newly extracted facts'
    embedding_cost_usd: float = 0.0
    rerank_cost_usd: float = 0.0
    generation_cost_usd: float = 0.0
    contextualizer_cost_usd: float = 0.0
    memory_extraction_cost_usd: float = 0.0
    memory_embedding_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


class AskResponse(BaseModel):
    question: str
    search_query: str | None = None  # set only when conversation_id was used AND the query was rewritten
    answer: str
    insufficient_evidence: bool
    refusal_reason: str | None = None  # no_results | low_relevance | model_insufficient_evidence | ungrounded_answer
    citations: list[Citation] = []
    degraded: bool = False
    warnings: list[str] = []
    cost: CostBreakdown = CostBreakdown()
    timings_ms: dict[str, float] = {}
    models: dict[str, str] = {}
