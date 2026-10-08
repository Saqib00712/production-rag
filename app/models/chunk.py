"""Data contracts for chunks and indexing reports."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    tenant_id: str = "default"
    chunk_index: int = Field(..., ge=0)
    page_number: int = Field(..., ge=1)  # the citation anchor
    text: str
    token_count: int = Field(..., ge=0)
    content_hash: str  # sha256(text): cache key + dedup key
    has_embedding: bool = False
    embedding_dim: int | None = None


class ChunkListResponse(BaseModel):
    document_id: str
    total: int
    limit: int
    offset: int
    chunks: list[Chunk]


class IndexReport(BaseModel):
    document_id: str
    model: str
    dry_run: bool
    pages_total: int
    pages_indexed: int
    skipped_pages: list[int]
    chunk_count: int
    chunks_from_cache: int
    chunks_needing_embedding: int  # in dry_run: "would be embedded"
    tokens: int  # dry_run: estimated; otherwise billed by OpenAI
    cost_usd: float
    duration_ms: float
    warnings: list[str] = []
    pii_detected: dict[str, int] = {}  # category -> count; see services/pii.py
    pii_mode: str = "warn"  # "off" | "warn" | "redact" -- what was actually applied for this request (Day 21)
