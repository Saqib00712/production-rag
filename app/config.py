"""
Centralized application configuration (Pydantic BaseSettings).
Real values live in a git-ignored `.env`; `.env.example` documents the shape.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- OpenAI (used from Day 2: embeddings) ---
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    # USD per 1M tokens. Verify on OpenAI's pricing page; kept configurable
    # so a price change never needs a code change.
    embedding_price_per_million_tokens: float = 0.02
    embedding_batch_size: int = 96
    embedding_timeout_seconds: float = 30.0
    embedding_max_retries: int = 3

    # --- App ---
    app_name: str = "production-rag"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Upload limits ---
    max_upload_size_mb: int = 20
    allowed_content_types: tuple[str, ...] = ("application/pdf",)

    # --- Chunking ---
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50

    # --- Cost / abuse guards for indexing ---
    max_chunks_per_document: int = 2000
    max_index_tokens: int = 300_000  # ~ $0.006 with text-embedding-3-small

    # --- Vector index (Day 12) ---
    vector_index_backend: str = "brute_force"   # "brute_force" (exact, Day 3) or "hnsw" (approximate, at scale)
    vector_index_ef_construction: int = 200     # HNSW build-time effort (higher = better recall, slower build)
    vector_index_m: int = 16                    # HNSW graph connectivity (higher = better recall, more memory)
    vector_index_ef_search: int = 50            # HNSW query-time effort (higher = better recall, slower query)

    # --- Multi-tenancy / auth ---
    tenant_db_path: str = "app/storage/tenants.db"
    tenant_rate_limit_per_minute: int = 30       # per authenticated tenant, on top of the Day 7 per-IP limiter
    tenant_daily_budget_usd: float = 1.00        # a safety ceiling, not a billing plan -- see docs/day10.md
    auto_bootstrap_dev_key: bool = True          # create+print a usable key on first startup if none exist

    # --- Conversation memory (Day 13) ---
    conversation_db_path: str = "app/storage/conversations.db"
    contextualizer_max_turns: int = 5            # how much recent history feeds the rewrite LLM call
    openai_contextualizer_model: str = "gpt-4o-mini"

    # --- Long-term, cross-conversation memory (Day 14) ---
    memory_db_path: str = "app/storage/memories.db"
    memory_max_facts: int = 20                   # per-tenant STORAGE cap; oldest fact evicted first (FIFO)
    openai_memory_extractor_model: str = "gpt-4o-mini"

    # --- Vector-indexed memory retrieval (Day 15) ---
    memory_top_k: int = 5                         # how many RELEVANT facts are injected per question (<= memory_max_facts)

    # --- Security guardrails ---
    rate_limit_per_minute: int = 20        # applies to /ask and /documents/*/index (the paid endpoints)
    rate_limit_window_seconds: float = 60.0
    # "off" (skip detection entirely) | "warn" (Day 7 default: detect + log + surface in warnings,
    # NEVER alters content) | "redact" (Day 21: detected spans are replaced with [REDACTED_*] markers
    # BEFORE chunking/embedding, so raw PII is never stored or sent to OpenAI) -- see services/pii.py
    pii_mode: str = "warn"

    # --- OpenTelemetry export (Day 21) ---
    # Off by default: the project's own app/observability/tracing.py (Day 9) already answers "where did
    # the time go" with zero infrastructure. Turning this on ADDITIONALLY mirrors the same spans into a
    # real OTel SDK TracerProvider -- the documented upgrade path from Day 9's docstring, now built.
    otel_enabled: bool = False
    otel_service_name: str = "production-rag"
    # Empty = ConsoleSpanExporter (prints spans locally, zero infra, good for seeing this work at all).
    # Set to a real collector URL (e.g. http://localhost:4318/v1/traces) to actually ship spans somewhere.
    otel_exporter_otlp_endpoint: str = ""

    # --- OCR for scanned PDFs (Day 21) ---
    # Off by default: this is the single most expensive step in the whole pipeline (a vision-capable
    # chat completion per page, vs. a few cents per million tokens for everything else), so it is
    # strictly opt-in and capped per document -- see ocr_max_pages_per_document.
    ocr_enabled: bool = False
    openai_ocr_model: str = "gpt-4o-mini"
    ocr_max_pages_per_document: int = 5
    ocr_price_input_per_million: float = 0.15
    ocr_price_output_per_million: float = 0.60

    # --- Chat models (rerank + generation). Prices: USD per 1M tokens, verify on OpenAI's pricing page ---
    openai_chat_model: str = "gpt-4o-mini"
    openai_rerank_model: str = "gpt-4o-mini"
    chat_price_input_per_million: float = 0.15
    chat_price_output_per_million: float = 0.60
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    max_answer_tokens: int = 500

    # --- RAG pipeline ---
    retrieve_k: int = 10          # candidates fetched by hybrid search
    max_context_tokens: int = 3000
    min_rerank_score: float = 4.0  # 0-10; below this = "not relevant enough to answer"
    min_vector_score: float = 0.2  # cosine gate used when reranking is unavailable

    # --- Search ---
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    search_candidate_multiplier: int = 4  # each retriever fetches top_k * this
    query_cache_max_rows: int = 5000

    # --- Storage ---
    raw_storage_dir: str = "app/storage/raw"
    sqlite_path: str = "app/storage/rag.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached per process; overridable in tests via dependency_overrides."""
    return Settings()
