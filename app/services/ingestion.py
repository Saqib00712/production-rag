"""
Indexing pipeline: parsed pages -> clean -> chunk -> (cache lookup) -> embed
only what is missing -> atomic save. Returns an IndexReport with tokens/cost.

Failure semantics: nothing is persisted unless embedding fully succeeds, so a
provider outage never leaves a half-indexed document.
"""

import hashlib
import logging
import time
from typing import Callable

from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.models.chunk import Chunk, IndexReport
from app.models.document import PageStatus, ParsedDocument
from app.repositories.chunk_repository import ChunkRepository
from app.services.chunker import Chunker, TokenCounter
from app.services.cost import estimate_cost_usd
from app.services.embedder import Embedder, EmbeddingError
from app.observability.metrics import cost_tracker
from app.observability.tracing import span
from app.services.pii import detect_pii, merge_counts, redact_pii
from app.services.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    pass


class IndexLimitError(IngestionError):
    """Document exceeds a configured cost/abuse guard."""


class NothingToIndexError(IngestionError):
    """No page contained usable text."""


async def index_document(
    parsed: ParsedDocument,
    *,
    repo: ChunkRepository,
    embedder_factory: Callable[[], Embedder],
    counter: TokenCounter,
    settings: Settings,
    request_id: str,
    dry_run: bool = False,
) -> IndexReport:
    start = time.perf_counter()
    model = settings.openai_embedding_model

    # 1) clean, page by page (page numbers are never merged). PageStatus.OCR (Day 21) is indexable
    # exactly like OK -- it's real transcribed text, just sourced from a vision call instead of the
    # PDF's own text layer; only EMPTY (no usable text, OCR off/unavailable/failed) and EXTRACTION_ERROR
    # are skipped.
    pages: list[tuple[int, str]] = []
    skipped: list[int] = []
    for page in parsed.pages:
        cleaned = clean_text(page.text) if page.status in (PageStatus.OK, PageStatus.OCR) else ""
        if cleaned:
            pages.append((page.page_number, cleaned))
        else:
            skipped.append(page.page_number)
    if not pages:
        raise NothingToIndexError("No page contains extractable text (scanned PDF with OCR off or unavailable?).")

    # 1b) PII handling (Day 7 detect, Day 21 redact). Runs on CLEANED text, before chunking, so a
    # "redact" decision changes what actually gets chunked/embedded/stored -- not just what gets
    # reported afterward. "warn" (the default) only ever counts; it never touches `pages`.
    pii_detected: dict[str, int] = {}
    if settings.pii_mode in ("warn", "redact"):
        pii_detected = merge_counts([detect_pii(text) for _, text in pages])
        if settings.pii_mode == "redact" and pii_detected:
            pages = [(page_number, redact_pii(text)) for page_number, text in pages]

    # 2) chunk
    chunker = Chunker(counter, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    drafts = await run_in_threadpool(chunker.chunk_pages, pages)
    if len(drafts) > settings.max_chunks_per_document:
        raise IndexLimitError(
            f"Document produces {len(drafts)} chunks; limit is {settings.max_chunks_per_document}."
        )
    chunks = [
        Chunk(
            chunk_id=f"{parsed.document_id}-c{d.chunk_index:04d}",
            document_id=parsed.document_id,
            tenant_id=parsed.metadata.tenant_id,
            chunk_index=d.chunk_index,
            page_number=d.page_number,
            text=d.text,
            token_count=d.token_count,
            content_hash=hashlib.sha256(d.text.encode("utf-8")).hexdigest(),
        )
        for d in drafts
    ]

    # 3) embedding cache: only unseen (hash, model) pairs cost money
    unique: dict[str, tuple[str, int]] = {}
    for c in chunks:
        unique.setdefault(c.content_hash, (c.text, c.token_count))
    cached = await run_in_threadpool(repo.get_cached_embeddings, list(unique), model)
    misses = {h: v for h, v in unique.items() if h not in cached}
    for h in unique:
        cost_tracker.record_cache("embedding", hit=h in cached)
    est_tokens = sum(tokens for _, tokens in misses.values())
    if est_tokens > settings.max_index_tokens:
        raise IndexLimitError(
            f"Indexing needs ~{est_tokens} embedding tokens; limit is {settings.max_index_tokens}."
        )

    # 4) embed only the misses, then save atomically
    billed = est_tokens
    if not dry_run:
        new_vectors: dict[str, list[float]] = {}
        billed = 0
        if misses:
            hashes = list(misses)
            with span("embedding", chunks=len(hashes)):
                result = await embedder_factory().embed([misses[h][0] for h in hashes])
            if len(result.vectors) != len(hashes) or len({len(v) for v in result.vectors}) != 1:
                raise EmbeddingError("Provider returned an unexpected number/shape of vectors.")
            new_vectors = dict(zip(hashes, result.vectors))
            billed = result.total_tokens
        await run_in_threadpool(repo.save_indexed_document, parsed.document_id, chunks, new_vectors, model)

    cost = estimate_cost_usd(billed, settings.embedding_price_per_million_tokens)
    if billed:
        cost_tracker.record_cost("embedding", billed, cost)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    n_cached = sum(1 for c in chunks if c.content_hash in cached)

    warnings: list[str] = []
    if skipped:
        warnings.append(f"Pages {skipped} had no usable text and were skipped.")
    if not counter.is_exact:
        warnings.append("tiktoken unavailable: token counts are approximate (~4 chars/token).")

    if pii_detected:
        kinds = ", ".join(f"{k}x{v}" for k, v in pii_detected.items())
        if settings.pii_mode == "redact":
            warnings.append(
                f"Document contained possible PII ({kinds}) that was REDACTED before indexing -- the "
                f"original values were never chunked, embedded, or stored."
            )
        else:
            warnings.append(
                f"Document contains possible PII ({kinds}). This is expected for documents like resumes; "
                f"this content will be sent to OpenAI when questions are asked about it."
            )
        logger.info("pii_detected_in_document", extra={"request_id": request_id, "document_id": parsed.document_id,
                                                         "pii_types": list(pii_detected.keys()),
                                                         "pii_mode": settings.pii_mode})

    logger.info(
        "document_dry_run" if dry_run else "document_indexed",
        extra={"request_id": request_id, "document_id": parsed.document_id,
               "duration_ms": duration_ms, "chunks": len(chunks), "tokens": billed, "cost_usd": cost},
    )
    return IndexReport(
        document_id=parsed.document_id, model=model, dry_run=dry_run,
        pages_total=len(parsed.pages), pages_indexed=len(pages), skipped_pages=skipped,
        chunk_count=len(chunks), chunks_from_cache=n_cached,
        chunks_needing_embedding=len(chunks) - n_cached, tokens=billed,
        cost_usd=cost, duration_ms=duration_ms, warnings=warnings, pii_detected=pii_detected,
        pii_mode=settings.pii_mode,
    )
