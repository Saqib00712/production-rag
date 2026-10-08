"""
/documents endpoints.

Design decision: the router's ONLY job is HTTP concerns -- validating the
request, generating a request_id for tracing, calling the parser service,
and shaping the response. It contains no PDF-parsing logic itself (that
lives in services/pdf_parser.py). This separation is what lets us later
swap "save JSON to disk" for "enqueue a background job" (Day 15/16) without
touching the parsing logic at all.

Day 10: every endpoint now requires an API key (`enforce_tenant_limits`,
which itself depends on `get_current_tenant`) and every document is
tenant-scoped. `_load_parsed` returns 404 -- not 403 -- when a document
exists but belongs to a different tenant, so the API never confirms or
denies that a given document_id exists to someone who doesn't own it.
"""

import json
import logging
import time
from pathlib import Path

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.auth import TenantContext, enforce_tenant_limits, get_current_tenant, record_tenant_spend
from app.config import Settings, get_settings
from app.dependencies import (
    EmbedderFactory, OcrFactory, get_embedder_factory, get_ocr_factory, get_repository,
    get_tenant_index_cache, get_tenant_repository, get_token_counter,
)
from app.models.chunk import ChunkListResponse, IndexReport
from app.models.document import ParsedDocument
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.tenant_repository import TenantRepository
from app.services.chunker import TokenCounter
from app.services.embedder import EmbeddingConfigError, EmbeddingError
from app.services.ingestion import IndexLimitError, NothingToIndexError, index_document
from app.services.vector_index_cache import TenantIndexCache
from app.observability.metrics import cost_tracker
from app.observability.tracing import get_trace_id
from app.services.pdf_parser import PdfParsingError, parse_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=ParsedDocument,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    tenant: TenantContext = Depends(enforce_tenant_limits),
    ocr_factory: OcrFactory = Depends(get_ocr_factory),
    tenant_repo: TenantRepository = Depends(get_tenant_repository),
) -> ParsedDocument:
    request_id = get_trace_id()
    start = time.perf_counter()

    # --- Input validation (see SECURITY section for why each check exists) ---
    if file.content_type not in settings.allowed_content_types:
        logger.warning(
            "rejected_content_type",
            extra={"request_id": request_id, "error_type": "InvalidContentType"},
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'. Only PDF is accepted.",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )

    if len(file_bytes) > settings.max_upload_size_bytes:
        logger.warning(
            "rejected_file_too_large",
            extra={"request_id": request_id, "error_type": "FileTooLarge"},
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb}MB limit",
        )

    # --- Parsing ---
    try:
        parsed = await parse_pdf(
            file_bytes=file_bytes,
            filename=file.filename or "unknown.pdf",
            content_type=file.content_type,
            request_id=request_id,
            ocr_factory=ocr_factory,
            ocr_max_pages=settings.ocr_max_pages_per_document,
            ocr_price_input_per_million=settings.ocr_price_input_per_million,
            ocr_price_output_per_million=settings.ocr_price_output_per_million,
        )
    except PdfParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    parsed.metadata.tenant_id = tenant.tenant_id

    if parsed.metadata.ocr_pages_used:
        cost_tracker.record_cost("ocr", parsed.metadata.ocr_tokens, parsed.metadata.ocr_cost_usd)
        record_tenant_spend(tenant_repo, tenant.tenant_id, parsed.metadata.ocr_cost_usd)
        logger.info(
            "ocr_pages_billed",
            extra={"request_id": request_id, "document_id": parsed.document_id,
                   "ocr_pages_used": parsed.metadata.ocr_pages_used, "cost_usd": parsed.metadata.ocr_cost_usd},
        )

    # --- Storage (dev-only: local JSON file. Replaced by a real DB/object
    # store starting Day 2 once chunking + embeddings need a persistence
    # layer with query capability, not just flat files.) ---
    storage_dir = Path(settings.raw_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    out_path = storage_dir / f"{parsed.document_id}.json"
    out_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "document_ingested",
        extra={
            "request_id": request_id,
            "document_id": parsed.document_id,
            "duration_ms": duration_ms,
        },
    )

    return parsed


def _load_parsed(settings: Settings, document_id: str, tenant_id: str) -> ParsedDocument:
    # document_id is a validated UUID -> no path traversal possible
    path = Path(settings.raw_storage_dir) / f"{document_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    parsed = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if parsed.metadata.tenant_id != tenant_id:
        # Same response as "does not exist" -- never confirm existence of
        # another tenant's document to a caller who doesn't own it.
        raise HTTPException(status_code=404, detail="Document not found")
    return parsed


@router.post("/{document_id}/index", response_model=IndexReport)
async def index_document_endpoint(
    document_id: UUID,
    dry_run: bool = Query(False, description="Estimate chunks/tokens/cost without calling OpenAI or writing anything"),
    settings: Settings = Depends(get_settings),
    repo: ChunkRepository = Depends(get_repository),
    embedder_factory: EmbedderFactory = Depends(get_embedder_factory),
    counter: TokenCounter = Depends(get_token_counter),
    tenant: TenantContext = Depends(enforce_tenant_limits),
    tenant_repo: TenantRepository = Depends(get_tenant_repository),
    index_cache: TenantIndexCache = Depends(get_tenant_index_cache),
) -> IndexReport:
    """Clean -> chunk -> embed -> store. Use ?dry_run=true first to see the cost."""
    request_id = get_trace_id()
    parsed = await run_in_threadpool(_load_parsed, settings, str(document_id), tenant.tenant_id)
    try:
        report = await index_document(
            parsed, repo=repo, embedder_factory=embedder_factory, counter=counter,
            settings=settings, request_id=request_id, dry_run=dry_run,
        )
        record_tenant_spend(tenant_repo, tenant.tenant_id, report.cost_usd)
        if not dry_run:
            # New/changed vectors for this tenant: the cached whole-tenant
            # search index (Day 12) is now stale and must rebuild on next use.
            index_cache.invalidate(tenant.tenant_id)
        return report
    except IndexLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except NothingToIndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail="Embedding provider failed; nothing was saved. Retry later.") from exc


@router.get("/{document_id}/chunks", response_model=ChunkListResponse)
async def list_chunks(
    document_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    settings: Settings = Depends(get_settings),
    repo: ChunkRepository = Depends(get_repository),
    tenant: TenantContext = Depends(get_current_tenant),
) -> ChunkListResponse:
    total, chunks = await run_in_threadpool(
        repo.list_chunks, str(document_id), settings.openai_embedding_model, limit, offset, tenant.tenant_id
    )
    if total == 0:
        raise HTTPException(status_code=404, detail="No chunks found. Index the document first.")
    return ChunkListResponse(document_id=str(document_id), total=total, limit=limit, offset=offset, chunks=chunks)
