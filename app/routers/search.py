"""POST /search: keyword | vector | hybrid retrieval with page-level results.

Day 10: requires an API key and every search is tenant-scoped (see
ChunkRepository's tenant_id filtering). Deliberately uses only
`get_current_tenant`, not `enforce_tenant_limits` -- search is the cheap,
no-generation endpoint (same reasoning as it being excluded from Day 7's
per-IP rate-limit prefixes), so it gets identity-checking without the
stricter per-tenant rate/budget gate reserved for paid-generation calls.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth import TenantContext, get_current_tenant
from app.config import Settings, get_settings
from app.dependencies import EmbedderFactory, get_embedder_factory, get_repository, get_tenant_index_cache
from app.models.search import SearchRequest, SearchResponse
from app.repositories.chunk_repository import ChunkRepository
from app.observability.tracing import get_trace_id
from app.services.embedder import EmbeddingConfigError, EmbeddingError
from app.services.search import SearchError, SearchService
from app.services.vector_index_cache import TenantIndexCache

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    settings: Settings = Depends(get_settings),
    repo: ChunkRepository = Depends(get_repository),
    embedder_factory: EmbedderFactory = Depends(get_embedder_factory),
    tenant: TenantContext = Depends(get_current_tenant),
    index_cache: TenantIndexCache = Depends(get_tenant_index_cache),
) -> SearchResponse:
    service = SearchService(repo, embedder_factory, settings, index_cache)
    try:
        return await service.search(
            query=body.query, mode=body.mode, top_k=body.top_k,
            document_id=str(body.document_id) if body.document_id else None,
            tenant_id=tenant.tenant_id,
            request_id=get_trace_id(),
        )
    except EmbeddingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail="Embedding provider failed; try mode=keyword or retry later.") from exc
    except SearchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
