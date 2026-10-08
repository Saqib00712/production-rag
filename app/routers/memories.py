"""
/memories endpoints (Day 14): inspect, manually add, or delete the
tenant-scoped facts long-term memory has accumulated (or will use) across
ALL of that tenant's conversations -- not just one, which is what makes
this different from GET /conversations/{id} (Day 13).

These exist mainly for transparency and control: an automatically
extracted fact can be wrong or stale, and a user should be able to see
what's being remembered about them and remove it, same spirit as any
"what Claude remembers about you" surface. `POST /memories` also lets a
fact be stated directly, without waiting for the extractor to infer it
from a future turn.

Day 15: a manually added fact is embedded on the way in too, so it's
retrievable by `/ask`'s relevance search (repositories/memory_repository.py
`search_memories`) exactly like an auto-extracted one -- using the SAME
`_embed_text` fail-open helper as routers/ask.py, so a missing API key or
a live embedding failure here just means the fact falls back to Day 14's
flat, most-recent-facts behavior, never a reason to reject the request.

Note: unlike the three call sites Day 16 folded into `AskResponse.cost`
(contextualization, extraction, and the per-question memory embedding, all
triggered automatically as a SIDE EFFECT of calling /ask), the embedding
cost of a manually, explicitly added fact here is not billed against the
tenant's daily budget -- it's a direct, deliberate, user-initiated write
with no corresponding "request" to attach a cost breakdown to, the same
way uploading a document (as opposed to indexing it) isn't billed either.

Same tenant-isolation pattern as /conversations and /documents:
`get_owner()` returning None OR a different tenant_id both produce an
identical 404.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth import TenantContext, get_current_tenant
from app.config import Settings, get_settings
from app.dependencies import EmbedderFactory, get_embedder_factory, get_memory_repository
from app.models.memory import CreateMemoryRequest, ListMemoriesResponse, MemoryItem
from app.repositories.memory_repository import MemoryRepository
from app.routers.ask import _embed_text

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("", response_model=ListMemoriesResponse)
async def list_memories(
    tenant: TenantContext = Depends(get_current_tenant),
    repo: MemoryRepository = Depends(get_memory_repository),
) -> ListMemoriesResponse:
    return ListMemoriesResponse(memories=repo.list_memories(tenant.tenant_id))


@router.post("", response_model=MemoryItem, status_code=201)
async def create_memory(
    body: CreateMemoryRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    repo: MemoryRepository = Depends(get_memory_repository),
    settings: Settings = Depends(get_settings),
    embedder_factory: EmbedderFactory = Depends(get_embedder_factory),
) -> MemoryItem:
    embedding, _tokens = await _embed_text(body.content, embedder_factory)
    memory_id = repo.add_memory(
        tenant.tenant_id, body.content, max_memories=settings.memory_max_facts, embedding=embedding,
    )
    match = next((m for m in repo.list_memories(tenant.tenant_id) if m.memory_id == memory_id), None)
    if match is None:
        # Can only happen if this memory was immediately evicted by its own
        # insert (max_memories == 0) -- surface that plainly rather than a
        # misleading 500 from an unhandled None.
        raise HTTPException(status_code=422, detail="Memory was not retained (memory_max_facts is 0).")
    return match


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
    repo: MemoryRepository = Depends(get_memory_repository),
) -> None:
    owner = repo.get_owner(memory_id)
    if owner is None or owner != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="Memory not found")
    repo.delete_memory(memory_id)
