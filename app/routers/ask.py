"""POST /ask: retrieve -> rerank -> generate a grounded, cited answer.

Day 10: both /ask and /ask/stream require an API key and enforce per-tenant
rate limits + daily budget (`enforce_tenant_limits`), on top of Day 7's
per-IP middleware limiter which still applies first (middleware runs before
these dependencies). Every answer is scoped to the caller's tenant_id.

Day 13: an optional `conversation_id` (created via POST /conversations)
turns a one-shot question into one turn of a multi-turn conversation.
`_load_history` applies the same 404-not-403 isolation pattern used for
documents (routers/documents.py) -- a conversation that exists but belongs
to another tenant looks identical to one that doesn't exist at all. When
there IS history, the follow-up is contextualized into a standalone search
query BEFORE retrieval (see services/contextualizer.py); the ORIGINAL
question is still what's generated from and shown back. After a successful
call, the turn is appended to the conversation so the next follow-up has
it available.

Day 14: independent of any conversation_id, every call also reads that
TENANT's long-term memory facts (see repositories/memory_repository.py)
and hands them to the generator as background context -- this is what
lets a fact survive into a conversation that has never happened before,
which Day 13's conversation-scoped history cannot do. After a successful
(non-refused) exchange, `_maybe_extract_memories` asks an LLM whether
anything durable was stated and, if so, stores it for future turns --
same fail-open philosophy as `_contextualize`: an extraction failure never
breaks the request that triggered it.

Day 15: `_fetch_memory_notes` no longer just dumps every stored fact into
the prompt -- it embeds the question and asks `MemoryRepository.search_
memories` for only the `memory_top_k` most RELEVANT facts (cosine
similarity, see repositories/memory_repository.py). A tenant with many
accumulated facts no longer has all of them injected into every question,
only the ones that actually bear on it. Each newly extracted fact is now
embedded before it's stored, so it's retrievable this way going forward;
embedding a fact or a query is just another OpenAI call, so every step
here fails open the same way reranking/generation/contextualization/
extraction already do -- falling back to Day 14's flat, most-recent-facts
behavior rather than ever failing the request over it.

Day 16: three real costs introduced by Days 13-15 (contextualization,
memory extraction, memory embeddings) were never counted in
`AskResponse.cost` or the Day 10 per-tenant daily budget until now.
`_contextualize`, `_fetch_memory_notes`, and `_maybe_extract_memories` all
now return their usage/token counts alongside their actual result, and
`_extra_cost_fields` turns that into the same `CostBreakdown` shape the
rest of the pipeline already uses, folded into the final response (and the
budget check) by `_apply_extra_costs` / `_apply_extra_costs_dict` before
`record_tenant_spend` is ever called. Each of the three calls that can be
skipped (no history, no memories, a refusal) correctly contributes zero
cost when skipped -- this is pure accounting, it changes nothing about
WHEN those calls happen, only makes their cost visible once they do.
Matching `tracing.span()`s (`contextualize`, `memory_retrieve`,
`memory_extract`) close the Day 13/14/15 observability gap the same way.

Day 17: Day 16 folded the three extra costs into `AskResponse.cost` and the
tenant budget, but never told `cost_tracker` (app/observability/metrics.py)
about them -- so `/observability/summary` and its dashboard kept silently
undercounting "total spend" for any request that touched a conversation or
memory. `_record_extra_stage_costs` feeds the same three categories into
`cost_tracker`, stage names matching the tracing spans above 1:1, with the
same "only record a stage when tokens were actually spent" rule
rag_pipeline.py already follows for embedding/rerank/generation.

Day 19: `_fetch_memory_notes`'s question embedding now goes through the
SAME query-embedding cache `services/search.py` already uses for document
retrieval (`ChunkRepository.get_query_embedding`/`save_query_embedding`,
keyed by sha256(question)+model -- see that module's docstring for why
sharing this cache across tenants and across features is safe). The
identical question asked twice -- whether on a later day, or even within
ONE request where retrieval and memory lookup embed the same raw question
-- now costs a real OpenAI call only once. Flagged as open since
docs/day15.md; newly extracted facts (`_maybe_extract_memories`) are NOT
cached, since a fact is essentially never re-embedded on purpose.
"""

import hashlib
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from fastapi.responses import StreamingResponse

from app.auth import TenantContext, enforce_tenant_limits, record_tenant_spend
from app.config import Settings, get_settings
from app.dependencies import (
    ContextualizerFactory, EmbedderFactory, GeneratorFactory, MemoryExtractorFactory, RerankerFactory,
    StreamingGeneratorFactory, get_contextualizer_factory, get_conversation_repository, get_embedder_factory,
    get_generator_factory, get_memory_extractor_factory, get_memory_repository, get_reranker_factory,
    get_repository, get_streaming_generator_factory, get_tenant_index_cache, get_tenant_repository,
    get_token_counter,
)
from app.models.ask import AskRequest, AskResponse, CostBreakdown
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.tenant_repository import TenantRepository
from app.services.chunker import TokenCounter
from app.services.cost import estimate_cost_usd
from app.services.embedder import EmbeddingConfigError, EmbeddingError
from app.services.llm import LLMError, LLMUsage
from app.services.rag_pipeline import RagDeps, answer_question, answer_question_stream
from app.observability.metrics import cost_tracker
from app.observability.tracing import get_trace_id, span
from app.services.search import SearchService
from app.services.vector_index_cache import TenantIndexCache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ask", tags=["ask"])


async def _contextualize(
    body: AskRequest, tenant_id: str, conversation_repo: ConversationRepository,
    contextualizer_factory: ContextualizerFactory, settings: Settings,
) -> tuple[str | None, LLMUsage]:
    """Returns the rewritten standalone search query (or None when no
    conversation_id was given -- the ordinary single-turn case) alongside
    the LLM usage that produced it, Day 16's cost-accounting addition.
    Every skip path (no conversation_id, no API key, a failed call)
    reports zero usage, exactly matching "no LLM call was actually made."
    Raises 404 if conversation_id doesn't exist or isn't owned by this
    tenant.

    Contextualization is an EXTRA LLM call on top of the existing rerank +
    generation calls -- if it fails (provider down, network blip), that
    failure must not take down the whole /ask request. Same fail-open
    philosophy as get_reranker_factory/get_generator_factory: fall back to
    the raw question (retrieval on a follow-up may be worse without the
    rewrite, but the endpoint stays up) rather than raising."""
    if body.conversation_id is None:
        return None, LLMUsage()
    conversation_id = str(body.conversation_id)
    owner = conversation_repo.get_owner(conversation_id)
    if owner is None or owner != tenant_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    history = conversation_repo.get_turns(conversation_id, limit=settings.contextualizer_max_turns)
    contextualizer = contextualizer_factory()
    if contextualizer is None:
        return body.question, LLMUsage()  # no API key configured; fall back to the raw question rather than failing
    try:
        with span("contextualize"):
            return await contextualizer.contextualize(body.question, history)
    except LLMError:
        logger.warning("contextualizer_failed", extra={"conversation_id": conversation_id})
        return body.question, LLMUsage()


def _append_turn(body: AskRequest, conversation_repo: ConversationRepository, result: AskResponse) -> None:
    if body.conversation_id is None:
        return
    conversation_repo.append_turn(
        str(body.conversation_id), question=result.question, search_query=result.search_query or result.question,
        answer=result.answer, citations=[c.model_dump() for c in result.citations],
        refusal_reason=result.refusal_reason,
    )


async def _embed_text(
    text: str, embedder_factory: EmbedderFactory, *, cache_repo: ChunkRepository | None = None,
    cache_model: str | None = None, cache_max_rows: int | None = None,
) -> tuple[list[float] | None, int]:
    """Shared by query-time retrieval and fact storage below. Returns
    (None, 0) on ANY failure -- missing API key (EmbeddingConfigError,
    raised by the factory itself) or a live provider failure
    (EmbeddingError, raised by .embed()) -- so every caller has exactly one
    fallback to write: treat a None embedding as "use the Day 14 flat-list
    behavior instead", never a reason to fail the /ask request this is a
    side-effect of. The token count (0 on failure) is Day 16's addition,
    folded into `AskResponse.cost` by the caller.

    Day 19: when `cache_repo`/`cache_model` are given (only `_fetch_memory_
    notes`'s question embedding passes them -- see that function), this
    reuses `services/search.py`'s own query-embedding cache
    (`ChunkRepository.get_query_embedding`/`save_query_embedding`, keyed by
    sha256(text)+model) instead of always paying for a real embed call. A
    cache hit returns 0 tokens, matching the real cost of "no call was
    made" -- the exact same convention every other skip path here already
    follows. Fact embedding deliberately does NOT pass these (a newly
    extracted fact is new text, essentially never a repeat, so caching it
    would just grow the table with rows that are never read back)."""
    qhash = hashlib.sha256(text.encode("utf-8")).hexdigest() if cache_repo is not None else None
    if cache_repo is not None and qhash is not None:
        cached = await run_in_threadpool(cache_repo.get_query_embedding, qhash, cache_model)
        if cached is not None:
            return cached, 0
    try:
        embedder = embedder_factory()
        result = await embedder.embed([text])
    except (EmbeddingConfigError, EmbeddingError):
        return None, 0
    if not result.vectors:
        return None, 0
    vector = result.vectors[0]
    if cache_repo is not None and qhash is not None:
        await run_in_threadpool(cache_repo.save_query_embedding, qhash, cache_model, vector, cache_max_rows)
    return vector, result.total_tokens


async def _fetch_memory_notes(
    question: str, tenant_id: str, memory_repo: MemoryRepository, embedder_factory: EmbedderFactory,
    settings: Settings, cache_repo: ChunkRepository,
) -> tuple[list[str], int]:
    """Day 15: only the `memory_top_k` facts most RELEVANT to this question
    (cosine similarity over embedded facts), not every stored fact. Cost +
    noise discipline, same shape as Day 13's contextualizer skipping a
    conversation's first turn: skip the embedding call entirely when this
    tenant has no memories at all yet (`memory_repo.count`), rather than
    paying for an embedding with nothing to compare it against. Falls back
    to Day 14's flat, most-recent-facts list when embedding the question
    fails, or when no stored fact has an embedding yet (e.g. it predates
    Day 15 and hasn't been re-stated since). The second return value
    (Day 16) is the embedding token count actually spent -- zero on every
    skip or fallback path, since none of those make a billed call (Day 19:
    now also zero on a CACHE HIT, since that's a real call that simply
    didn't need to happen again -- see `_embed_text`)."""
    if memory_repo.count(tenant_id) == 0:
        return [], 0
    fallback = [m.content for m in memory_repo.list_memories(tenant_id, limit=settings.memory_max_facts)]
    with span("memory_retrieve"):
        qvec, tokens = await _embed_text(
            question, embedder_factory, cache_repo=cache_repo,
            cache_model=settings.openai_embedding_model, cache_max_rows=settings.query_cache_max_rows,
        )
    if qvec is None:
        return fallback, 0
    relevant = memory_repo.search_memories(tenant_id, qvec, top_k=settings.memory_top_k)
    return ([m.content for m in relevant] if relevant else fallback), tokens


async def _maybe_extract_memories(
    *, question: str, answer: str, refusal_reason: str | None, tenant_id: str, memory_repo: MemoryRepository,
    memory_extractor_factory: MemoryExtractorFactory, embedder_factory: EmbedderFactory, settings: Settings,
    existing_notes: list[str],
) -> tuple[LLMUsage, int]:
    """An EXTRA LLM call made AFTER the request's own work is already done
    -- a failure here must never turn an otherwise-successful /ask into an
    error (same fail-open philosophy as _contextualize). Skipped entirely
    on a refusal: no real exchange happened for there to be anything
    durable to learn from. Returns the extraction call's own usage plus the
    total embedding tokens spent on any newly stored facts (Day 16) -- both
    zero when extraction is skipped or fails, matching the real cost of
    "nothing was actually called"."""
    if refusal_reason is not None:
        return LLMUsage(), 0
    extractor = memory_extractor_factory()
    if extractor is None:
        return LLMUsage(), 0
    try:
        with span("memory_extract"):
            facts, usage = await extractor.extract(question, answer, existing_notes)
    except LLMError:
        logger.warning("memory_extraction_failed", extra={"tenant_id": tenant_id})
        return LLMUsage(), 0
    embedding_tokens = 0
    for fact in facts:
        # Embedding failure here only means this fact stays in the Day 14
        # flat fallback until it's re-stated (add_memory backfills the
        # embedding then) -- never a reason to drop the fact entirely.
        embedding, tokens = await _embed_text(fact, embedder_factory)
        embedding_tokens += tokens
        memory_repo.add_memory(tenant_id, fact, max_memories=settings.memory_max_facts, embedding=embedding)
    return usage, embedding_tokens


def _extra_cost_fields(
    settings: Settings, contextualizer_usage: LLMUsage, memory_query_tokens: int,
    extraction_usage: LLMUsage, new_fact_embedding_tokens: int,
) -> dict:
    """Day 16: turns the three previously-uncounted costs (contextualizer,
    memory extraction, memory embeddings) into the same shape
    `CostBreakdown` already uses for retrieval/rerank/generation, so they
    can be merged into an `AskResponse.cost` that already exists (the
    non-streaming path) or into its dict form (the streaming path) with
    the same arithmetic either way -- see `_apply_extra_costs` /
    `_apply_extra_costs_dict` below."""
    contextualizer_cost = (
        estimate_cost_usd(contextualizer_usage.prompt_tokens, settings.chat_price_input_per_million)
        + estimate_cost_usd(contextualizer_usage.completion_tokens, settings.chat_price_output_per_million)
    )
    extraction_cost = (
        estimate_cost_usd(extraction_usage.prompt_tokens, settings.chat_price_input_per_million)
        + estimate_cost_usd(extraction_usage.completion_tokens, settings.chat_price_output_per_million)
    )
    memory_embedding_tokens = memory_query_tokens + new_fact_embedding_tokens
    memory_embedding_cost = estimate_cost_usd(memory_embedding_tokens, settings.embedding_price_per_million_tokens)
    return {
        "contextualizer_prompt_tokens": contextualizer_usage.prompt_tokens,
        "contextualizer_completion_tokens": contextualizer_usage.completion_tokens,
        "contextualizer_cost_usd": contextualizer_cost,
        "memory_extraction_prompt_tokens": extraction_usage.prompt_tokens,
        "memory_extraction_completion_tokens": extraction_usage.completion_tokens,
        "memory_extraction_cost_usd": extraction_cost,
        "memory_embedding_tokens": memory_embedding_tokens,
        "memory_embedding_cost_usd": memory_embedding_cost,
    }


def _apply_extra_costs(cost: CostBreakdown, extra: dict) -> None:
    """Non-streaming path: `result.cost` is already a real `CostBreakdown`
    instance, so the extra fields are set directly and the total is
    recomputed from every component -- the exact same summation
    `rag_pipeline._refuse`/`answer_question` already do, just extended with
    the three new terms."""
    for key, value in extra.items():
        setattr(cost, key, value)
    cost.total_cost_usd = round(
        cost.embedding_cost_usd + cost.rerank_cost_usd + cost.generation_cost_usd
        + cost.contextualizer_cost_usd + cost.memory_extraction_cost_usd + cost.memory_embedding_cost_usd, 8,
    )


def _apply_extra_costs_dict(cost: dict, extra: dict) -> None:
    """Streaming path: the "done" event's `cost` is a plain dict (already
    serialized via `model_dump(mode="json")` inside rag_pipeline), so the
    same merge is done dict-style -- identical arithmetic to
    `_apply_extra_costs` above, just without a Pydantic model to mutate."""
    cost.update(extra)
    cost["total_cost_usd"] = round(
        cost.get("embedding_cost_usd", 0.0) + cost.get("rerank_cost_usd", 0.0) + cost.get("generation_cost_usd", 0.0)
        + cost["contextualizer_cost_usd"] + cost["memory_extraction_cost_usd"] + cost["memory_embedding_cost_usd"], 8,
    )


def _record_extra_stage_costs(extra: dict) -> None:
    """Day 17: feeds the three Day 16 cost categories into the same
    in-process `cost_tracker` that rag_pipeline.py already uses for
    embedding/rerank/generation, so `/observability/summary` and the
    dashboard stop undercounting as soon as a conversation or memory is
    involved. Stage names (`contextualize`, `memory_extract`,
    `memory_embedding`) match the tracing spans of the same name one-to-
    one. Mirrors rag_pipeline.py's own rule: only record a stage when
    tokens were actually spent, so a skipped call (no history, no
    memories, a refusal) adds nothing -- not even a zero-cost entry."""
    ctx_tokens = extra["contextualizer_prompt_tokens"] + extra["contextualizer_completion_tokens"]
    if ctx_tokens:
        cost_tracker.record_cost("contextualize", ctx_tokens, extra["contextualizer_cost_usd"])
    extraction_tokens = extra["memory_extraction_prompt_tokens"] + extra["memory_extraction_completion_tokens"]
    if extraction_tokens:
        cost_tracker.record_cost("memory_extract", extraction_tokens, extra["memory_extraction_cost_usd"])
    if extra["memory_embedding_tokens"]:
        cost_tracker.record_cost("memory_embedding", extra["memory_embedding_tokens"], extra["memory_embedding_cost_usd"])


@router.post("", response_model=AskResponse)
async def ask(
    body: AskRequest,
    settings: Settings = Depends(get_settings),
    repo: ChunkRepository = Depends(get_repository),
    embedder_factory: EmbedderFactory = Depends(get_embedder_factory),
    reranker_factory: RerankerFactory = Depends(get_reranker_factory),
    generator_factory: GeneratorFactory = Depends(get_generator_factory),
    counter: TokenCounter = Depends(get_token_counter),
    tenant: TenantContext = Depends(enforce_tenant_limits),
    tenant_repo: TenantRepository = Depends(get_tenant_repository),
    index_cache: TenantIndexCache = Depends(get_tenant_index_cache),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    contextualizer_factory: ContextualizerFactory = Depends(get_contextualizer_factory),
    memory_repo: MemoryRepository = Depends(get_memory_repository),
    memory_extractor_factory: MemoryExtractorFactory = Depends(get_memory_extractor_factory),
) -> AskResponse:
    deps = RagDeps(
        search=SearchService(repo, embedder_factory, settings, index_cache),
        reranker=reranker_factory() if body.rerank else None,
        generator=generator_factory(),
        counter=counter, settings=settings,
    )
    search_query, ctx_usage = await _contextualize(body, tenant.tenant_id, conversation_repo, contextualizer_factory, settings)
    memory_notes, memory_query_tokens = await _fetch_memory_notes(
        body.question, tenant.tenant_id, memory_repo, embedder_factory, settings, repo
    )
    try:
        result = await answer_question(
            question=body.question, search_query=search_query,
            document_id=str(body.document_id) if body.document_id else None,
            tenant_id=tenant.tenant_id, top_k=body.top_k, use_rerank=body.rerank, deps=deps,
            request_id=get_trace_id(), memory_notes=memory_notes,
        )
        extraction_usage, new_fact_tokens = await _maybe_extract_memories(
            question=result.question, answer=result.answer, refusal_reason=result.refusal_reason,
            tenant_id=tenant.tenant_id, memory_repo=memory_repo, memory_extractor_factory=memory_extractor_factory,
            embedder_factory=embedder_factory, settings=settings, existing_notes=memory_notes,
        )
        extra = _extra_cost_fields(settings, ctx_usage, memory_query_tokens, extraction_usage, new_fact_tokens)
        _record_extra_stage_costs(extra)
        _apply_extra_costs(result.cost, extra)
        record_tenant_spend(tenant_repo, tenant.tenant_id, result.cost.total_cost_usd)
        _append_turn(body, conversation_repo, result)
        return result
    except EmbeddingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail="Embedding provider failed; try again later.") from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail="Answer generation failed; nothing was returned. Retry later.") from exc


@router.post("/stream")
async def ask_stream(
    body: AskRequest,
    settings: Settings = Depends(get_settings),
    repo: ChunkRepository = Depends(get_repository),
    embedder_factory: EmbedderFactory = Depends(get_embedder_factory),
    reranker_factory: RerankerFactory = Depends(get_reranker_factory),
    streaming_generator_factory: StreamingGeneratorFactory = Depends(get_streaming_generator_factory),
    counter: TokenCounter = Depends(get_token_counter),
    tenant: TenantContext = Depends(enforce_tenant_limits),
    tenant_repo: TenantRepository = Depends(get_tenant_repository),
    index_cache: TenantIndexCache = Depends(get_tenant_index_cache),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    contextualizer_factory: ContextualizerFactory = Depends(get_contextualizer_factory),
    memory_repo: MemoryRepository = Depends(get_memory_repository),
    memory_extractor_factory: MemoryExtractorFactory = Depends(get_memory_extractor_factory),
) -> StreamingResponse:
    """
    Server-Sent Events version of /ask. Event types: delta, correction
    (rare -- see rag_pipeline.answer_question_stream docstring), done
    (always last, carries the same shape as a non-streaming AskResponse).

    curl example:
        curl -N -X POST http://localhost:8000/ask/stream \
             -H "Authorization: Bearer <your-api-key>" \
             -H "Content-Type: application/json" \
             -d '{"question": "...", "document_id": "..."}'
    """
    deps = RagDeps(
        search=SearchService(repo, embedder_factory, settings, index_cache),
        reranker=reranker_factory() if body.rerank else None,
        generator=None,
        streaming_generator=streaming_generator_factory(),
        counter=counter, settings=settings,
    )
    document_id = str(body.document_id) if body.document_id else None
    request_id = get_trace_id()
    # Resolved before the SSE response starts: a 404 for an unowned/missing
    # conversation_id must be a real HTTP status, not an "error" SSE event
    # (the response is already committed to 200 once streaming begins).
    search_query, ctx_usage = await _contextualize(body, tenant.tenant_id, conversation_repo, contextualizer_factory, settings)
    memory_notes, memory_query_tokens = await _fetch_memory_notes(
        body.question, tenant.tenant_id, memory_repo, embedder_factory, settings, repo
    )

    async def event_source():
        try:
            async for event in answer_question_stream(
                question=body.question, search_query=search_query, document_id=document_id,
                tenant_id=tenant.tenant_id, top_k=body.top_k, use_rerank=body.rerank, deps=deps,
                request_id=request_id, memory_notes=memory_notes,
            ):
                event_type = event["event"]
                if event_type == "delta":
                    payload = {"text": event["text"]}
                elif event_type == "correction":
                    payload = {"message": event["message"]}
                else:  # "done"
                    payload = event["data"]
                    if body.conversation_id is not None:
                        conversation_repo.append_turn(
                            str(body.conversation_id), question=payload["question"],
                            search_query=payload.get("search_query") or payload["question"],
                            answer=payload["answer"], citations=payload.get("citations", []),
                            refusal_reason=payload.get("refusal_reason"),
                        )
                    extraction_usage, new_fact_tokens = await _maybe_extract_memories(
                        question=payload["question"], answer=payload["answer"],
                        refusal_reason=payload.get("refusal_reason"), tenant_id=tenant.tenant_id,
                        memory_repo=memory_repo, memory_extractor_factory=memory_extractor_factory,
                        embedder_factory=embedder_factory, settings=settings, existing_notes=memory_notes,
                    )
                    extra = _extra_cost_fields(settings, ctx_usage, memory_query_tokens, extraction_usage, new_fact_tokens)
                    _record_extra_stage_costs(extra)
                    _apply_extra_costs_dict(payload.setdefault("cost", {}), extra)
                    record_tenant_spend(tenant_repo, tenant.tenant_id, payload["cost"].get("total_cost_usd", 0))
                yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        except (EmbeddingConfigError, EmbeddingError, LLMError) as exc:
            # A failure before or during retrieval that the generator-level
            # try/except inside answer_question_stream doesn't cover (e.g.
            # the embedding call itself fails). The HTTP status is already
            # committed to 200 once streaming has started, so the error
            # must be communicated as an SSE event, not an HTTP status code.
            yield f"event: error\ndata: {json.dumps({'detail': str(exc) or type(exc).__name__})}\n\n"

    return StreamingResponse(
        event_source(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
