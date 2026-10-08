"""
Orchestrates: hybrid search -> rerank (optional) -> evidence gate -> context
build -> grounded generation -> citation validation -> cost rollup.

Refusal is a FEATURE, tracked by `refusal_reason`, not an exception:
  no_results                  - search returned nothing
  low_relevance                - best reranked score is below threshold
  model_insufficient_evidence  - model itself said the sources don't answer it (JSON mode only, see below)
  ungrounded_answer            - model answered but cited nothing verifiable (rejected, not shown)

Two entry points share the retrieval/rerank/gate stage (`_retrieve_and_prepare`):
  answer_question         - non-streaming, JSON-mode generation (Day 4)
  answer_question_stream  - streaming, plain-text generation (Day 8)
They diverge only at generation, because streaming and structured JSON output
are in real tension -- see services/llm.py's OpenAIStreamingLLM docstring.
One consequence: the streaming path cannot distinguish "model explicitly
said insufficient evidence" from "model just didn't cite anything" the way
JSON mode can (no separate boolean field exists in plain text) -- both
collapse into `ungrounded_answer`. This is a real, documented capability
difference between the two endpoints, not an oversight.
"""

import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from app.config import Settings
from app.models.ask import AskResponse, Citation, CostBreakdown
from app.services.chunker import TokenCounter
from app.services.citations import validate_citations
from app.services.context_builder import Source, build_sources
from app.services.cost import estimate_cost_usd
from app.services.generator import Generator, StreamingGenerator
from app.services.llm import LLMError
from app.observability.metrics import cost_tracker
from app.observability.tracing import span
from app.services.reranker import Reranker
from app.services.search import SearchService

logger = logging.getLogger(__name__)

NO_RESULTS = "no_results"
LOW_RELEVANCE = "low_relevance"
MODEL_INSUFFICIENT = "model_insufficient_evidence"
UNGROUNDED = "ungrounded_answer"

_REFUSAL_MESSAGE = "I don't have enough information in the indexed documents to answer that."


@dataclass
class RagDeps:
    search: SearchService
    reranker: Reranker | None
    generator: Generator | None
    counter: TokenCounter
    settings: Settings
    streaming_generator: StreamingGenerator | None = None


@dataclass
class _Prepared:
    sources: list[Source]
    reranked: bool
    warnings: list[str]
    cost: CostBreakdown
    timings: dict[str, float]
    models: dict[str, str]
    degraded: bool


async def _retrieve_and_prepare(
    *, question: str, search_query: str | None, document_id: str | None, tenant_id: str, top_k: int,
    use_rerank: bool, deps: RagDeps, request_id: str,
) -> tuple[_Prepared | None, AskResponse | None]:
    """Runs retrieval, optional reranking, and the relevance gate -- shared
    by both the streaming and non-streaming entry points. Returns EITHER a
    `_Prepared` bundle (ready for generation) OR a fully-formed early-
    refusal `AskResponse` (nothing left to generate).

    `search_query` (Day 13): what actually gets embedded/searched. Defaults
    to `question` when not given (the single-turn case). When a multi-turn
    caller has contextualized a follow-up into a standalone query (see
    services/contextualizer.py), `search_query` is that rewritten text --
    `question` stays the user's real wording for generation and display."""
    timings: dict[str, float] = {}
    warnings: list[str] = []
    cost = CostBreakdown()
    s = deps.settings
    t0 = time.perf_counter()
    effective_query = search_query or question

    t = time.perf_counter()
    with span("retrieval", document_id=document_id or "all"):
        search_result = await deps.search.search(
            query=effective_query, mode="hybrid", top_k=s.retrieve_k, document_id=document_id,
            tenant_id=tenant_id, request_id=request_id,
        )
    timings["retrieval"] = _ms(t)
    warnings.extend(search_result.warnings)
    cost.embedding_tokens = search_result.query_tokens
    cost.embedding_cost_usd = search_result.query_cost_usd
    if search_result.query_tokens:
        cost_tracker.record_cost("embedding", search_result.query_tokens, search_result.query_cost_usd)
        cost_tracker.record_cache("query", hit=False)
    else:
        cost_tracker.record_cache("query", hit=True)

    if not search_result.hits:
        cost_tracker.record_refusal(NO_RESULTS)
        return None, _refuse(question, effective_query, NO_RESULTS, warnings, cost, timings, t0,
                              {"embedding": s.openai_embedding_model})

    hits = search_result.hits
    scores: dict[str, float] | None = None
    reranked = False
    if use_rerank and deps.reranker is not None:
        try:
            t = time.perf_counter()
            with span("rerank", candidates=len(hits)):
                rr = await deps.reranker.rerank(question, [(h.chunk_id, h.text) for h in hits])
            timings["rerank"] = _ms(t)
            scores = rr.scores
            reranked = True
            cost.rerank_prompt_tokens = rr.usage.prompt_tokens
            cost.rerank_completion_tokens = rr.usage.completion_tokens
            cost.rerank_cost_usd = estimate_cost_usd(rr.usage.prompt_tokens, s.chat_price_input_per_million) \
                + estimate_cost_usd(rr.usage.completion_tokens, s.chat_price_output_per_million)
            cost_tracker.record_cost("rerank", rr.usage.prompt_tokens + rr.usage.completion_tokens, cost.rerank_cost_usd)
            hits = sorted(hits, key=lambda h: scores.get(h.chunk_id, 0.0), reverse=True)
        except LLMError as exc:
            warnings.append(f"Reranker unavailable ({type(exc).__name__}); used hybrid ranking instead.")
            logger.warning("rerank_failed", extra={"request_id": request_id, "error_type": type(exc).__name__})

    # The relevance gate only fires post-rerank: the reranker's 0-10 scale is
    # calibrated for this purpose. Hybrid's own score is an RRF fusion value
    # (small, rank-based) and vector cosine is on yet another scale, so
    # neither is comparable to a single fixed threshold. Without a rerank
    # score, we rely on the generator's own signal instead of guessing a
    # cross-scale cutoff.
    if reranked:
        best = scores.get(hits[0].chunk_id, 0.0)
        if best < s.min_rerank_score:
            models = {"embedding": s.openai_embedding_model, "rerank": s.openai_rerank_model}
            cost_tracker.record_refusal(LOW_RELEVANCE)
            return None, _refuse(question, effective_query, LOW_RELEVANCE, warnings, cost, timings, t0, models)

    sources = build_sources(hits, deps.counter, max_sources=top_k, max_tokens=s.max_context_tokens, scores=scores)
    models = {"embedding": s.openai_embedding_model, "rerank": s.openai_rerank_model if reranked else "none"}
    return _Prepared(sources, reranked, warnings, cost, timings, models, search_result.degraded), None


async def answer_question(
    *, question: str, search_query: str | None = None, document_id: str | None, tenant_id: str, top_k: int,
    use_rerank: bool, deps: RagDeps, request_id: str, memory_notes: list[str] | None = None,
) -> AskResponse:
    t0 = time.perf_counter()
    prepared, early_refusal = await _retrieve_and_prepare(
        question=question, search_query=search_query, document_id=document_id, tenant_id=tenant_id,
        top_k=top_k, use_rerank=use_rerank, deps=deps, request_id=request_id,
    )
    if early_refusal is not None:
        return early_refusal
    s = deps.settings
    effective_query = search_query or question
    warnings, cost, timings, models = prepared.warnings, prepared.cost, prepared.timings, dict(prepared.models)

    if deps.generator is None:
        return _refuse(question, effective_query, LOW_RELEVANCE, warnings + ["Generator unavailable (no API key)."],
                        cost, timings, t0, {"embedding": s.openai_embedding_model})

    t = time.perf_counter()
    with span("generation", sources=len(prepared.sources)):
        # LLMError propagates to the router (-> 502). memory_notes (Day 14)
        # are long-term facts from PAST conversations -- see
        # services/generator.format_memory_notes for why they're never
        # treated as citable, grounded source content.
        gen = await deps.generator.generate(question, prepared.sources, memory_notes=memory_notes)
    timings["generation"] = _ms(t)
    cost.generation_prompt_tokens = gen.usage.prompt_tokens
    cost.generation_completion_tokens = gen.usage.completion_tokens
    cost.generation_cost_usd = estimate_cost_usd(gen.usage.prompt_tokens, s.chat_price_input_per_million) \
        + estimate_cost_usd(gen.usage.completion_tokens, s.chat_price_output_per_million)
    cost.total_cost_usd = round(cost.embedding_cost_usd + cost.rerank_cost_usd + cost.generation_cost_usd, 8)
    models["generation"] = s.openai_chat_model
    cost_tracker.record_cost("generation", gen.usage.prompt_tokens + gen.usage.completion_tokens, cost.generation_cost_usd)

    if gen.insufficient_evidence or not gen.answer.strip():
        cost_tracker.record_refusal(MODEL_INSUFFICIENT)
        return _refuse(question, effective_query, MODEL_INSUFFICIENT, warnings, cost, timings, t0, models)

    valid_ids = {s_.source_id for s_ in prepared.sources}
    validated = validate_citations(gen.answer, gen.cited_ids, valid_ids)
    if not validated.cited_ids:
        logger.warning("ungrounded_answer_rejected", extra={"request_id": request_id})
        cost_tracker.record_refusal(UNGROUNDED)
        return _refuse(question, effective_query, UNGROUNDED, warnings, cost, timings, t0, models)
    if validated.invalid_ids:
        warnings.append(f"Model cited unknown source id(s) {validated.invalid_ids}; they were removed.")

    citations = _build_citations(prepared.sources, validated.cited_ids)
    timings["total"] = _ms(t0)
    logger.info("ask_answered", extra={"request_id": request_id, "duration_ms": timings["total"],
                                        "cost_usd": cost.total_cost_usd})
    return AskResponse(
        question=question, search_query=effective_query if effective_query != question else None,
        answer=validated.text, insufficient_evidence=False, refusal_reason=None,
        citations=citations, degraded=prepared.degraded, warnings=warnings,
        cost=cost, timings_ms=timings, models=models,
    )


async def answer_question_stream(
    *, question: str, search_query: str | None = None, document_id: str | None, tenant_id: str, top_k: int,
    use_rerank: bool, deps: RagDeps, request_id: str, memory_notes: list[str] | None = None,
) -> AsyncIterator[dict]:
    """
    Async generator yielding SSE-ready event dicts:
      {"event": "delta", "text": "..."}                  -- zero or more, in order
      {"event": "correction", "message": "..."}           -- at most one, ONLY if the
                                                             streamed text must be replaced
      {"event": "done", "data": <AskResponse as a dict>}  -- always last, exactly once

    The "correction" event is the real UX trade-off streaming + citation
    validation creates: citations can only be validated once the FULL text
    has arrived (a citation marker could be split across two chunks, and an
    invalid id can't be known until the whole answer is seen). If the
    finished text turns out to have zero valid citations, we cannot "un-
    stream" what the client already rendered -- so we emit an explicit
    correction event telling the client to replace the displayed text with
    the refusal message, rather than silently leaving a wrong answer on
    screen. A real UI should treat "correction" as "swap the bubble's
    content", not append to it. See docs/day8.md for the full discussion.
    """
    t0 = time.perf_counter()
    prepared, early_refusal = await _retrieve_and_prepare(
        question=question, search_query=search_query, document_id=document_id, tenant_id=tenant_id,
        top_k=top_k, use_rerank=use_rerank, deps=deps, request_id=request_id,
    )
    if early_refusal is not None:
        yield {"event": "done", "data": early_refusal.model_dump(mode="json")}
        return

    s = deps.settings
    effective_query = search_query or question
    warnings, cost, timings, models = prepared.warnings, prepared.cost, prepared.timings, dict(prepared.models)

    if deps.streaming_generator is None:
        refusal = _refuse(question, effective_query, LOW_RELEVANCE, warnings + ["Generator unavailable (no API key)."],
                           cost, timings, t0, {"embedding": s.openai_embedding_model})
        yield {"event": "done", "data": refusal.model_dump(mode="json")}
        return

    t = time.perf_counter()
    full_text = ""
    try:
        with span("generation_stream", sources=len(prepared.sources)):
            async for chunk in deps.streaming_generator.generate_stream(
                question, prepared.sources, memory_notes=memory_notes
            ):
                if chunk.text:
                    full_text += chunk.text
                    yield {"event": "delta", "text": chunk.text}
                if chunk.usage:
                    cost.generation_prompt_tokens = chunk.usage.prompt_tokens
                    cost.generation_completion_tokens = chunk.usage.completion_tokens
    except LLMError as exc:
        logger.error("stream_failed", extra={"request_id": request_id, "error_type": type(exc).__name__})
        cost_tracker.record_refusal(LOW_RELEVANCE)
        refusal = _refuse(question, effective_query, LOW_RELEVANCE,
                           warnings + [f"Generation failed mid-stream ({type(exc).__name__})."],
                           cost, timings, t0, models)
        yield {"event": "correction", "message": refusal.answer}
        yield {"event": "done", "data": refusal.model_dump(mode="json")}
        return
    timings["generation"] = _ms(t)
    cost.generation_cost_usd = estimate_cost_usd(cost.generation_prompt_tokens, s.chat_price_input_per_million) \
        + estimate_cost_usd(cost.generation_completion_tokens, s.chat_price_output_per_million)
    cost.total_cost_usd = round(cost.embedding_cost_usd + cost.rerank_cost_usd + cost.generation_cost_usd, 8)
    models["generation"] = s.openai_chat_model
    cost_tracker.record_cost("generation", cost.generation_prompt_tokens + cost.generation_completion_tokens,
                              cost.generation_cost_usd)

    valid_ids = {s_.source_id for s_ in prepared.sources}
    validated = validate_citations(full_text, [], valid_ids)
    if validated.invalid_ids:
        warnings.append(f"Model cited unknown source id(s) {validated.invalid_ids}; they were removed.")

    if not validated.cited_ids:
        # Plain-text mode has no separate "insufficient_evidence" boolean
        # (that's a JSON-mode-only field, see module docstring) -- a
        # citation-free answer is refused uniformly as ungrounded_answer,
        # whether the model explicitly declined or simply failed to cite.
        logger.warning("stream_ungrounded_answer_rejected", extra={"request_id": request_id})
        cost_tracker.record_refusal(UNGROUNDED)
        refusal = _refuse(question, effective_query, UNGROUNDED, warnings, cost, timings, t0, models)
        yield {"event": "correction", "message": refusal.answer}
        yield {"event": "done", "data": refusal.model_dump(mode="json")}
        return

    citations = _build_citations(prepared.sources, validated.cited_ids)
    timings["total"] = _ms(t0)
    if validated.text != full_text:
        # An invalid citation was stripped from the text: what the client
        # already rendered via deltas no longer matches exactly. Flagged
        # here explicitly rather than silently -- see docs/day8.md.
        warnings.append("The streamed text was lightly edited after validation (an invalid citation was removed).")
    logger.info("ask_stream_answered", extra={"request_id": request_id, "duration_ms": timings["total"],
                                               "cost_usd": cost.total_cost_usd})
    final = AskResponse(
        question=question, search_query=effective_query if effective_query != question else None,
        answer=validated.text, insufficient_evidence=False, refusal_reason=None,
        citations=citations, degraded=prepared.degraded, warnings=warnings,
        cost=cost, timings_ms=timings, models=models,
    )
    yield {"event": "done", "data": final.model_dump(mode="json")}


def _build_citations(sources: list[Source], cited_ids: list[str]) -> list[Citation]:
    by_id = {s_.source_id: s_ for s_ in sources}
    return [
        Citation(source_id=sid, chunk_id=by_id[sid].chunk_id, document_id=by_id[sid].document_id,
                 page_number=by_id[sid].page_number, snippet=by_id[sid].text[:240],
                 rerank_score=by_id[sid].rerank_score)
        for sid in cited_ids
    ]


def _refuse(question, effective_query, reason, warnings, cost, timings, t0, models) -> AskResponse:
    cost.total_cost_usd = round(cost.embedding_cost_usd + cost.rerank_cost_usd + cost.generation_cost_usd, 8)
    timings["total"] = _ms(t0)
    return AskResponse(
        question=question, search_query=effective_query if effective_query != question else None,
        answer=_REFUSAL_MESSAGE, insufficient_evidence=True, refusal_reason=reason,
        citations=[], warnings=warnings, cost=cost, timings_ms=timings, models=models,
    )


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)
