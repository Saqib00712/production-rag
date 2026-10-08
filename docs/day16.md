# Day 16: Full Cost + Trace Accounting for Memory and Contextualization

## 1. Objective
Close the cost and observability gap that's been flagged, tracked, and deliberately left open across three
straight days: contextualization (Day 13), memory extraction (Day 14), and memory embeddings (Day 15) were all
real OpenAI calls that never showed up in `AskResponse.cost`, never counted against the Day 10 per-tenant daily
budget, and never got their own `tracing.span()`. Today folds all three into both systems in one pass, exactly as
previewed at the end of `docs/day15.md`.

## 2. Why this matters
A tenant's "daily budget" (Day 10) is supposed to be a hard safety ceiling on real spend. Through Day 15, it
quietly wasn't: every multi-turn conversation with a follow-up paid for a contextualizer call that never touched
the ledger, every answered question paid for a memory-extraction call that never touched the ledger, and every
question asked by a tenant with existing memories paid for an embedding call that never touched the ledger. None
of these were large individually, but a budget that silently undercounts real spend isn't a safety ceiling, it's
a false sense of one -- and `AskResponse.cost`, the one place a caller can see what a single `/ask` call actually
cost, was telling the same incomplete story. This is purely an accounting fix: nothing about WHEN any of these
three calls happen changes today, only whether their cost is visible and counted once they do.

## 3. Theory
**Why this had to wait until Day 15, not get fixed incrementally on Day 13 or 14.** Each of the three calls was
added on a different day, by a different service, returning a different shape (a string, then a list of strings,
then a list of strings again) that threw its own `LLMUsage`/token count away rather than returning it. Fixing one
in isolation on the day it was introduced would have meant redesigning `CostBreakdown` three separate times,
once per day, for a feature whose shape nobody could have fully predicted on Day 13. Doing it once, now that all
three extra calls exist and their shapes are settled, means one coherent design instead of three partial ones.

**Why usage flows all the way up through return values, not a side channel.** `OpenAIContextualizer.
contextualize()` and `OpenAIMemoryExtractor.extract()` both already called `JsonLLM.complete_json()`, which
already returned `(data, usage)` -- the usage was computed and then silently discarded (`data, _usage = ...`)
on both days. The fix is mechanical and consistent with how every other cost-bearing call in this project works
(`OpenAIReranker.rerank()` and `OpenAIGenerator.generate()` have always returned their usage alongside their
result): both Protocols now return `tuple[result, LLMUsage]` instead of bolting a mutable "cost so far" parameter
onto every signature, or reaching into a global/contextvar accumulator that would make unit-testing these
services in isolation (tests/test_contextualizer.py, tests/test_memory_extractor.py) harder, not easier.

**Why the embedding token count flows through `_embed_text`'s return value, not a separate lookup.**
`EmbeddingBatchResult` (Day 2) already carries `total_tokens` from every embed call; Day 15's `_embed_text`
helper just wasn't passing it along. Same fix, same shape: `_embed_text` now returns `(vector | None, tokens)`
instead of just the vector, so both of its two call sites (`_fetch_memory_notes`'s question embedding, `_maybe_
extract_memories`'s per-fact embedding) get the number without a second round of bookkeeping.

**Why the skip paths matter as much as the real ones.** The whole point of folding these into cost accounting is
that the ZERO cases have to actually be zero, not just unset fields that happen to default to zero. A
conversation's first turn (no history to contextualize), a refusal (extraction skipped entirely), a tenant with
no memories yet (embedding skipped entirely) -- every one of these paths now explicitly returns `LLMUsage()` /
`0` rather than omitting a value and relying on a default, because "this call was never made" and "this call was
made and happened to cost nothing" are different facts, and only the former is actually true on these paths. The
test suite's cost tests (`test_cost_accounting.py`) assert both directions for exactly this reason: the real cost
is nonzero when the call fires, AND it's exactly zero when it doesn't.

**Why `_extra_cost_fields` is one pure function, shared by two apply sites.** The non-streaming `/ask` response
already exists as a `CostBreakdown` Pydantic instance by the time these extras are known; the streaming `/ask/
stream` "done" event is already a plain dict (post-`model_dump`) by that same point. Rather than duplicate the
arithmetic (contextualizer cost + extraction cost + embedding cost, then a new grand total) once for each shape,
`_extra_cost_fields` computes the values once as a plain dict, and `_apply_extra_costs`/`_apply_extra_costs_dict`
are two one-line-different adapters that merge that dict into whichever shape they're given. A bug in the
underlying arithmetic can only exist in one place, not two that could silently drift apart.

**Why `record_tenant_spend` moved to AFTER extraction, not before.** Through Day 15, the non-streaming `/ask`
recorded spend immediately after `answer_question` returned, then ran extraction afterward -- meaning extraction's
own cost (once it existed) would need a SECOND `record_tenant_spend` call, or get silently dropped. Day 16
reorders this: extraction now runs, its usage is folded into `result.cost` alongside the contextualizer and
embedding costs already known, and `record_tenant_spend` is called exactly once, with the complete total. One
accounting pass, one ledger write, not two that could get out of sync.

## 4. Architecture
```
app/services/contextualizer.py
  Contextualizer.contextualize(question, history) -> (search_query, LLMUsage)   <- was: -> search_query

app/services/memory_extractor.py
  MemoryExtractor.extract(question, answer, existing) -> (facts, LLMUsage)      <- was: -> facts

app/routers/ask.py
  _embed_text(text, embedder_factory) -> (vector | None, tokens)                <- was: -> vector | None
  _contextualize(...) -> (search_query | None, LLMUsage)                        <- was: -> search_query | None
    wraps the actual contextualizer call in span("contextualize")
  _fetch_memory_notes(...) -> (notes, embedding_tokens)                         <- was: -> notes
    wraps the question-embedding call in span("memory_retrieve")
  _maybe_extract_memories(...) -> (LLMUsage, new_fact_embedding_tokens)         <- was: -> None
    wraps the extractor call in span("memory_extract")
  _extra_cost_fields(settings, ctx_usage, memory_query_tokens, extraction_usage, new_fact_tokens) -> dict
    pure function: turns the three extra usages/token-counts into CostBreakdown-shaped fields + their sum
  _apply_extra_costs(cost: CostBreakdown, extra: dict)        -- non-streaming /ask
  _apply_extra_costs_dict(cost: dict, extra: dict)            -- streaming /ask/stream "done" payload

  /ask:         answer_question -> _maybe_extract_memories -> _apply_extra_costs -> record_tenant_spend (ONCE)
  /ask/stream:  answer_question_stream -> (on "done") -> _maybe_extract_memories -> _apply_extra_costs_dict
                -> record_tenant_spend (ONCE)

app/models/ask.py
  CostBreakdown: + contextualizer_{prompt,completion}_tokens, contextualizer_cost_usd
                 + memory_extraction_{prompt,completion}_tokens, memory_extraction_cost_usd
                 + memory_embedding_tokens, memory_embedding_cost_usd
```

## 5. Files
NEW: scripts/verify_day16.py, docs/day16.md, tests/test_cost_accounting.py, tests/test_memory_observability.py
CHANGED: app/models/ask.py (6 new `CostBreakdown` fields), app/services/contextualizer.py (`contextualize()`
returns usage), app/services/memory_extractor.py (`extract()` returns usage), app/routers/ask.py (`_embed_text`,
`_contextualize`, `_fetch_memory_notes`, `_maybe_extract_memories` all return their cost data; new
`_extra_cost_fields`/`_apply_extra_costs`/`_apply_extra_costs_dict`; three new `tracing.span()`s;
`record_tenant_spend` reordered to after extraction in both `/ask` and `/ask/stream`), app/routers/memories.py
(unpacks `_embed_text`'s new tuple return; documents why manual-add embedding cost stays unbilled),
tests/fakes.py (`ScriptedContextualizerFactory`/`ScriptedMemoryExtractorFactory` return usage tuples),
tests/test_contextualizer.py + tests/test_memory_extractor.py (assert the returned usage), app/main.py (version
bump), README.md (Day 16 row, test count, script name, cost breakdown mention)

## 6-7. Code notes
- `_extra_cost_fields` is a pure function (settings + four plain values in, a dict out) specifically so it has no
  idea whether it's being merged into a Pydantic model or a dict -- that decision lives entirely in
  `_apply_extra_costs` vs. `_apply_extra_costs_dict`, keeping the one piece of real arithmetic in exactly one
  place.
- Every one of the three services' SKIP paths was audited to return a real zero, not an omitted value:
  `_contextualize` returns `LLMUsage()` on no-conversation-id, no-API-key, AND on an `LLMError` (where the actual
  usage is unknowable anyway, since the exception path in `JsonLLM.complete_json` never returns one) --
  `LLMUsage()`'s `0, 0` defaults make "unknown/none" and "actually zero" indistinguishable, which is the correct
  choice here: a request that's never going to be billed shouldn't need to distinguish the two.
- `_fetch_memory_notes`'s `span("memory_retrieve")` wraps only the embedding call, not the whole function -- the
  `memory_repo.count() == 0` short-circuit and the `search_memories` ranking itself are cheap, synchronous, and
  not where the interesting latency or cost lives, so they're deliberately left out of the span rather than
  padding it with near-zero-duration work.
- `_contextualize`'s `span("contextualize")` wraps the OUTER call to `contextualizer.contextualize(...)`, which
  means the span fires even on a conversation's FIRST turn (where the contextualizer itself skips the LLM call
  internally and the span shows ~0ms duration) -- this is intentional, not a gap: the span records that
  contextualization was considered for this request, and the cost fields (always zero on that path) are what
  actually tell you whether an LLM call happened, not the span's mere presence. Confirmed directly by
  `test_memory_observability.py`'s `test_contextualize_span_still_recorded_but_free_on_a_conversations_first_turn`.
- `routers/memories.py`'s manually-added-fact embedding is deliberately NOT billed to the tenant's daily budget --
  it has no `/ask` request to attach a cost breakdown to, and it's an explicit, low-frequency, user-initiated
  write, not an automatic side effect of answering a question. Documented in the module docstring so it doesn't
  read as an oversight later.

## 8. Run
```
python scripts/verify_day16.py            # pytest + a real cost-accounted multi-turn exchange
python scripts/verify_day16.py --offline  # pytest only

# Manual walkthrough:
curl -X POST localhost:8000/memories -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"content": "Always wants answers in metric units."}'
curl -X POST localhost:8000/conversations -H "Authorization: Bearer <key>"
# -> {"conversation_id": "..."}
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "What is the warranty?", "document_id": "<id>", "conversation_id": "<id>"}'
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "What about shipping instead?", "document_id": "<id>", "conversation_id": "<id>"}'
# -> the second call's "cost" now has nonzero contextualizer_cost_usd, and total_cost_usd includes it
```

## 9. Testing
332 tests total (17 new since Day 15's 315): cost accounting (11, in tests/test_cost_accounting.py -- contextualizer
cost zero on a first turn / nonzero and exactly matching the expected arithmetic on a follow-up / zero with no
conversation_id at all; memory extraction cost nonzero after a real answer / zero on a refusal / zero when the
extractor fails; memory embedding cost zero with no stored memories / nonzero once memories exist / also nonzero
purely from a newly extracted fact's embedding with no prior memories; `total_cost_usd` equals the sum of every
component with every extra actually firing; the per-tenant daily budget reflects the full total) and observability
(6, in tests/test_memory_observability.py -- the `contextualize` span fires on a follow-up and is absent without a
`conversation_id` at all, present-but-free on a conversation's first turn; `memory_retrieve` fires only when
memories exist; `memory_extract` fires after a real answer), plus the two existing unit-test files
(test_contextualizer.py, test_memory_extractor.py) updated to assert the newly-returned usage.

## 10. Failure cases
No new failure modes were introduced -- this is purely an accounting change on top of behavior that already
existed and was already fail-open (Days 13-15). The one thing worth calling out: on an `LLMError` inside
`_contextualize` or `_maybe_extract_memories`, the real token cost of that FAILED call is unknowable (the
exception path never returns a usage) and is therefore reported as zero, not estimated -- a provider that fails
after partially billing a request (rare, but not impossible with some providers) could in principle under-report
by a small amount here; this is an accepted, documented approximation rather than a gap, since there is no
reliable way to recover a real number from an exception.

## 11. Security
No new surface here -- cost fields are purely numeric accounting data with no new data flow, no new storage, and
no new cross-tenant exposure. `record_tenant_spend` is still only ever called with the authenticated caller's own
`tenant_id`, exactly as it was through Day 15.

## 12. Observability
The gap this section has flagged as open in `docs/day13.md`, `docs/day14.md`, and `docs/day15.md` is now closed:
`contextualize`, `memory_retrieve`, and `memory_extract` all appear in the same per-request audit log line
(`app/middleware/audit_log.py`'s consolidated `spans` list) alongside `retrieval`/`rerank`/`generation`, for the
first time giving a complete, single-line trace of everything a multi-turn, memory-aware `/ask` call actually did.

## 13. Cost
This IS the cost fix -- there is no longer a known, uncounted cost gap in this project's `/ask` pipeline.
`AskResponse.cost.total_cost_usd` and the Day 10 per-tenant daily budget both now reflect retrieval, reranking,
generation, contextualization, memory extraction, and memory embeddings: everything a single `/ask` or `/ask/
stream` call can spend.

## 14. Production improvements
Cache a question's embedding for memory retrieval the way `services/search.py` already caches query embeddings
for document search (`get_query_embedding`), since the identical question asked twice currently re-embeds for
memory retrieval both times (flagged in `docs/day15.md`, still open); batch-embed multiple newly extracted facts
in one API call instead of one `_embed_text` call per fact; surface the new cost breakdown fields in
`scripts/run_eval.py`'s reports and `/observability/summary`, which currently only roll up the original three
cost categories; consider whether manually-added memories (routers/memories.py) should get a lightweight,
separate "management API cost" ledger distinct from the per-`/ask` budget, now that the precedent for tracking
embedding cost exists.

## 15. Interview Q&A
- *Why did this wait until Day 16 instead of being fixed as each cost was introduced?* Each of the three extra
  calls arrived on a different day with a different, not-yet-settled shape; fixing the cost model incrementally,
  once per day, would have meant three separate partial redesigns instead of one coherent one.
- *How do you guarantee a skipped call reports exactly zero cost, not just "unset"?* Every skip path explicitly
  returns `LLMUsage()` / `0` rather than omitting a value -- "no call was made" and "a call was made and cost
  nothing" are different claims, and the test suite asserts the zero case as deliberately as the nonzero one.
- *Why does the contextualizer's tracing span still fire on a conversation's first turn, even though there's no
  LLM call?* The span records that contextualization was considered for this request; the cost fields (correctly
  zero on that path) are what tell you whether work actually happened. Presence of a span and presence of cost
  are two different questions.
- *Why is a pure function (`_extra_cost_fields`) used instead of mutating cost objects directly in each of the
  three helper functions?* It keeps the one piece of real arithmetic (three costs + a new grand total) in exactly
  one place, usable identically by both the Pydantic-model (non-streaming) and dict (streaming) response shapes.
- *Why isn't a manually-added memory's embedding cost billed to the tenant budget?* It isn't a side effect of
  answering a question -- there's no `/ask` request to attach the cost to, and it's a deliberate, infrequent,
  user-initiated write, not automatic spend a budget ceiling needs to guard against.

## 16. Day-end checklist
- [ ] `python scripts/verify_day16.py` ends with ALL CHECKS PASSED
- [ ] you can explain why `AskResponse.cost.total_cost_usd` now includes three fields it didn't before
- [ ] you can explain why every skip path reports an explicit zero, not an omitted value
- [ ] you can explain why the contextualizer's span can fire with zero cost behind it
- [ ] you can explain why `record_tenant_spend` is now called once, after extraction, not before it

## 17. Learned
Closing a cost/observability gap that was deliberately tracked (not hidden) across three consecutive days, in one
coherent pass once all three shapes existed, rather than three partial fixes; threading real usage data through
return values instead of a side channel, consistent with every other cost-bearing call already in this codebase;
and the discipline of treating "zero because skipped" and "zero because free" as the same correct value, while
still testing both directions explicitly so a regression in either one would be caught.

## 18. Next: Day 17
With both the pipeline's core cost accounting and its memory layers now complete, a natural next step is applying
the same completeness instinct to surfacing this data: extending `/observability/summary` and
`scripts/run_eval.py`'s reports with the new cost categories, and/or caching the repeated question-embedding cost
flagged in Section 14 -- or, stepping back to the project's remaining larger gaps (`docs/day15.md`'s tracker),
containerization (Docker/Compose) as the next piece of "production-ready" that hasn't been addressed yet.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection mitigation + test suite, shared client lifecycle, golden datasets, retrieval
metrics, citation-based answer correctness, LLM-as-judge, eval CLI + reports, data-driven tuning, offline
retrieval regression testing, rate limiting, audit logging, PII-aware ingestion, streaming responses (SSE),
unified request tracing, Prometheus metrics, live observability dashboard, API key auth, multi-tenant data
isolation, per-tenant rate limits + persisted daily cost budgets, key management CLI, two-tier CI/CD with a
tested cost ceiling, local pre-push test hook, pluggable ANN vector indexing (HNSW) with per-tenant caching and a
realistic benchmark, short-term multi-turn conversation memory with query contextualization, long-term
cross-conversation memory (tenant-scoped fact extraction, storage, and generation-time use), vector-indexed
memory retrieval (relevance-ranked facts instead of flat injection), full cost + trace accounting for
contextualization, memory extraction, and memory embeddings.
NOT YET: question-embedding cache for memory retrieval, batch-embedding of multiple newly extracted facts, the
new cost categories surfaced in /observability/summary and eval reports, real OTel export, real Prometheus/
Grafana deployment, PII redaction mode, RAGAS/DeepEval, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~99%. Ingestion, indexing, retrieval (exact and approximate), grounded generation, evaluation,
tuning, security baseline, observability, multi-tenancy, CI/CD, vector-store scaling, short-term and long-term
agent memory, relevance-ranked memory retrieval, and complete cost/trace accounting across the entire /ask
pipeline are done; containerization and a short list of smaller polish items remain.
