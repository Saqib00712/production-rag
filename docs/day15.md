# Day 15: Vector-Indexed Memory Retrieval

## 1. Objective
Replace Day 14's "inject every stored fact into every prompt" behavior with relevance-based retrieval: each
memory fact is embedded when it's stored, each question is embedded when it's asked, and only the
`memory_top_k` facts most similar (cosine) to the question are fed to the generator -- the same brute-force
vector math (`services/vector_math.cosine_top_k`) Day 3's document search and Day 12's small-scale path already
use, applied to a tenant's own fact store instead of document chunks.

## 2. Why this matters
Day 14 worked, but it doesn't scale in the one dimension that matters: as a tenant accumulates more remembered
facts, EVERY one of them gets dumped into EVERY generation prompt, whether or not it has anything to do with the
question being asked. A tenant who has "prefers metric units," "works in procurement," "is allergic to
shellfish," and fifteen other unrelated facts pays the token cost of all of them on a question about none of
them -- and worse, a generator sifting through fifteen irrelevant notes to find the one relevant one is exactly
the kind of prompt dilution that makes long context windows less reliable in practice, not more. This was
already flagged as the known next step at the end of `docs/day14.md`'s Section 14 and Section 18: today builds
it.

## 3. Theory
**Why embedding beats "just send everything" at this scale, and why it's still NOT the Day 12 ANN story.**
Day 12 introduced HNSW because a tenant's document chunks can run into the tens of thousands, where even a
single brute-force matrix-vector product starts to cost real latency. A tenant's memory store is capped at
`memory_max_facts` (tens of facts, by design -- see `docs/day14.md` Section 3's FIFO cap reasoning) specifically
so it never needs that: `cosine_top_k` over a few dozen vectors is effectively free. Day 15 is about picking the
RIGHT subset of a small store, not about searching a LARGE one fast -- a genuinely different problem from Day 12's,
even though both end up calling the same `cosine_top_k` function.

**Why embed at write time, not at read time.** A fact's embedding never changes once the fact's text is fixed,
so computing it once when the fact is first stored (or restated) and reusing it on every future question is the
obvious trade: one embedding call per fact, forever, instead of one per fact per question. This mirrors the
project's embedding-cache instinct from Day 2 (`embedding_batch_size`/cache) applied to a new kind of text.

**What "relevant" means here, and why memory notes still can't be cited.** Relevance is purely semantic
similarity between the question and a fact's text -- it says nothing about whether the fact is TRUE or where it
came from, and `services/generator.py`'s `format_memory_notes()` still renders selected facts in the same
clearly-separate, non-citable block introduced on Day 14 (see `docs/day14.md` Section 3). Vector-selecting WHICH
facts to show the generator doesn't change WHAT the generator is allowed to do with them once shown.

**Why a fact with no embedding yet doesn't just vanish.** A memory fact can lack an embedding for two reasons:
it predates Day 15 (created before this column existed), or the embedding call failed when it was first stored
(the embedder was fail-open even then -- see `docs/day14.md`'s `_maybe_extract_memories`). Either way, the fact
is still real and still worth using. `MemoryRepository.search_memories()` only ranks EMBEDDED facts and excludes
the rest from its own result rather than scoring them arbitrarily; `routers/ask.py`'s `_fetch_memory_notes`
decides what to do when that leaves too little to work with (Section 6-7) -- falling back to Day 14's flat,
most-recent list rather than silently losing those facts from the prompt altogether.

**Cost discipline, now with one more skip.** Day 13 skips the contextualizer call on a conversation's first
turn; Day 14 skips extraction on a refusal. Day 15 adds: skip embedding the question AT ALL when the tenant has
zero stored memories (`MemoryRepository.count()` checked first, before ever calling the embedder) -- there is
nothing to rank a question against, so paying for that embedding would be pure waste, the same "never pay for
work a cheap check proves unnecessary" instinct this project has applied at every single memory/context layer
added since Day 4.

**Why embedding failure falls back to Day 14's behavior, never to silence.** Fail-open has one new twist here:
unlike the contextualizer or extractor, where failing open means "do less" (skip the rewrite, skip storing a new
fact), failing open for RETRIEVAL could mean "show nothing" -- which would be strictly worse than Day 14's
behavior, not a safe degradation. So the fallback target is deliberately specific: `_fetch_memory_notes` never
degrades to an empty list when the tenant actually has memories; it degrades exactly to what Day 14 already did
(the flat, most-recent-`memory_max_facts` list), a behavior this project already shipped and already trusts.

## 4. Architecture
```
app/repositories/memory_repository.py
  memories(memory_id, tenant_id, content, created_at, embedding BLOB NULL)   <- self-healing ALTER TABLE
  MemoryRepository:
    add_memory(..., embedding=None)   -- stores it; backfills onto an existing row on a dedup match if that
                                          row had none yet (never overwrites an existing embedding)
    count(tenant_id)                  -- lets callers skip embedding a question when the store is empty
    search_memories(tenant_id, query_embedding, top_k)
      -> loads this tenant's EMBEDDED facts only, ranks via vector_math.cosine_top_k, returns top_k

app/routers/ask.py
  _embed_text(text, embedder_factory)         -- shared fail-open helper: EmbeddingConfigError/EmbeddingError -> None
  _fetch_memory_notes(question, tenant_id, memory_repo, embedder_factory, settings)
    count == 0?            -> [] (nothing to rank against, nothing to embed)
    embed(question) fails? -> Day 14 flat list (list_memories, limit=memory_max_facts)
    search_memories empty? -> Day 14 flat list (no embedded facts to rank yet)
    else                   -> the memory_top_k most relevant facts' content
  _maybe_extract_memories(...)
    each newly extracted fact -> _embed_text(fact, embedder_factory) -> add_memory(..., embedding=...)

app/routers/memories.py
  POST /memories -> _embed_text(content, embedder_factory) -> add_memory(..., embedding=...)
```

## 5. Files
NEW: scripts/verify_day15.py, docs/day15.md
CHANGED: app/repositories/memory_repository.py (`embedding` column + self-healing migration, `count`,
`search_memories`, embedding-aware/backfilling `add_memory`), app/routers/ask.py (`_embed_text` helper,
relevance-based `_fetch_memory_notes`, embedding-on-extraction in `_maybe_extract_memories`),
app/routers/memories.py (embeds a manually added fact via the same `_embed_text` helper), app/config.py
(`memory_top_k`), app/main.py (version bump), tests/test_memory_repository.py (9 new: ranking, exclusion of
unembedded facts, empty-store result, top_k, per-tenant scoping, embedding backfill, never-overwrite, `count`,
old-schema migration), tests/test_memories_endpoint.py (3 new: top_k cap over HTTP, fallback to the flat list
when embedding the question fails, a memory created with no API key still stores and lists correctly), README.md
(Day 15 row, test count, script name, `/memories` description)

## 6-7. Code notes
- `search_memories()` deliberately returns an EMPTY result (not a partial, unranked one) when no fact has an
  embedding yet, rather than mixing ranked and unranked facts in one list with made-up scores for the latter --
  `_fetch_memory_notes` treats "nothing usable from search" as a single, clean trigger for its Day 14 fallback,
  not a partial result it has to reason about further.
- `add_memory()`'s dedup path backfills an embedding onto an existing row ONLY when that row's embedding is
  currently NULL (`existing_embedding is None and blob is not None`) -- it never overwrites a real, already-
  stored embedding with a new one from a restatement, since the original text (and therefore its embedding) is,
  by the dedup check itself, identical anyway.
- `_embed_text()` lives in `routers/ask.py`, not `memory_repository.py` or a new service module, because it's a
  thin, fail-open orchestration of an EXISTING protocol (`Embedder`, from Day 2) -- there's no new embedding
  logic here, just the same fail-open wrapping pattern every other LLM-backed call in this router already uses
  (`_contextualize`, `_maybe_extract_memories`). `routers/memories.py` imports and reuses it directly rather than
  duplicating the same try/except shape a third time.
- `MemoryRepository` reuses `ChunkRepository`'s `pack_vector`/`unpack_vector` (plain `array("f")` byte packing)
  instead of JSON-encoding floats -- same format, same space/precision trade-off Day 2 already made for document
  chunk vectors, imported rather than re-implemented.
- The embedding migration follows `ChunkRepository._migrate_tenant_column`'s exact shape: check
  `PRAGMA table_info`, `ALTER TABLE ... ADD COLUMN` only if missing, so a Day 14-created `memories.db` upgrades
  in place the first time this code runs against it -- no separate migration script, no manual intervention.

## 8. Run
```
python scripts/verify_day15.py            # pytest + a real relevance-ranked memory retrieval
python scripts/verify_day15.py --offline  # pytest only

# Manual walkthrough:
curl -X POST localhost:8000/memories -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"content": "Always wants answers in metric units."}'
curl -X POST localhost:8000/memories -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"content": "Is allergic to shellfish."}'
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "Can you give the refund window in weeks instead of days?", "document_id": "<id>"}'
# -> only the metric-units fact is relevant and gets used; the shellfish fact stays stored but unused here
```

## 9. Testing
315 tests total (12 new since Day 14's 303): the memory repository (9: cosine ranking order, facts with no
embedding excluded from results, an empty result when nothing is embedded yet, `top_k` is respected, per-tenant
scoping of the search itself, an embedding backfilled onto a previously-unembedded restated fact, an existing
embedding is never overwritten by a later restatement, `count()` tracks inserts/deletes, and a pre-Day-15 schema
missing the `embedding` column migrates in place without error) and HTTP-level tests (3: the facts sent to
generation are capped at `memory_top_k` even with many more stored, a failing embedder falls back to the exact
Day 14 flat list rather than sending nothing, and a memory created with no `OPENAI_API_KEY` configured still
stores and lists correctly with no embedding).

## 10. Failure cases
No `OPENAI_API_KEY` configured, or a live embedding failure mid-request: `_fetch_memory_notes` falls back to
Day 14's flat, most-recent-facts list rather than sending no memory context at all or failing the request (same
fail-open principle as every other optional LLM/embedding call added since Day 10). A tenant with memories but
none of them embedded yet (e.g. every one predates Day 15 and hasn't been restated): `search_memories` returns
nothing usable, and the same flat-list fallback applies -- functionally identical to running plain Day 14 until
those facts get re-extracted or manually re-added. A restated fact embeds to a near- but not exactly-identical
vector each time (if the embedding provider isn't perfectly deterministic): irrelevant here, since dedup matches
on the fact's TEXT, not its embedding, and an existing embedding is never replaced by a newer one anyway (Section
6-7) -- there's no drift to accumulate.

## 11. Security
No new surface here beyond what Day 14 already locked down: `search_memories` is always scoped to the tenant_id
already authenticated by `enforce_tenant_limits`/`get_current_tenant`, so one tenant's question can never rank or
retrieve a different tenant's facts -- confirmed by `test_search_memories_is_scoped_per_tenant`. Embeddings
themselves carry no more information than the fact's own text already does (an embedding is a function of the
text, not an independent secret), so storing them introduces no new sensitive data, just a derived representation
of data that was already being stored.

## 12. Observability
Like Day 13's contextualization and Day 14's extraction, the question-embedding call inside `_fetch_memory_notes`
doesn't yet get its own `tracing.span()` -- a third small gap in the same shape, now worth fixing as a batch
rather than one at a time (see Section 14).

## 13. Cost
Zero extra cost when a tenant has no memories yet (`count()` check, Section 3) and zero extra cost for every
fact whose embedding was already computed when it was stored (the common case after the first time a fact is
seen). Every question asked by a tenant WITH at least one memory costs one extra, cheap embedding call (reusing
`openai_embedding_model`, the same model and price already configured for document search) -- on top of
whatever Day 13's contextualizer and Day 14's extractor already cost on top of retrieval + generation. This is
the THIRD uncounted LLM/embedding call now (alongside contextualization and extraction) not yet reflected in
`AskResponse.cost` or the per-tenant daily budget -- see Section 14, which batches all three into one proposed
fix rather than letting the list of "known, tracked, uncounted costs" keep growing unaddressed.

## 14. Production improvements
Fold ALL THREE now-uncounted extra calls (Day 13's contextualizer, Day 14's extractor, Day 15's question/fact
embeddings) into `AskResponse.cost` and the per-tenant daily budget in one pass, rather than one-at-a-time --
the gap has compounded for three days straight and deserves a single, complete fix; add the matching
`tracing.span()`s for all three at the same time, for the same reason; cache a question's embedding the way
`services/search.py` already caches query embeddings for document search (`get_query_embedding`/hash lookup),
since the identical question asked twice currently re-embeds for memory retrieval both times; batch-embed
multiple newly extracted facts in one API call when more than one comes back from a single extraction, instead
of one `_embed_text` call per fact; a background job to opportunistically re-embed older, unembedded facts
instead of waiting for them to be restated.

## 15. Interview Q&A
- *How is this different from Day 12's HNSW work?* Day 12 is about searching a LARGE collection (document
  chunks, tens of thousands) fast. Day 15 is about selecting the RIGHT subset of a SMALL, capped collection
  (a tenant's memory facts) -- brute-force cosine is not just acceptable here, it's the correct choice, since an
  ANN index would add real complexity for a corpus size that never needs it.
- *Why embed facts when they're stored instead of when they're used?* A fact's embedding is purely a function of
  its (unchanging) text, so computing it once and reusing it on every future question is strictly cheaper than
  re-embedding the same fact on every question that might use it.
- *What happens to a fact that was never successfully embedded?* It's excluded from the ranked search results,
  not scored arbitrarily -- and the caller (`_fetch_memory_notes`) falls back to Day 14's flat list specifically
  so that fact is still usable, just without relevance filtering, rather than being silently dropped from every
  future prompt.
- *Why does a failed embedding fall back to the FLAT LIST, not an empty list?* Because an empty list would be a
  worse outcome than what the system already shipped and trusted on Day 14 -- failing open here means falling
  back to a known-good prior behavior, not degrading all the way to nothing.
- *How do you keep a restated fact from corrupting its own embedding?* Dedup matches on identical text, and an
  existing embedding is never overwritten by a new one from that same restatement -- there's nothing to
  reconcile, since the text (and therefore the correct embedding for it) hasn't changed.

## 16. Day-end checklist
- [ ] `python scripts/verify_day15.py` ends with ALL CHECKS PASSED
- [ ] you can explain why this isn't "Day 12 again" despite both calling `cosine_top_k`
- [ ] you can explain why facts are embedded at write time, not read time
- [ ] you can explain why a failed question-embedding falls back to Day 14's flat list, not an empty list
- [ ] you can explain why memory notes still can't be cited, even after being selected by relevance

## 17. Learned
Recognizing when a familiar tool (brute-force cosine ranking, already built for Day 3/12) solves a DIFFERENT
problem at a different scale rather than reapplying the same solution's reasoning uninspected; designing a
fallback that degrades to a previously-shipped, trusted behavior instead of to silence or emptiness; and once
again tracking a compounding, known cost/observability gap explicitly across three consecutive days (Day 13,
14, 15) rather than letting each day's "not yet counted" note quietly become four, then five.

## 18. Next: Day 16
With both memory layers now relevance-aware, the project's three real remaining uncounted-cost gaps
(contextualization, extraction, memory embedding) are an obvious, high-value next target: folding all of them
into `AskResponse.cost` and the Day 10 per-tenant daily budget in a single pass, plus the matching
`tracing.span()`s, so the full cost and trace of a single `/ask` call -- not just retrieval/rerank/generation --
is finally visible end to end.

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
memory retrieval (relevance-ranked facts instead of flat injection).
NOT YET: contextualizer-, extractor-, and memory-embedding cost accounted in the budget ledger, matching trace
spans for all three, query-embedding cache for memory retrieval, real OTel export, real Prometheus/Grafana
deployment, PII redaction mode, RAGAS/DeepEval, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~98%. Ingestion, indexing, retrieval (exact and approximate), grounded generation, evaluation,
tuning, security baseline, observability, multi-tenancy, CI/CD, vector-store scaling, short-term agent memory,
long-term cross-conversation memory, and relevance-ranked memory retrieval are done; cost/trace completeness for
the three newest LLM/embedding call sites and containerization remain as the major pieces.
