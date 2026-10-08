# Day 19: A Question-Embedding Cache for Memory Retrieval

## 1. Objective
Close the last flagged, non-infrastructure gap: Day 15 introduced embedding the question for memory retrieval,
and every day since (`docs/day16.md`, `docs/day17.md`, `docs/day18.md`) flagged the same open item -- the
identical question asked twice re-embeds from scratch both times, paying for an OpenAI call that Day 3's document
search solved for itself back on Day 3. Today reuses that exact mechanism instead of building a new one.

## 2. Why this matters
`services/search.py` has cached query embeddings since Day 3 (`ChunkRepository.get_query_embedding`/
`save_query_embedding`, keyed by `sha256(text)+model`) -- the same question asked twice for document retrieval
has always been free the second time. Memory retrieval (Day 15) never got this: `_embed_text` in `routers/ask.py`
called the embedder directly every time, with no cache check at all. A tenant who asks the same question
repeatedly, or who simply asks ANY question while they have stored memories, was paying for a real embedding
call every single time purely to look up facts -- a real, repeating cost with an off-the-shelf fix already
sitting one file away.

## 3. Theory
**Why this reuses `services/search.py`'s cache instead of building a new one.** An embedding is a pure function
of (text, model) -- `sha256(text)+model` already uniquely identifies the vector regardless of WHY it's being
computed. Memory retrieval embedding a question and document retrieval embedding a question are the same
operation on the same input; there is no reason for them to be two separate caches that could each hold a stale
or missing entry for the same text. Giving `_embed_text` the SAME `ChunkRepository` the router already depends on
(both `/ask` and `/ask/stream` already inject it for other reasons) means one cache, one eviction policy
(`query_cache_max_rows`, Day 3), and one place that already has its own safety analysis (the module docstring on
why sharing this cache across tenants is safe -- see Section 11) instead of a second copy of that reasoning.

**Why this also makes retrieval's OWN embedding free in the common case, as a side effect, not a second feature.**
`routers/ask.py`'s handler order is: contextualize -> fetch memory notes -> answer_question (which does
retrieval). When there's no conversation rewrite, `search_query` equals the raw question, so memory lookup and
retrieval embed the IDENTICAL text, one call apart, within the SAME request. Before today, both paid separately
(memory lookup had no cache at all; retrieval's own cache had nothing cached yet for that exact request). Now
memory lookup's call -- which runs first -- pays once and writes the cache; retrieval's call, moments later,
finds it already there. This wasn't separately engineered; it falls directly out of memory lookup finally using
the cache that was already sitting right next to it.

**Why fact embedding (`_maybe_extract_memories`) deliberately does NOT use this cache.** A newly extracted fact
is new text pulled out of a specific answer -- there's no reason to expect the exact same fact string to recur,
so caching it would only grow the `query_embeddings` table with rows that are essentially never read back,
competing for space (bounded by `query_cache_max_rows`) with question embeddings that genuinely do repeat. The
cache parameter on `_embed_text` is opt-in per call site specifically so this distinction is explicit, not
accidental.

**Why the cache check happens before the embedder call, not after.** The whole point is to avoid the OpenAI call
entirely on a hit, not merely to avoid re-saving a vector that's about to be computed anyway. `_embed_text` hashes
the text, checks the cache, and only reaches the `embedder_factory()`/`.embed()` call on a miss -- a cache hit
returns in one SQLite read, with the token count (and therefore cost) correctly reported as zero, the same
"zero should mean zero, not estimated" convention every skip path in this project already follows since Day 16.

## 4. Architecture
```
app/routers/ask.py
  _embed_text(text, embedder_factory, *, cache_repo=None, cache_model=None, cache_max_rows=None)
    cache_repo given  -> check ChunkRepository.get_query_embedding(sha256(text), model) FIRST
      hit  -> return (vector, 0)                         <- no OpenAI call at all
      miss -> call the embedder, then ChunkRepository.save_query_embedding(...), return (vector, tokens)
    cache_repo=None   -> unchanged Day 16 behavior (always calls the embedder) -- used for NEW FACTS only

  _fetch_memory_notes(question, ..., cache_repo: ChunkRepository)
    now passes cache_repo=cache_repo, cache_model=settings.openai_embedding_model,
    cache_max_rows=settings.query_cache_max_rows into _embed_text for the QUESTION embedding

  _maybe_extract_memories(...)                             <- UNCHANGED: still calls _embed_text with no cache_repo

  ask() / ask_stream(): pass the existing `repo: ChunkRepository` dependency into _fetch_memory_notes
    (same ChunkRepository instance services/search.py's SearchService already reads/writes
    query_embeddings through -- ONE shared cache, ONE table, ONE eviction policy)
```

## 5. Files
CHANGED: app/routers/ask.py (`_embed_text` gains optional cache parameters; `_fetch_memory_notes` takes and uses a
`cache_repo`; both call sites in `ask()`/`ask_stream()` pass their existing `repo` dependency through; module
docstring), tests/test_cost_accounting.py (2 new tests), scripts/verify_day19.py (NEW), docs/day19.md (NEW),
app/main.py (version bump), README.md (Day 19 row, test count, script name, walkthrough step)

## 6-7. Code notes
- `_embed_text`'s three new parameters (`cache_repo`, `cache_model`, `cache_max_rows`) are keyword-only and all
  default to `None`, so the one existing call site that must NOT cache (`_maybe_extract_memories`'s fact
  embedding) needed no change at all -- it simply never passes them, and the function's existing behavior for
  that call site is untouched byte-for-byte.
- The hash is computed once (`qhash = hashlib.sha256(...)  if cache_repo is not None else None`) and reused for
  both the lookup and the save, rather than re-hashing after a miss -- a minor efficiency, but also a correctness
  guard: the save can never target a different key than the lookup just checked.
- Both new tests in `tests/test_cost_accounting.py` use `HashEmbedder.calls` (a fake embedder that records every
  text it was actually asked to embed) to assert the real thing that matters -- not just that the reported cost
  is zero, but that the embedder was never even called a second time for the same text.

## 8. Run
```
python scripts/verify_day19.py            # pytest + a real two-calls-same-question check
python scripts/verify_day19.py --offline  # pytest only

# Manual walkthrough:
curl -X POST localhost:8000/memories -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"content": "Always wants answers in metric units."}'
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "What is the warranty?", "document_id": "<id>"}'
# -> cost.memory_embedding_tokens > 0 (first time), cost.embedding_tokens == 0 (retrieval got it free already)
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "What is the warranty?", "document_id": "<id>"}'
# -> cost.memory_embedding_tokens == 0 too now -- the exact same question, a pure cache hit
```

## 9. Testing
338 tests total (2 new since Day 18's 336, both in tests/test_cost_accounting.py):
`test_memory_embedding_cache_avoids_a_second_real_call_for_a_repeated_question` asks the identical question twice
and asserts the first call's `memory_embedding_tokens > 0`, the second's `== 0`, AND that the literal question
text was only ever sent to the fake embedder once across both requests; `test_memory_embedding_cache_also_saves_
the_retrieval_embedding_in_the_same_request` asks it ONCE and asserts `memory_embedding_tokens > 0` while
`embedding_tokens` (retrieval's own cost) is ALREADY `0` on that very first call -- proving the cross-feature,
same-request sharing described in Section 3, not just the across-request repeat case.

## 10. Failure cases
No new failure modes. A cache miss falls through to exactly the same embedder call and exactly the same
`EmbeddingConfigError`/`EmbeddingError` handling `_embed_text` already had -- the cache check is a pure read that
can only ever short-circuit toward LESS work, never toward a new way to fail. A cache WRITE failure isn't
specially handled either, but that's consistent with the existing cache's own behavior (Day 3): `_embed_text`
still returns the freshly computed vector and real token count on a miss regardless of whether the subsequent
`save_query_embedding` call succeeds -- a failed write only costs a future cache opportunity, never the current
request.

## 11. Security
No new surface -- this reuses `services/search.py`'s existing cache exactly as designed, including its existing
cross-tenant sharing rationale (an embedding vector reveals nothing about tenant identity or document content by
itself; see `app/repositories/chunk_repository.py`'s module docstring, unchanged today). No new table, no new
column, no new data retained that wasn't already being retained for document-search queries since Day 3.

## 12. Cost
This IS a cost reduction, measured directly by the new tests: the same question asked twice for memory lookup now
costs one real embedding call instead of two, and a single question asked once with existing memories present
(the common case, not an edge case) now costs one embedding call instead of two, because retrieval's own
embedding of that same text is already cached by the time it runs.

## 13. Production improvements
Fact embedding still re-embeds a brand-new fact every time one is extracted, which is correct (facts aren't
expected to repeat) but means `_maybe_extract_memories` has no analogous win available; publishing the Docker
image to a registry (GHCR) as part of the existing Day 11 CI pipeline remains open from `docs/day18.md`; splitting
`requirements.txt` into prod/dev files remains open from the same day.

## 14. Interview Q&A
- *Why extend an existing cache instead of adding a second one for memory retrieval?* An embedding is a pure
  function of (text, model) -- the same text always produces the same vector no matter which feature asked for
  it, so a second cache keyed the same way would just be a second place the same information could go stale or
  disagree with the first.
- *How does asking one question with no repeats still benefit from this change?* Memory lookup and document
  retrieval embed the identical raw question within the same request (when there's no conversation rewrite).
  Memory lookup runs first in `routers/ask.py`'s handler order, so by the time retrieval needs that same text, it's
  already cached -- a same-request win, not just an across-request one.
- *Why doesn't newly extracted fact text get cached the same way?* A fact pulled out of one specific answer isn't
  expected to recur verbatim, so caching it would spend cache capacity (`query_cache_max_rows` is a bounded table)
  on rows that are essentially write-only, crowding out question embeddings that actually do repeat.
- *What would you check first if this cache ever returned a WRONG vector for a given question?* Whether the
  `model` used to look it up matches the model used to store it -- the cache key is `(sha256(text), model)`
  specifically so a provider/model change can never silently serve a stale vector computed under a different
  model.

## 15. Day-end checklist
- [ ] `python scripts/verify_day19.py` ends with ALL CHECKS PASSED
- [ ] you can explain why this reuses Day 3's cache instead of building a new one
- [ ] you can explain why retrieval's embedding cost can be zero even on a conversation's very FIRST question
- [ ] you can explain why fact embedding was deliberately left uncached

## 16. Learned
The cheapest performance win is often "this already exists two files away, just use it here too" rather than new
infrastructure; a pure function of (text, model) is naturally cacheable regardless of which feature is asking for
it; and a fix that was flagged as open for four consecutive days (15 through 18) sometimes turns out to be a
small, contained change once you actually sit down and trace exactly where the redundant call happens.

## 17. Next: Day 20
With cost accounting, full observability, containerization, and the last flagged caching gap all now closed, the
project's core feature set and production-readiness checklist are essentially complete. A natural next step is
publishing the Docker image to a registry (GHCR) as part of the existing Day 11 CI pipeline -- turning Day 18's
local-only build into something a real deployment could pull -- or splitting `requirements.txt` into prod/dev
files, the smaller polish item flagged since Day 18.

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
contextualization/memory extraction/memory embeddings, that accounting fully surfaced in both the live
observability dashboard and eval reports, containerization (Dockerfile + docker-compose.yml with persistent
storage and a real healthcheck), and a shared question-embedding cache for memory retrieval (closing the last
flagged caching gap, with a same-request bonus for document retrieval too).
NOT YET: publishing the built Docker image to a registry, splitting prod/dev dependency files, real OTel export,
real Prometheus/Grafana deployment, PII redaction mode, RAGAS/DeepEval, LangGraph, OCR.
PROJECT STATUS: ~99%. Ingestion, indexing, retrieval (exact and approximate), grounded generation, evaluation,
tuning, security baseline, observability (including full per-stage cost visibility), multi-tenancy, CI/CD,
vector-store scaling, short-term and long-term agent memory, relevance-ranked memory retrieval with a shared
embedding cache, complete cost/trace accounting across the entire `/ask` pipeline, and containerization are done;
a short list of smaller polish/deployment items remain.
