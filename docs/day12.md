# Day 12: Vector Database at Scale - ANN Indexing with HNSW

## 1. Objective
A pluggable vector index abstraction with two backends: Day 3's exact brute-force search, and a new approximate
(HNSW) index via `hnswlib`, selectable per deployment via config. A per-tenant in-process index cache so the
(real) cost of building an index is paid once, not per query. A benchmark script that MEASURES whether the upgrade
is actually worth it at a given scale, rather than assuming "ANN is always better."

## 2. Why this matters
Day 3's brute-force cosine search is a full scan over every stored vector, every query - `O(n)` per query. At the
scale every document in this project has used so far (a resume, a handful of synthetic pages), that scan takes
microseconds and is nowhere near the bottleneck (the network round-trip to embed the query dwarfs it). That was the
right engineering call at the time: don't add complexity a system doesn't need yet. Past tens of thousands of
chunks, the arithmetic changes, and today builds the upgrade path - with the honesty to show, with real numbers,
exactly where that crossover point is.

## 3. Theory
**What HNSW actually trades away.** Hierarchical Navigable Small World graphs let search skip most of the index
instead of scanning everything, by building a layered graph where each query only "walks" a small neighborhood.
The cost: it's APPROXIMATE - it can occasionally miss the true nearest neighbor. `ef_search` (query-time effort)
and `ef_construction`/`M` (build-time effort/connectivity) are the knobs that trade recall against speed; higher
values of each mean more accuracy at the cost of more computation.

**Why the benchmark uses CLUSTERED synthetic vectors, not uniform random noise.** An earlier draft of
`bench_vector_index.py` used pure random Gaussian vectors and measured only ~35% recall - which would wrongly
suggest HNSW is unreliable. The actual problem was the synthetic data: in high dimensions, uniformly random vectors
are nearly equidistant from each other (the "curse of dimensionality"), which is close to the ADVERSARIAL worst
case for any ANN method - there's no structure to exploit. Real embeddings are nothing like this: semantically
related text clusters together in vector space, which is exactly the structure HNSW's graph is built to exploit.
Fixing the benchmark to generate vectors around cluster "topic" centers (simulating real semantic structure)
immediately produced 96-100% recall with a 400-600x speedup at 50,000 vectors - numbers that actually reflect what
happens with real OpenAI embeddings. This is kept in the code and in this doc deliberately: a benchmark built on
unrealistic synthetic data would have taught the wrong lesson, and catching that WAS the lesson.

**Why per-tenant indexes, built once and cached, not rebuilt per query.** Building an HNSW graph has a real cost
(see the benchmark's build-time numbers) - doing it on every query would spend more time building than any
approximate search could ever save. `TenantIndexCache` builds an index once per tenant and reuses it, invalidating
only when that tenant's documents actually change (after a successful, non-dry-run index call). This is also where
Day 10's isolation guarantee gets a second, structural reinforcement: because each tenant gets its OWN graph built
only from their own vectors, there is no shared index to accidentally leak across a forgotten filter - isolation
by construction, not by a check that could be skipped.

**Why single-document search still bypasses the cache.** Searching within one already-identified document is a
small, fast scan either way - adding a cache for that case would add complexity (one more thing that can go stale)
for a benchmark-invisible benefit. The cache specifically targets the "search across everything I have" case,
which is exactly the case that scales with total corpus size, not with one document's size.

**Why timing is printed, never asserted, in automated checks.** Wall-clock latency depends on the machine running
it - a CI runner under load, a different CPU, background processes. Asserting "HNSW must be under 2ms" in a test
is a recipe for flaky failures that have nothing to do with a real regression. RECALL, by contrast, is a property
of the algorithm and the data, not the hardware - `verify_day12.py` asserts on recall and only prints timing,
directly applying the same "don't gate CI on nondeterministic numbers" principle Day 11 established.

## 4. Architecture
```
app/services/vector_index.py
  VectorIndex (Protocol): build(ids, vectors), search(query, k), .size
  BruteForceIndex   - wraps Day 3's cosine_top_k; exact
  HnswIndex         - wraps hnswlib; approximate, tunable via ef_construction/M/ef_search

app/services/vector_index_cache.py
  TenantIndexCache: (tenant_id, model) -> built VectorIndex, in-process
    .get(tenant_id, model, loader)  - build-once, reuse until invalidated
    .invalidate(tenant_id)          - called by routers/documents.py after a real (non-dry-run) index

SearchService._vector():
  document_id given?      -> unchanged Day 3 path: load_vectors() + cosine_top_k() directly (already small/fast)
  document_id is None?    -> TenantIndexCache.get(tenant_id, model, loader).search(query, k)   <- Day 12 path

scripts/bench_vector_index.py:
  generate clustered synthetic vectors at a chosen scale -> build both indexes -> compare
  recall@k (asserted-worthy) and query latency / speedup (informational only)
```

## 5. Files
NEW: app/services/{vector_index,vector_index_cache}.py, scripts/{bench_vector_index,verify_day12}.py,
tests/{test_vector_index,test_vector_index_cache,test_vector_index_integration}.py, docs/day12.md
CHANGED: config.py (vector index backend + HNSW tuning settings), dependencies.py (`get_tenant_index_cache`),
services/search.py (SearchService accepts an optional index_cache, used for whole-tenant vector search),
routers/{search,ask,documents}.py (wire the cache through; invalidate after indexing), tests/conftest.py
(autouse reset for the new cache singleton), requirements.txt (`hnswlib`)

## 6-7. Code notes
- `vector_index.py`: `HnswIndex.search()` catches `RuntimeError` from hnswlib and fails soft to no results rather
  than crashing a search request - an edge case around `k` exceeding indexed elements that some hnswlib versions
  raise on despite an upstream clamp.
- `vector_index.py`: hnswlib's `'cosine'` space returns DISTANCE (`1 - similarity`); `HnswIndex.search()` converts
  back to similarity so callers see the same scale as `BruteForceIndex`/`cosine_top_k` - a backend swap should
  never change what the NUMBERS mean to code downstream.
- `vector_index_cache.py`: explicitly NOT lock-protected against two concurrent requests both rebuilding the same
  tenant's index - documented as an accepted trade-off (redundant work, never wrong results) rather than silently
  assumed safe.
- `tests/test_vector_index_integration.py`: parametrized over BOTH backends (`brute_force`, `hnsw`) for every
  test - this is what caught a real bug while building today (a dimension-check guard that treated "unknown
  dimension" as "wrong dimension" for `HnswIndex`, which doesn't expose the raw matrix `BruteForceIndex` does).
  Running the same test twice, once per backend, is what surfaced it immediately rather than shipping silently.

## 8. Run
```
pip install -r requirements.txt   # adds hnswlib
python scripts/verify_day12.py            # pytest + a benchmark + a real /ask call on the hnsw backend
python scripts/verify_day12.py --offline  # pytest + benchmark only
python scripts/bench_vector_index.py --n-chunks 50000 --n-queries 30   # see the real trade-off at scale
```
To actually switch your deployment to HNSW: set `VECTOR_INDEX_BACKEND=hnsw` in `.env`. Brute-force remains the
default - this project's own document counts don't need HNSW yet, which is itself the point being taught.

## 9. Testing
256 tests total (21 new): both index backends (10, including exact brute-force ranking and HNSW agreement with
brute-force at generous settings), the per-tenant cache (5, including independent tenants/models and invalidation),
and HTTP-level integration across BOTH backends (6, including the test that caught the dimension-check bug and a
direct proof that indexing a second document actually invalidates and refreshes the cached search index).
`verify_day12.py` asserts HNSW recall on a small clustered benchmark (not timing) and makes one real `/ask` call
configured for the `hnsw` backend end-to-end.

## 10. Failure cases
`hnswlib` not installed: only `HnswIndex` fails (on import, lazily, inside `build()`) - `brute_force` (the default)
is completely unaffected, so a missing optional dependency never breaks the whole app. A `k` larger than the
indexed corpus: clamped, not an error, on both backends. Two concurrent requests triggering a rebuild for the same
tenant: both succeed, redundant work only, never incorrect results (see the cache's documented concurrency note).

## 11. Security
Per-tenant index isolation is a direct extension of Day 10's isolation guarantee, not a new concern - each tenant's
HNSW graph is built from ONLY that tenant's vectors, so there is no cross-tenant index to leak from even in
principle, independent of any runtime filter.

## 12. Observability
Day 9's existing `embed_query`/`vector` span timings in `/ask`'s trace breakdown now reflect whichever backend is
configured - a deployment that switches to `hnsw` should see that span's duration drop at scale, visible in the
same unified trace log Day 9 built, with no new instrumentation needed.

## 13. Cost
The index itself costs no OpenAI calls either way (it operates on already-computed embeddings). The benchmark
script uses synthetic vectors specifically so comparing backends at any scale, including very large ones, costs
$0. The live verify step makes one real `/ask` call (a few cents at most) purely to confirm real end-to-end wiring.

## 14. Production improvements
A per-tenant `asyncio.Lock` around cache rebuilds to eliminate (not just bound) redundant concurrent rebuilds;
persisting built HNSW graphs to disk (`hnswlib` supports save/load) to avoid rebuilding from scratch on every
process restart for large tenants; auto-selecting the backend based on a tenant's chunk count instead of one
global setting; a real vector database (pgvector, Qdrant, etc.) once a single process's memory can no longer hold
every tenant's index comfortably - the `VectorIndex` Protocol makes that a new implementation, not a rewrite of
`SearchService`.

## 15. Interview Q&A
- *When would you reach for an ANN index over brute-force search?* Once corpus size makes a full scan the actual
  latency bottleneck - measured, not assumed; at a few thousand vectors brute-force is often still faster than the
  embedding API call that produced the query vector in the first place.
- *What does HNSW trade away for speed?* Recall - it's approximate, and can occasionally miss the true nearest
  neighbor; `ef_search`/`ef_construction`/`M` tune how much speed you trade for how much accuracy.
- *How do you benchmark a vector index fairly?* With data that has the same STRUCTURE as production data -
  uniformly random vectors in high dimensions are close to the adversarial worst case for ANN methods and will
  understate real-world recall; semantically clustered data is far more representative.
- *Why assert recall but not latency in an automated test?* Recall is a property of the algorithm and data;
  latency depends on the machine running the test - asserting on it invites flaky failures unrelated to real
  regressions.
- *How do you keep tenant data isolated in a vector index?* One index per tenant, built only from that tenant's
  vectors, rather than one shared index with a runtime filter that could be forgotten.

## 16. Day-end checklist
- [ ] `python scripts/verify_day12.py` ends with ALL CHECKS PASSED
- [ ] you've run `scripts/bench_vector_index.py` at a large scale and seen a real speedup with high recall
- [ ] you can explain why the benchmark uses clustered, not random, synthetic vectors
- [ ] you can explain what `ef_search`, `ef_construction`, and `M` each trade off
- [ ] you can explain why timing is printed but recall is asserted in `verify_day12.py`

## 17. Learned
ANN indexing with HNSW and its recall/speed trade-off, realistic benchmark data design (and catching an
unrealistic benchmark rather than trusting its numbers), per-tenant index caching with explicit invalidation,
and the broader principle of choosing data structures based on measured scale, not assumed scale.

## 18. Next: Day 13
Agent memory systems: this marks a shift in the curriculum's focus from "one great RAG answer" toward
multi-turn, stateful interactions - short-term conversation memory, long-term memory across sessions, and how
much of today's citation/grounding machinery carries over unchanged versus needs to be rethought when a question
depends on what was asked three turns ago.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection mitigation + test suite, shared client lifecycle, golden datasets, retrieval
metrics, citation-based answer correctness, LLM-as-judge, eval CLI + reports, data-driven tuning, offline
retrieval regression testing, rate limiting, audit logging, PII-aware ingestion, streaming responses (SSE),
unified request tracing, Prometheus metrics, live observability dashboard, API key auth, multi-tenant data
isolation, per-tenant rate limits + persisted daily cost budgets, key management CLI, two-tier CI/CD with a
tested cost ceiling, local pre-push test hook, pluggable ANN vector indexing (HNSW) with per-tenant caching and
a realistic benchmark.
NOT YET: agent memory (short/long-term), real OTel export, real Prometheus/Grafana deployment, PII redaction
mode, RAGAS/DeepEval, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~95%. Ingestion, indexing, retrieval (exact and approximate), grounded generation, evaluation,
tuning, security baseline, observability, multi-tenancy, CI/CD, and vector-store scaling are done; agent memory
and containerization remain as the major pieces.
