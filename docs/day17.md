# Day 17: Surfacing the Day 16 Costs in Observability + Eval Reports

## 1. Objective
Day 16 made every real cost in the `/ask` pipeline countable (`AskResponse.cost`, the per-tenant daily budget) but
left one consumer of that data behind: `/observability/summary` (and its dashboard) and `scripts/run_eval.py`'s
reports still only ever showed embedding/rerank/generation, because `routers/ask.py` never told the in-process
`cost_tracker` singleton about the three new categories. Today closes that, exactly as previewed at the end of
`docs/day16.md`.

## 2. Why this matters
`cost_tracker` (app/observability/metrics.py) is what `/observability/summary`'s dashboard reads to answer "what
is this process actually spending, and on what." Through Day 16, a tenant could run a dozen multi-turn,
memory-aware conversations -- real money, correctly counted in `AskResponse.cost` and the per-tenant budget -- and
the dashboard's "total spend" card would never move for any of it, because `rag_pipeline.py`'s `cost_tracker.
record_cost(...)` calls only covered embedding/rerank/generation, and the three Day 13-15 extras lived entirely in
`routers/ask.py`, one layer up, where nobody was calling `cost_tracker` at all. A monitoring view that silently
excludes a real, already-billed cost category isn't a smaller version of the truth, it's a wrong one -- this is
the same "zero should mean zero, not unmeasured" discipline Day 16 applied to the budget, now applied to the
dashboard.

## 3. Theory
**Why this is a one-way feed from `routers/ask.py` into `cost_tracker`, not a new accounting system.**
`_extra_cost_fields` (Day 16) already computes the exact dollar amount for each of the three categories, every
time `/ask` or `/ask/stream` runs. Day 17 doesn't recompute anything -- it just also calls `cost_tracker.
record_cost(stage, tokens, cost_usd)` with that same already-computed number, the same function every other stage
in this codebase (`rag_pipeline.py`'s embedding/rerank/generation) already calls. One source of truth
(`_extra_cost_fields`'s dict), two consumers (`AskResponse.cost` via `_apply_extra_costs`, `cost_tracker` via the
new `_record_extra_stage_costs`).

**Why stage names match the tracing spans, not the `CostBreakdown` field prefixes.** `CostBreakdown` uses
`contextualizer_cost_usd`, `memory_extraction_cost_usd`, `memory_embedding_cost_usd` (noun-ish, matching its other
fields like `rerank_cost_usd`). `cost_tracker.cost_by_stage`'s keys instead use `contextualize`, `memory_extract`,
`memory_embedding` -- the exact names of the `tracing.span()`s Day 16 already introduced. This means a trace and
a cost line for the same unit of work are always labeled identically, so "which span cost how much" is a lookup
by the same string in both places, not a mental cross-reference table.

**Why a skipped call still records nothing, not a zero-cost entry.** `_record_extra_stage_costs` only calls
`cost_tracker.record_cost(...)` when the relevant token count is nonzero -- identical to how
`rag_pipeline.py`'s rerank/embedding calls are already guarded (`if search_result.query_tokens:`, `if use_rerank
and deps.reranker is not None:`). A tenant that never uses conversations or memory should see a dashboard with no
`contextualize`/`memory_extract`/`memory_embedding` rows at all, not three rows permanently pinned at `$0.00`
cluttering the view.

**Why `cost_by_stage` is a new dict, not folded into the existing `tokens_by_stage`.** `tokens_by_stage` already
mixes chat tokens (contextualize, generation) and embedding tokens (embedding, memory_embedding) on one numeric
scale, which was already slightly apples-to-oranges before today, and adding three more stages to the same
single-unit view wouldn't make "which stage is actually expensive" any more readable. `cost_by_stage` gives the
dashboard a second, dollar-only table where every row is directly comparable, independent of how many tokens of
which kind it took to get there.

**Why the eval report's breakdown is a snapshot-and-diff of the SAME singleton, not a parallel counter.**
`scripts/run_eval.py` already calls `answer_question` directly (bypassing the router entirely, by design -- see
`docs/day5.md`), which already feeds `cost_tracker` via `rag_pipeline.py`'s own `record_cost` calls. Rather than
build a second, eval-specific cost accumulator, `run_eval.py` just reads `cost_tracker.cost_by_stage` once before
the per-question loop starts (right after indexing, so one-time indexing cost doesn't leak into "per-question"
numbers) and again at the end, and reports the difference. This is free instrumentation: the eval harness gets a
real per-stage breakdown without a single new cost calculation anywhere.

## 4. Architecture
```
app/observability/metrics.py
  CostTracker.cost_by_stage: dict[str, float]          <- NEW, alongside tokens_by_stage
  CostTracker.record_cost(stage, tokens, cost_usd)      now accumulates both dicts
  CostTracker.reset()                                   now clears cost_by_stage too

app/routers/observability.py
  GET /observability/summary -> cost.usd_by_stage       <- NEW key, alongside total_usd/tokens_by_stage
  GET /observability/dashboard                          cost table gets a $ column

app/routers/ask.py
  _record_extra_stage_costs(extra: dict) -> None        <- NEW
    cost_tracker.record_cost("contextualize", ...)        only if contextualizer tokens > 0
    cost_tracker.record_cost("memory_extract", ...)        only if extraction tokens > 0
    cost_tracker.record_cost("memory_embedding", ...)      only if embedding tokens > 0
  /ask, /ask/stream: extra = _extra_cost_fields(...) -> _record_extra_stage_costs(extra) -> _apply_extra_costs(...)

app/eval/models.py
  EvalReport.cost_by_stage: dict[str, float] = {}       <- NEW, alongside cost_usd

app/eval/runner.py
  run_eval(..., cost_by_stage_fn: Callable[[], dict[str, float]] | None = None)
  _aggregate(..., cost_by_stage_fn) -> EvalReport.cost_by_stage = cost_by_stage_fn() if given else {}

scripts/run_eval.py
  cost_by_stage_start = dict(cost_tracker.cost_by_stage)      snapshot, taken AFTER indexing
  cost_by_stage_fn() -> diff against that snapshot, dropping zero/negative entries
  prints "By stage  embedding=$... rerank=$... generation=$..." when any stage is nonzero
```

## 5. Files
NEW: scripts/verify_day17.py, docs/day17.md
CHANGED: app/observability/metrics.py (`CostTracker.cost_by_stage`), app/routers/observability.py
(`cost.usd_by_stage` in the summary JSON, a $ column in the dashboard table, module docstring), app/routers/ask.py
(`_record_extra_stage_costs`, wired into both `/ask` and `/ask/stream`, module docstring), app/eval/models.py
(`EvalReport.cost_by_stage`), app/eval/runner.py (`cost_by_stage_fn` parameter, threaded into `_aggregate`),
scripts/run_eval.py (snapshot/diff against `cost_tracker.cost_by_stage`, printed breakdown line), tests/
test_metrics.py (2 new), tests/test_observability_endpoints.py (1 new), tests/test_eval_runner.py (2 new),
app/main.py (version bump), README.md (Day 17 row, test count, script name, walkthrough step)

## 6-7. Code notes
- `_record_extra_stage_costs` takes the exact same `extra: dict` that `_apply_extra_costs`/`_apply_extra_costs_dict`
  already consume -- it reads `extra["contextualizer_prompt_tokens"] + extra["contextualizer_completion_tokens"]`
  etc. rather than being handed `LLMUsage` objects directly, so there is exactly one place (`_extra_cost_fields`)
  that knows how a `CostBreakdown`-shaped dict maps to token/cost numbers; this function only decides WHETHER and
  WHERE to record them.
- The dashboard's token table and the new $ column read from two different dicts (`tokens_by_stage`,
  `usd_by_stage`) that are guaranteed to have the same keys (both are written together inside one `record_cost`
  call) -- the frontend JS defaults a missing key to `0` defensively anyway, in case an older cached summary ever
  lacks `usd_by_stage` (e.g. a stale browser tab mid-deploy).
- `scripts/run_eval.py`'s snapshot is taken after the `--pdf` indexing branch, not at the very top of `main()` --
  indexing calls `cost_tracker.record_cost("embedding", ...)` too (via `services/ingestion.py`), and that one-time
  cost is already reported separately (`Indexed: ... (~$...)`, printed right after indexing). Snapshotting before
  the per-question loop keeps the diffed breakdown scoped to exactly what the eval run itself spent, matching
  `running_cost`'s own reset-to-`0.0` at the same point.
- `cost_by_stage_fn`'s dict comprehension drops any stage whose diff rounds to `<= 0` -- without this, a report
  run back-to-back with another process activity touching the same global (unlikely in a dedicated `run_eval.py`
  invocation, but the singleton is still process-wide) could show a spurious near-zero or negative entry from
  floating-point rounding; the breakdown is meant to answer "what did THIS run cost, by stage," not "what noise
  exists at the margins."

## 8. Run
```
python scripts/verify_day17.py            # pytest + a real check that the dashboard now counts every stage
python scripts/verify_day17.py --offline  # pytest only

# Manual walkthrough:
curl localhost:8000/observability/summary -H "Authorization: Bearer <key>" | python3 -m json.tool
# -> cost.usd_by_stage now includes "contextualize"/"memory_extract"/"memory_embedding" once a conversation
#    or memory-bearing request has happened; cost.total_usd reflects them too

python scripts/run_eval.py --document-id <id> --golden evals/golden/your_set.json
# -> prints a new "By stage  embedding=$... rerank=$... generation=$..." line (contextualize/memory_*
#    stay absent here, since run_eval.py calls answer_question() directly without conversation_id/memory)
```

## 9. Testing
336 tests total (4 new since Day 16's 332): `test_cost_tracker_tracks_cost_by_stage_separately_from_tokens` and an
extended `test_cost_tracker_reset` (tests/test_metrics.py) prove `CostTracker.cost_by_stage` accumulates and clears
correctly; `test_summary_reflects_contextualizer_and_memory_costs` (tests/test_observability_endpoints.py) is the
real end-to-end proof -- a two-turn conversation with an existing memory fact, then asserting `/observability/
summary`'s `cost.usd_by_stage` has nonzero `contextualize`, `memory_extract`, AND `memory_embedding` entries, and
that `total_usd` is at least their sum; `test_cost_by_stage_fn_is_captured_in_report` and `test_cost_by_stage_
defaults_to_empty_dict` (tests/test_eval_runner.py) cover the optional-callable pass-through in the eval runner.

## 10. Failure cases
No new failure modes. `_record_extra_stage_costs` can only be reached after `_extra_cost_fields` has already
succeeded (it's pure arithmetic on numbers already in hand), so there's no new exception path to fail open around.
`scripts/run_eval.py`'s `cost_by_stage_fn` reads a plain dict snapshot and does subtraction -- the one edge case
(a stage present in the snapshot but gone from the tracker, or vice versa) is handled by `.get(stage, 0.0)`
defaulting cleanly in both directions.

## 11. Security
No new surface. `cost_by_stage` is aggregate, in-process, numeric accounting data with no tenant_id, no document
content, and no new storage -- the same exposure profile `tokens_by_stage` and `total_usd` already had. `/
observability/summary` still returns the same whole-process view it always has (a known, documented limitation
in a multi-tenant deployment, unchanged from Day 9/10 -- see `docs/day10.md`'s discussion of per-tenant vs.
process-wide observability).

## 12. Cost
No new spend is introduced by this change itself -- every dollar `cost_by_stage` now shows was already being
spent and already being counted in `AskResponse.cost` and the per-tenant budget since Day 16. This is purely
about where that same number is also visible.

## 13. Production improvements
The question-embedding cache flagged in `docs/day15.md`/`docs/day16.md` (re-embedding an identical question asked
twice for memory retrieval) is still open; `/observability/summary` is still a single process's in-memory view,
not aggregated across instances (flagged since Day 9) -- a real deployment would want `cost_by_stage` as
Prometheus counters too (`llm_cost_usd_total` already has a `stage` label, so Grafana can already build this exact
view once a real Prometheus server exists, independent of today's in-process dashboard); and containerization
(Docker/Compose) remains the largest unaddressed "production-ready" gap in the project as a whole.

## 14. Interview Q&A
- *Why does `/observability/summary` need its own call into `cost_tracker` -- doesn't `AskResponse.cost` already
  have this data?* `AskResponse.cost` is per-request, returned to the one caller who made that call. `cost_tracker`
  is the process-wide running total the dashboard reads; Day 16 computed the numbers, but nothing fed them into
  the second, aggregate place that also needed them.
- *Why guard every new `record_cost` call with "only if tokens > 0" instead of always recording?* So a tenant
  that never touches conversations or memory sees a dashboard with no misleading `$0.00` rows for features they
  never used -- presence in `cost_by_stage` should mean "this was actually spent at least once," not "this
  category theoretically exists."
- *Why does the eval harness's cost breakdown reuse the global `cost_tracker` instead of its own accumulator?*
  `answer_question` (what `run_eval.py` calls directly) already feeds that singleton via `rag_pipeline.py`'s
  existing `record_cost` calls -- reading it before and after the eval run is a free diff, not a second system
  that could drift from the first.
- *Why is the snapshot for the eval diff taken after indexing, not at the very start of the script?* Indexing's
  embedding cost is a one-time cost already reported separately (`Indexed: ... tokens (~$...)`); snapshotting
  after it keeps the per-question breakdown scoped to what the actual evaluation run spent, matching how
  `running_cost` itself is reset to `0.0` at that same point.

## 15. Day-end checklist
- [ ] `python scripts/verify_day17.py` ends with ALL CHECKS PASSED
- [ ] you can explain why `cost_by_stage` is a separate dict from `tokens_by_stage`, not folded into it
- [ ] you can explain why a skipped call (no conversation, no memories) leaves no entry at all, not a zero one
- [ ] you can explain why `scripts/run_eval.py`'s breakdown needed no new cost computation, only a snapshot/diff

## 16. Learned
Feeding an already-computed number into a second consumer (a monitoring singleton) instead of recomputing or
duplicating the arithmetic that produced it; naming a cost-tracking stage after the tracing span that already
exists for the same unit of work, so two different observability tools agree on vocabulary for free; and getting
a real per-run cost breakdown out of an evaluation harness by reading a shared, already-instrumented counter
before and after, rather than building a parallel accounting path just for eval.

## 17. Next: Day 18
With cost accounting now complete AND fully visible end to end (the dashboard, the eval reports, the per-request
response, and the per-tenant budget all agree), the next natural step is containerization (Docker + docker-compose
for the app and its SQLite/volume layout) -- the largest remaining "production-ready" gap called out repeatedly
since Day 16's tracker. A smaller alternative, if preferred first, is the question-embedding cache for memory
retrieval that's been flagged as open since Day 15.

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
contextualization/memory extraction/memory embeddings, and that accounting now fully surfaced in both the live
observability dashboard and eval reports via a shared per-stage cost breakdown.
NOT YET: question-embedding cache for memory retrieval, batch-embedding of multiple newly extracted facts, real
OTel export, real Prometheus/Grafana deployment, PII redaction mode, RAGAS/DeepEval, LangGraph, Docker/Compose,
OCR.
PROJECT STATUS: ~99%. Ingestion, indexing, retrieval (exact and approximate), grounded generation, evaluation,
tuning, security baseline, observability (including full per-stage cost visibility), multi-tenancy, CI/CD,
vector-store scaling, short-term and long-term agent memory, relevance-ranked memory retrieval, and complete
cost/trace accounting across the entire `/ask` pipeline are done; containerization and a short list of smaller
polish items remain.
