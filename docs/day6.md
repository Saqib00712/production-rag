# Day 6: Tuning Chunk Size, retrieve_k, and Rerank Threshold - With Data

## 1. Objective
Stop guessing at hyperparameters. `scripts/tune.py` sweeps chunk size and `retrieve_k` against your golden set
using real measurements, and replays real rerank scores against candidate thresholds without extra API cost.
Plus your first regression test: an offline, $0 pytest that fails the build if retrieval quality drops.

## 2. Why this matters
Days 1-4 picked chunk_size=350, retrieve_k=10, min_rerank_score=4.0 - all reasonable defaults, none of them
measured. Day 5 built the ruler (hit@k, MRR, false-refusal rate). Today is the day those numbers actually change
a setting, which is the entire point of building an eval harness: an eval you never act on is decoration.

## 3. Theory
**Tune one variable at a time, not a grid search.** A 3-value chunk-size x 3-value retrieve_k x 5-value threshold
grid is 45 combinations. Each chunk-size value needs a full re-index (real embedding cost); each threshold value,
if tested by re-running `/ask`, needs real rerank+generation calls per question. A full grid here would cost real
money and produce a result that's hard to explain ("hit@3 was 71% in cell (350, 10, 4.0) but 68% in (350, 15,
4.0)... why?"). Three focused, sequential sweeps (fix two, vary one) tell a clear story and cost a fraction as
much. This is a real trade-off production teams make constantly: grid search is more thorough and almost always
not worth what it costs.

**Why the chunk-size sweep skips rerank/generation entirely.** Chunking only affects what's AVAILABLE to retrieve
and how it's split. Measuring hit@k/MRR right after hybrid search isolates that effect from downstream noise
(a reranker could paper over bad chunking, or a bad prompt could make good chunking look bad). Interview line:
"I test each pipeline stage's contribution independently before testing the whole pipeline end-to-end."

**Why the retrieve_k sweep needs only ONE re-index.** `retrieve_k` doesn't change what's stored - it changes how
many candidates you ask the index for. So we fetch the maximum k once per question (`top_k=max(retrieve_ks)`,
query embeddings cached from Day 3) and re-slice the same ranked list in plain Python for every k value. Zero
extra embedding calls after the first pass over the golden set.

**Why the threshold sweep needs NO extra LLM calls at all.** A rerank score is a property of (query, document) -
raising the threshold can only ever flip an "answered" question to "refused"; it can never change what score the
reranker gave in the first place. So we run the real pipeline ONCE per question (unavoidable - this is the only
step that must call the reranker and generator for real), record each question's top rerank score, and simulate
every candidate threshold afterward in `app/eval/tuning.py` with plain comparisons. This is the single most
cost-effective idea in today's code: threshold tuning is FREE once you have the scores.

**Reading a threshold trade-off.** For each candidate threshold, "lost" = correct answers that would newly refuse
(bad - a false refusal you're introducing) and "blocked" = wrong answers that would newly refuse (good - a
hallucination you're preventing). `net_score = blocked - lost`. The best threshold isn't always the one with the
highest net score in isolation - `recommend_threshold` breaks ties toward the SMALLEST change from your current
setting, because an unnecessarily large jump is itself a risk on data you haven't tested.

**Regression testing: what can and can't be tested offline.** `tests/test_regression_retrieval.py` asserts
hit@3 and MRR against a fixed bar using a FAKE embedder (`BowEmbedder`) - it catches real bugs (someone breaks
hybrid fusion, a chunking change silently drops a page) for $0, on every single `pytest` run. It deliberately does
NOT assert answer/refusal correctness, because refusal quality depends on a real model's judgment call that a
scripted reranker can't meaningfully exercise. This is an honest, permanent limitation, not a gap to code around:
retrieval regressions get a cheap, instant, every-commit gate; answer-quality regressions need a live run
(`scripts/verify_day6.py` without `--offline`, or `scripts/run_eval.py`) - run periodically or before a release,
not on every save. Day 11 (CI/CD) formalizes exactly this split into two separate pipeline stages.

## 4. Architecture
```
scripts/tune.py --pdf your.pdf --golden your_set.json
  [1] for each chunk_size in sweep:
        re-index (isolated temp storage) -> hybrid search per question -> hit@k/MRR
      -> best_chunk_point() picks the winner (hit@3, then MRR)
  [2] re-index ONCE at the winning chunk_size
      for each retrieve_k in sweep: re-slice one cached search -> hit@k/MRR
      -> smallest_k_reaching() picks the cheapest sufficient k
  [3] [unless --skip-threshold] run the REAL /ask pipeline once per question
      -> simulate_thresholds() replays scores against candidate thresholds, $0 extra
      -> recommend_threshold() picks the best net (blocked-bad minus lost-correct)
  -> evals/reports/tune-<timestamp>.json + printed recommendations for .env

tests/test_regression_retrieval.py  (runs with plain `pytest`, $0, every commit)
  fake embedder + fixed 8-page doc + golden set -> hit@3/MRR must clear a fixed bar
```

## 5. Files
NEW: app/eval/tuning.py, scripts/{tune,verify_day6}.py, tests/{test_tuning,test_regression_retrieval}.py, docs/day6.md
CHANGED: nothing in `app/routers` or `app/services` - like Day 5, tuning sits on top of the existing pipeline via
its public functions and `Settings`, changing behavior only through `.env` values you choose to apply.

## 6-7. Code notes
- `tuning.py`: every function here takes already-collected data and returns a decision - no I/O, so all 10 tests
  run in milliseconds.
- `simulate_thresholds`: skips questions that are `currently_refused` - raising a threshold cannot un-refuse
  something, so re-checking those would only ever add noise, never information.
- `tune.py`: three isolated `tempfile.TemporaryDirectory` blocks for the chunk sweep (each chunk size gets its own
  SQLite file, so sweeps never contaminate each other or your real indexed data).
- `test_regression_retrieval.py`: reuses the exact PAGES content `evals/golden/sample_8page.json` was written
  against - the test asserts that link explicitly (`len(golden) == 8`) so the two files can't silently drift apart.

## 8. Run
```
pip install -r requirements.txt
python scripts/verify_day6.py            # pytest + a small live tuning run (a few cents at most)
python scripts/verify_day6.py --offline  # pytest only
python scripts/tune.py --pdf your.pdf --golden evals/golden/your_set.json
python scripts/tune.py --pdf your.pdf --golden evals/golden/your_set.json --skip-threshold  # cheapest option
```
After a run, apply a recommendation by editing the matching line in `.env` (`CHUNK_SIZE_TOKENS`, `MAX_INDEX_TOKENS`
stays as-is, `min_rerank_score` isn't yet an env var - see Production Improvements) and re-run your eval to confirm
it actually helped on your real document, not just the sweep's small sample.

## 9. Testing
142 tests total (11 new): `tuning.py`'s pure functions (10 cases: chunk-point comparison incl. tie-break, k
selection incl. unreachable target, threshold simulation incl. skip-already-refused, recommendation incl. ties and
empty input) plus the offline retrieval regression gate (1 test, asserting hit@3 >= 85% and MRR >= 0.70 on the
canonical 8-page set). `verify_day6.py` adds a small live sweep (2 chunk sizes x 2 retrieve_k values, threshold
step skipped by default in the verify script to keep it cheap) proving `tune.py` runs correctly end-to-end.

## 10. Failure cases
Golden set with zero answerable questions: `tune.py` exits with a clear error rather than dividing by zero or
tuning against nothing. A chunk size sweep value larger than the document's total content: still works, just
fewer/larger chunks - the metrics will show whether that's good or bad rather than the code guessing.

## 11. Security
No new surface: `tune.py` and the regression test are local developer tools, not exposed endpoints.

## 12. Observability
`tune.py` writes a full sweep report to `evals/reports/tune-<timestamp>.json` - a permanent record of what was
tried and what was recommended, useful later when someone (including future you) asks "why is chunk size 300 and
not the 350 every tutorial uses?"

## 13. Cost
Chunk sweep: one re-index per value (embedding cost only, no rerank/generation) - a few thousand tokens total for
a small document. retrieve_k sweep: effectively free (one embedding pass, cached, re-sliced in Python). Threshold
step: one real `/ask` call per golden question (rerank + generation) - the same cost as running `run_eval.py`
once. `--skip-threshold` avoids this if you only want the (cheaper) retrieval-side recommendations.

## 14. Production improvements
Promote `min_rerank_score`, and the values chosen from a chunk-size sweep, from constants/`.env` into a small
versioned config so a change can be reviewed like code; track eval reports over time to catch slow drift, not just
single-run snapshots; expand the golden set (Day 5's point still holds - the best sets grow from real traffic);
eventually replace the hand-rolled sweep loop with a proper experiment tracker once you're running these often
enough that `evals/reports/*.json` stops being enough.

## 15. Interview Q&A
- *How do you choose a chunk size in production?* Measure hit@k/MRR at a few candidate sizes against a real golden
  set, isolating chunking from reranking/generation - not by picking a number that "feels right" or copying a
  tutorial default.
- *Why not grid search all your hyperparameters together?* Cost and interpretability - each additional dimension
  multiplies API calls, and a single clear "hit@3 rose when X changed" is more actionable than a 45-cell table.
- *How do you tune a relevance threshold without spending on every candidate value?* If the threshold only gates
  on a score you've already computed once, replay it in plain code against many candidate values instead of
  re-querying the model for each one.
- *What can and can't a regression test catch without live API calls?* Wiring and retrieval-logic regressions,
  yes (fake embeddings still expose real bugs in fusion, chunking, ranking code); genuine answer-quality or
  refusal-judgment regressions, no - those need periodic live evaluation.

## 16. Day-end checklist
- [ ] `python scripts/verify_day6.py` ends with ALL CHECKS PASSED
- [ ] you've run `scripts/tune.py` against your own resume and golden set at least once
- [ ] you can explain why threshold tuning needs zero extra API calls once scores are collected
- [ ] you can explain what the offline regression test can and cannot catch
- [ ] `pytest` alone (no flags) still passes, including the new regression gate

## 17. Learned
One-variable-at-a-time tuning vs grid search, isolating pipeline stages when measuring, replaying already-paid-for
scores instead of re-querying, and the honest boundary between offline and live regression testing.

## 18. Next: Day 7
Security guardrails: input validation hardening, prompt-injection test cases against the `/ask` endpoint, PII
awareness, and rate limiting basics - closing gaps flagged since Day 4 rather than leaving them as "later".

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection-from-data mitigation, shared client lifecycle, golden datasets, retrieval
metrics, citation-based answer correctness, LLM-as-judge, eval CLI + reports, data-driven chunk/retrieve_k/
threshold tuning, offline retrieval regression testing.
NOT YET: RAGAS/DeepEval integration, CI-wired regression gating (GitHub Actions), prompt-injection test suite,
rate limiting/auth, PII redaction, LangGraph, OpenTelemetry/Prometheus/Grafana/LangSmith, Docker/Compose, OCR,
streaming.
PROJECT STATUS: ~70%. Ingestion, indexing, retrieval, grounded generation, evaluation, and data-driven tuning are
done; security hardening, observability, streaming, and containerization remain.
