# Day 9: Observability - Unified Tracing, Prometheus Metrics, a Live Dashboard

## 1. Objective
Fix the "two different ids for one request" gap flagged since Day 7, add per-stage span timing that finally
answers Day 8's "why did that take 5.65s, where did the time go" question directly in the logs, expose Prometheus
metrics at `/metrics`, and ship a small live dashboard at `/observability/dashboard` - all without standing up any
external infrastructure.

## 2. Why this matters
"It's slow" and "it's expensive" are useless bug reports without a breakdown. Before today, one HTTP request
produced TWO uncorrelated ids (the audit middleware's, and whatever `str(uuid.uuid4())` each router generated for
its own service-layer logs) - making "show me everything that happened for this one request" require guessing
which id matched which log line. Today fixes that structurally, then builds the two things every production system
needs on top of it: a metrics endpoint something can scrape, and a way to actually look at the numbers.

## 3. Theory
**contextvars: the real mechanism behind distributed tracing.** A `ContextVar` holds per-async-task state that
flows automatically through `await` calls within the same logical task, with no manual parameter threading and no
global-mutable-state race conditions between concurrent requests. This is not a simplification of how real tracing
libraries work - it's literally the primitive OpenTelemetry's Python SDK is built on. `tracing.py`'s `start_trace()`
/ `get_trace_id()` / `span()` implement the same core idea (one id per request, nested timed spans) with zero
infrastructure, which is exactly what "one process, no collector" needs right now.

**Why not adopt full OpenTelemetry today.** OTel's value is in EXPORTING traces to a real backend (Jaeger, Tempo,
an OTel Collector) for cross-service, cross-time analysis - genuinely valuable, but real infrastructure to run, not
a pip install. Building the same core concept by hand first means: (a) you understand what OTel is actually doing
under the hood before treating it as a black box, and (b) swapping this module for real OTel later is a like-for-
like replacement (`get_trace_id()` / `span()` become OTel API calls) rather than a rewrite - flagged explicitly as
the upgrade path once a real backend exists to send data to (a later, deployment-focused day).

**Prometheus's actual division of labor.** `prometheus_client` (a pure Python library) lets THIS process record
counters and histograms and expose them as text at `/metrics` - that part needs no server. Computing a P95 from a
histogram's bucket counts via PromQL, though, is something a real Prometheus SERVER does at query time after
scraping `/metrics` on a schedule. We are not running that server today. So `/metrics` is real and correctly
formatted right now - but nothing LOCALLY reads a percentile back out of it. That's an honest gap, not glossed
over: `LatencyRecorder` is a small, separate, in-process rolling-window structure that exists ONLY to give
`/observability/summary` a percentile NOW, and is explicitly documented as something a real Prometheus+Grafana
setup makes redundant once it exists (a later, deployment-focused day).

**Spans as the direct answer to "why did /ask/stream take 5.65s?" (Day 8).** Each pipeline stage - retrieval,
rerank, generation - is wrapped in `tracing.span("name")`. The audit middleware collects every span recorded during
a request and logs them as one array on the SAME line as the request's overall duration. Grep one `request_id` and
you see the full waterfall: 0.3s retrieval, 3.1s rerank, 2.2s generation - not a mystery total.

**Why metrics recording lives inside the pipeline, not just the middleware.** HTTP-level metrics (request count,
duration) belong in middleware - they apply to every route uniformly. But token counts, cost, cache hits, and
refusal reasons are DOMAIN facts about what the RAG pipeline did, not generic HTTP facts - they're recorded at the
exact point they're already being computed (`ingestion.py`, `rag_pipeline.py`), which is also where the existing
per-call `IndexReport`/`AskResponse` cost fields come from. One computation, two destinations (the response body,
and the running counters) - not two separate calculations that could drift apart.

## 4. Architecture
```
Request -> AuditLogMiddleware
             start_trace(new_id)                      <- ONE id for this request, everywhere
             ... routers/services call get_trace_id() instead of generating their own ...
             ... rag_pipeline wraps stages: with span("retrieval"): / span("rerank"): / span("generation"): ...
           on completion:
             latency_recorder.record(duration)          -> feeds /observability/summary percentiles
             http_requests_total / http_request_duration_seconds .observe()   -> feeds /metrics
             ONE log line: {request_id, duration_ms, spans: [{name, duration_ms}, ...]}   <- the full waterfall

ingestion.py / rag_pipeline.py, at the point cost is already computed:
             cost_tracker.record_cost(stage, tokens, cost_usd)
             cost_tracker.record_cache(cache, hit)
             cost_tracker.record_refusal(reason)

GET /metrics                    -> Prometheus text format (scrape-able)
GET /observability/summary      -> JSON: p50/p95/p99, error rate, cost by stage, cache hit ratio, refusals
GET /observability/dashboard    -> small HTML page polling /observability/summary every 3s
```

## 5. Files
NEW: app/observability/{__init__,tracing,metrics}.py, app/routers/observability.py, scripts/verify_day9.py,
tests/{test_tracing,test_metrics,test_observability_endpoints}.py, docs/day9.md
CHANGED: middleware/audit_log.py (unified trace id, span logging, metrics recording), routers/{documents,search,
ask}.py (use `get_trace_id()` instead of local `uuid.uuid4()`), services/rag_pipeline.py (spans + cost/refusal
metrics around retrieval/rerank/generation), services/ingestion.py (spans + cache/cost metrics), logging_config.py
(new `spans` field), main.py (mounted observability router), tests/conftest.py (autouse reset for the new
singletons), requirements.txt (`prometheus_client`)

## 6-7. Code notes
- `tracing.py`: `span()` is usable even with no active trace (e.g. a unit test, or a script like `run_eval.py`
  calling pipeline code directly) - it just discards what it recorded, rather than raising. Nothing outside an HTTP
  request needs to change to remain compatible.
- `audit_log.py`: `response` is initialized to `None` and checked before setting the response header, because if
  `call_next()` raises, there IS no response to attach a header to - the `finally` block still needs to log and
  record metrics either way.
- `metrics.py`: a fresh `CollectorRegistry()` is used instead of Prometheus's global default registry
  specifically so tests can exist in the same process without duplicate-metric registration errors across test
  files - a real gotcha with `prometheus_client`'s default global state.
- `test_one_trace_id_unifies_audit_log_and_internal_service_logs`: the single most important test added today -
  it doesn't just check that ids exist, it proves the actual Day 7-flagged gap ("two different ids for one
  request") is closed, by asserting equality between the response header and every internal service log line.

## 8. Run
```
pip install -r requirements.txt
python scripts/verify_day9.py            # pytest + one real /ask call, then inspects the trace it produced
python scripts/verify_day9.py --offline  # pytest only
uvicorn app.main:app --reload --port 8000
```
Then open http://localhost:8000/observability/dashboard while making a few requests in another tab/terminal, and
watch the numbers move. `curl -s localhost:8000/metrics | grep http_requests_total` to see raw Prometheus output.

## 9. Testing
216 tests total (27 new): tracing (8, including cross-await propagation), metrics (11, including known-value
percentile checks and Prometheus text format), and observability integration (8, including the dashboard serving
real HTML and, most importantly, the trace-unification proof). All run in milliseconds with zero OpenAI calls.
`verify_day9.py` adds a live check that a real `/ask` call produces a `retrieval` span, a non-zero cost in the
summary, and one consistent id across the audit log, the response header, and internal service logs.

## 10. Failure cases
A request that raises an unhandled exception: the `finally` block still records its (5xx) latency and logs it as
`http_request_failed`, so failures show up in `/observability/summary`'s error rate rather than vanishing. Two test
files both importing `metrics.py`: the dedicated `CollectorRegistry()` (not Prometheus's global default) avoids
the "duplicated timeseries" registration error this would otherwise cause.

## 11. Security
`/metrics` and `/observability/*` are deliberately NOT behind the Day 7 rate limiter (monitoring endpoints
shouldn't be throttled) - but they're also not authenticated yet, meaning right now anyone who can reach the
server can see aggregate cost/token/refusal counts. Not raw document content or citations, but still real
operational fields - flagged for the Day 10 auth work rather than left as a silent gap.

## 12. Observability
This IS today's subject matter. One thing worth naming directly: the dashboard and `/observability/summary` show
THIS PROCESS's in-memory history only - restart the server and it resets. That's correct for local development and
explicitly not a promise of durable, multi-instance history (real Prometheus + Grafana, once deployed, is what
gives you that).

## 13. Cost
Recording metrics is local computation - $0 marginal cost. The live verify step makes one real `/ask` call (a few
cents at most) specifically to produce real spans and real cost numbers to inspect.

## 14. Production improvements
Stand up real Prometheus + Grafana once deployment exists (Day 10+/containerization) and retire `LatencyRecorder`
in favor of PromQL; export traces to a real OTel collector instead of a single consolidated log line; add
authentication to `/metrics` and `/observability/*`; persist metrics across restarts (today's in-memory counters
reset on every deploy, which is fine for dev, not for tracking trends over days).

## 15. Interview Q&A
- *How do you implement request tracing without a tracing backend?* A context variable holding one id per request,
  propagated automatically through async calls, with a lightweight span recorder timing each pipeline stage -
  the same core mechanism real tracing SDKs use, without needing a collector to send data to yet.
- *What's the difference between what Prometheus's client library does and what a Prometheus server does?* The
  client library records and exposes counters/histograms as text; the SERVER scrapes that text on a schedule and
  computes things like percentiles at query time via PromQL - two different responsibilities, easy to conflate.
- *Why compute your own percentiles instead of relying on the Prometheus histogram?* Because no Prometheus server
  is running locally to query; a small in-process rolling window is a deliberate, temporary stand-in for local
  development, explicitly retired once real infrastructure exists.
- *How would you debug a single slow request in production?* Grep its trace/request id across logs and read the
  recorded span breakdown - exactly what today's consolidated audit log line provides, rather than guessing which
  pipeline stage was slow from a single total duration number.

## 16. Day-end checklist
- [ ] `python scripts/verify_day9.py` ends with ALL CHECKS PASSED
- [ ] you've opened `/observability/dashboard` in a browser and watched it update
- [ ] you can find one request's full span breakdown by its request_id in the logs
- [ ] you can explain why `/metrics` alone doesn't give you a percentile without a real Prometheus server
- [ ] you can explain the ROOT problem Day 9 actually fixed (two uncorrelated ids -> one)

## 17. Learned
contextvars as the real mechanism behind request tracing, the actual division of labor between a metrics client
library and a metrics server, span-based latency breakdowns, and building observability as a cross-cutting concern
that domain code feeds without owning.

## 18. Next: Day 10
Multi-tenancy and authentication: API keys, per-tenant document isolation, per-tenant cost/rate quotas built on
top of today's metrics (now that cost and requests are tracked per-process, the next step is tracking them per
tenant), and what changes in the data model to support more than one user safely.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection mitigation + test suite, shared client lifecycle, golden datasets, retrieval
metrics, citation-based answer correctness, LLM-as-judge, eval CLI + reports, data-driven tuning, offline
retrieval regression testing, rate limiting, audit logging, PII-aware ingestion, streaming responses (SSE),
unified request tracing, Prometheus metrics, live observability dashboard.
NOT YET: authentication, per-tenant quotas/isolation, real OTel export, real Prometheus/Grafana deployment,
PII redaction mode, RAGAS/DeepEval, CI-wired gating, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~85%. Ingestion, indexing, retrieval, grounded generation, evaluation, tuning, security baseline,
and observability are done; multi-tenancy/auth and containerization remain.
