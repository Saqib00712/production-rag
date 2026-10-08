# Day 8: Streaming /ask Responses (Server-Sent Events)

## 1. Objective
Add `POST /ask/stream`, returning the answer token-by-token over Server-Sent Events, while keeping every safety
guarantee built since Day 4 - grounding, citation validation, refusal on unverifiable answers. The non-streaming
`/ask` from Day 4 is untouched and still the right choice for programmatic/non-UI callers.

## 2. Why this matters
A user staring at a blank screen for 3-5 seconds while `/ask` computes a full JSON response feels broken, even
when it isn't. Every production chat UI streams. But streaming collides with something this project cares about
a lot: citation validation needs the COMPLETE answer text before it can check whether a citation is real. Day 8 is
about resolving that tension honestly, not papering over it.

## 3. Theory
**Why not stream the structured JSON output from Day 4 directly.** `/ask` uses `response_format=json_schema` so
`{"answer": ..., "citations": [...], "insufficient_evidence": ...}` arrives as one clean object. Streaming THAT
would mean streaming raw JSON characters - a user would watch `{"ans` `wer": "` `Ref` `unds` render, which is
useless as a chat experience. Getting real prose streaming AND structured fields simultaneously requires
incrementally parsing partial JSON (extracting just the growing "answer" string value while ignoring the rest of
the object until it's complete) - genuine production complexity. Day 8 takes the simpler, honest alternative:
plain-text generation with inline `[S1]` citations for the streaming endpoint specifically, reusing the exact
same regex-based `validate_citations()` from Day 4 to parse them once the full text has arrived. This is a real
design trade-off, not a shortcut swept under the rug - documented explicitly in `services/llm.py`.

**The cost of that trade-off: one lost signal.** JSON mode gives a separate `insufficient_evidence` boolean,
distinct from "cited nothing." Plain text has no equivalent field - so the streaming path can't tell "the model
explicitly said it doesn't know" apart from "the model just forgot to cite." Both collapse into one refusal
reason, `ungrounded_answer`. This is stated plainly in the code and here, not hidden: structured outputs buy you
cleaner failure-mode distinctions; plain-text streaming gives some of that up for a better UX. A fair trade to
know you're making, not something to be caught off guard by six months from now.

**What "validate after the full text arrives" means for the user experience: the "correction" event.** A model
could stream a page of plausible-looking text and still, only once it's done, turn out to have cited nothing
real. The pipeline can't un-render what the client already displayed. So when this happens, the stream emits an
explicit `correction` event carrying the real refusal message, and the client is expected to REPLACE the
displayed answer, not append to it. This is a genuine, user-visible cost of combining streaming with a
verification backstop - and it's the right trade anyway, because showing an unverifiable answer permanently is
worse than a brief flicker before a correction. Name this trade-off directly in an interview and it reads as
engineering maturity, not a bug you're hiding.

**A smaller, related cost: cosmetic drift when an invalid citation is stripped.** If the model cites both a real
source and a hallucinated one (`"...[S1, S9]."`), the FINAL validated text has `[S9]` removed, but the client
already rendered the original `[S1, S9]` via deltas. The final `done` event's `answer` field is the source of
truth; a real UI should replace the displayed text with it on completion rather than trusting the concatenated
deltas verbatim. Flagged as a warning in the response specifically so this isn't a silent inconsistency.

**Refactor: retrieval/rerank/gate logic is now shared.** Rather than duplicate the hybrid-search-then-rerank-
then-gate sequence for a second, streaming code path, `rag_pipeline.py` now factors it into `_retrieve_and_prepare`,
called by both `answer_question` (Day 4, behavior unchanged - all 13 of its tests still pass unmodified) and the
new `answer_question_stream`. They diverge only at the generation step, which is exactly where streaming and
non-streaming genuinely need to differ.

**SSE, not WebSockets.** Server-Sent Events are one-directional (server to client) over plain HTTP, which is all
this needs (the client sends one request, then only receives). Simpler than WebSockets: no separate handshake
protocol, works through ordinary HTTP infrastructure, and `text/event-stream` is natively understood by browser
`EventSource` and trivially parsed by anything else. WebSockets would be the right tool only for the client
needing to send more messages mid-stream (e.g. "stop generating"), which isn't a requirement here.

## 4. Architecture
```
POST /ask/stream {question, document_id?, top_k, rerank}
  _retrieve_and_prepare()  <- SHARED with /ask: hybrid search, optional rerank, relevance gate
    early refusal (no_results / low_relevance / no key)?
      -> yield {event: done, data: <refusal AskResponse>}                          [no deltas at all]
    otherwise:
      -> stream plain-text generation, yielding {event: delta, text: "..."} as tokens arrive
      -> accumulate full_text; capture usage from the final stream chunk
      -> validate_citations(full_text) against real source ids
         zero valid citations?
           -> yield {event: correction, message: <refusal text>}   <- tells client: REPLACE, don't append
           -> yield {event: done, data: <refusal AskResponse>}
         else:
           -> yield {event: done, data: <full AskResponse, citations included>}
```

## 5. Files
NEW: scripts/verify_day8.py, tests/{test_streaming_llm,test_rag_pipeline_stream,test_ask_stream_endpoint}.py,
docs/day8.md
CHANGED: services/llm.py (`StreamChunk`, `StreamingLLM` Protocol, `OpenAIStreamingLLM`), services/generator.py
(`StreamingGenerator` Protocol, `OpenAIStreamingGenerator`, `STREAM_SYSTEM` prompt), services/rag_pipeline.py
(refactored to share `_retrieve_and_prepare`; added `answer_question_stream`), dependencies.py (streaming LLM/
generator factories), routers/ask.py (`POST /ask/stream`), main.py (version bump), tests/fakes.py
(`FakeStreamingGenerator`, `FailingStreamingGenerator`), pytest.ini (`norecursedirs` - fixes old `_backup_before_*`
folders from prior days' installers being incorrectly collected by pytest)

## 6-7. Code notes
- `llm.py`'s `OpenAIStreamingLLM` never sets `response_format` - a deliberate, commented omission, not an oversight.
- `generator.py`'s `OpenAIStreamingGenerator` is a thin wrapper: all it does is build the prompt and delegate to
  `StreamingLLM.stream_completion()`, passing chunks straight through - the same "logic vs. SDK glue" separation
  used everywhere else in this project, and why it's fully testable with a fake `StreamingLLM`.
- `rag_pipeline.py`'s `_Prepared` dataclass is the contract between the shared stage and each generation path -
  notice `answer_question`'s behavior and all its existing tests were verified unchanged after this refactor
  (run them before and after a refactor like this, always).
- `routers/ask.py`: SSE payloads are built explicitly per event type (`delta` -> `{"text": ...}`, `correction` ->
  `{"message": ...}`, `done` -> the full response dict) rather than one generic fallback - an earlier draft used a
  single `.get("data", event)` fallback that accidentally leaked the internal `{"event": ...}` wrapper into the
  payload; caught by the HTTP-level test, not the unit-level one, which is exactly why both layers of tests exist.

## 8. Run
```
pip install -r requirements.txt
python scripts/verify_day8.py            # pytest + a real streamed call (a few cents at most)
python scripts/verify_day8.py --offline  # pytest only
```
Try it directly:
```
curl -N -X POST http://localhost:8000/ask/stream \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"...\", \"document_id\": \"<id>\"}"
```
`-N` disables curl's output buffering so you see tokens arrive live rather than all at once at the end.

## 9. Testing
189 tests total (15 new): `OpenAIStreamingLLM` chunk parsing (3, using a fake SDK-shaped async iterator), the
streaming pipeline orchestration (7, covering the happy path, no-results, ungrounded-with-correction, mid-stream
failure, no-generator-configured, hallucinated-citation-stripped-with-warning, and usage-feeds-cost), and full
HTTP-level SSE parsing against the real wired-up app (5, including that `/ask/stream` shares the same rate limiter
as `/ask`). `verify_day8.py` adds a live check that a real stream produces multiple delta chunks before the final
citation-validated answer.

## 10. Failure cases
The LLM stream fails partway through (network drop, provider error mid-generation): caught, a `correction` event
replaces whatever partial text had streamed, and `done` carries a clean refusal - never leaves a truncated,
unexplained answer on screen. Client disconnects before the stream finishes: the async generator simply stops
being iterated; no special handling needed since nothing was ever fully "committed" until the `done` event.

## 11. Security
Identical guarantees to `/ask` (Day 4/7): sources remain untrusted data in the prompt, real chunk ids never leak,
and the citation-validation backstop still applies. `/ask/stream` sits under the same `/ask` rate-limit prefix, so
the throttling from Day 7 protects it automatically without extra configuration.

## 12. Observability
Both `ask_answered` (Day 4) and the new `ask_stream_answered` log events carry the same fields (request_id,
duration_ms, cost_usd), so dashboards built later don't need special-casing for which endpoint answered a
question.

## 13. Cost
Streaming and non-streaming generation cost the same in tokens - streaming changes WHEN you see the answer, not
how much it costs. The live verify step makes one real streamed call, a few cents at most.

## 14. Production improvements
Incremental JSON parsing to regain the `insufficient_evidence` distinction while still streaming prose (real
complexity, worth it once the collapsed refusal reason becomes a measured problem via Day 5's eval harness, not
before); a heartbeat/keep-alive comment line for long-idle streams behind certain proxies; client-side
reconnection/resume semantics for dropped connections; consider WebSockets only if a future feature needs the
client to send interrupt/steering messages mid-stream.

## 15. Interview Q&A
- *How do you stream an LLM response while still validating citations?* Stream plain text with inline citation
  markers, accumulate the full text, then run the same citation validator used for non-streaming responses once
  the stream completes - accepting that verification is inherently a whole-text operation.
- *What happens if a streamed answer turns out to be unverifiable after the fact?* An explicit correction event
  tells the client to replace the displayed text with a refusal - a visible but honest cost of verifying after
  the fact, better than silently leaving an unverifiable answer on screen.
- *Why SSE instead of WebSockets here?* The client only needs to receive, never send mid-stream; SSE is simpler,
  works over plain HTTP, and is natively supported by browsers via EventSource.
- *What's lost by streaming plain text instead of structured JSON?* A clean, separate "the model says it doesn't
  know" signal - it collapses into the same refusal path as "the model didn't cite anything," which the code and
  docs state explicitly rather than hiding.

## 16. Day-end checklist
- [ ] `python scripts/verify_day8.py` ends with ALL CHECKS PASSED
- [ ] you've called `/ask/stream` yourself and watched tokens arrive live via curl
- [ ] you can explain why the streaming endpoint doesn't use `response_format=json_schema`
- [ ] you can explain what a `correction` event means and why a real UI must replace, not append, on receiving one
- [ ] you can explain the one refusal-reason distinction that streaming mode loses versus non-streaming `/ask`

## 17. Learned
SSE vs. WebSockets, the real tension between token streaming and whole-text verification, explicit correction
events as an honest UX pattern, and safely refactoring shared pipeline logic while proving existing behavior is
unchanged via the existing test suite.

## 18. Next: Day 9
Observability: structured tracing across the full request lifecycle (tying the per-router `request_id`s and the
audit-log id from Day 7 into one propagated trace), Prometheus-style metrics (latency percentiles, token/cost
counters, cache hit rate), and a first real dashboard view of the system's health.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection mitigation + test suite, shared client lifecycle, golden datasets, retrieval
metrics, citation-based answer correctness, LLM-as-judge, eval CLI + reports, data-driven tuning, offline
retrieval regression testing, rate limiting, audit logging, PII-aware ingestion, streaming responses (SSE).
NOT YET: OpenTelemetry tracing, Prometheus/Grafana, LangSmith, authentication, per-tenant quotas, PII redaction
mode, RAGAS/DeepEval, CI-wired gating, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~80%. Ingestion, indexing, retrieval, grounded generation (streaming and non-streaming),
evaluation, tuning, and baseline security guardrails are done; observability, auth/multi-tenancy, and
containerization remain.
