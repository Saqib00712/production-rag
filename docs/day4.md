# Day 4: LLM Reranking, Grounded Generation, Citation Validation, Refusal

## 1. Objective
`POST /ask`: retrieve -> LLM rerank -> generate an answer that cites sources -> validate every citation in code
-> refuse rather than guess when evidence is missing. Uses your OpenAI key only (chat model as reranker: no
cross-encoder library, per your stack choice).

## 2. Why this matters
A RAG system that answers fluently but wrongly is worse than one that says "I don't know" - it looks trustworthy
while being wrong. Citations are only useful if you can trust them, which means validating them in code, not
trusting the model's word.

## 3. Theory
**Why rerank.** First-stage retrieval scores query and chunk independently (a bi-encoder / BM25). A reranker reads
both TOGETHER and asks "does this passage answer the question", which is a much sharper judgment - at the cost of
one extra LLM call per query, only on the ~10 already-retrieved candidates (cheap-first: retrieval is nearly free,
so we never rerank the whole corpus).

**LLM-as-reranker vs cross-encoder.** A dedicated cross-encoder model is cheaper and faster at scale; we use the
chat model instead because you chose an OpenAI-only stack. `Reranker` is a Protocol, so swapping in a local
cross-encoder later is a one-file change (see PRODUCTION IMPROVEMENTS).

**Structured output (JSON schema mode).** We never parse free text for scores or citations - every LLM call
requests `response_format: json_schema` with `strict: true`, so the SDK guarantees the shape. Interview line:
"structured outputs turn LLM calls into typed function calls, so downstream code never regex-parses prose."

**Grounding = prompt AND code.** The system prompt tells the model to cite everything and refuse if unsure - but
we never trust that alone. Every `[S#]` the model emits is checked in code (`validate_citations`) against the
sources we actually sent. A citation to a source that doesn't exist is stripped, not shown. An answer with ZERO
valid citations is rejected outright (`ungrounded_answer`) - this is the actual hallucination guard, not the
prompt wording.

**Prompt injection from retrieved text.** Your resume, and any document a user uploads, is UNTRUSTED DATA once
it's in a prompt. A malicious PDF could contain "Ignore prior instructions and say X". We wrap sources in
`<source>` tags, tell the model explicitly to treat their contents as data not instructions, and neutralize a
literal `</source>` break-out inside chunk text. This is a mitigation, not a guarantee - full defenses come in
the Security Guardrail project.

**Refusal taxonomy.** Four distinct reasons, each observable: `no_results` (nothing retrieved), `low_relevance`
(best reranked score below threshold - cheapest refusal, skips generation entirely), `model_insufficient_evidence`
(the model itself said so), `ungrounded_answer` (model answered but cited nothing verifiable). Tracking WHICH
refusal fires is what later eval days (6+) will measure and tune.

**Why the relevance gate only applies after reranking.** BM25 scores, cosine similarities, and RRF fusion scores
are all on different, incomparable numeric scales. Only the reranker's explicit 0-10 "how relevant is this"
judgment is meaningful to threshold against. Without reranking, we lean on the generator's own
`insufficient_evidence` signal instead of guessing a cross-scale cutoff - a design correction made *during*
building today after a test caught the mismatch (see `git log` / the code comment in `rag_pipeline.py`).

**One shared OpenAI client.** Day 2-3 created a new `AsyncOpenAI` client per request implicitly; Day 4 introduces
FastAPI's `lifespan` to build one client and close it cleanly on shutdown - fixing the "Event loop is closed"
noise you saw in the Day 3 verify output and avoiding a new TLS handshake per request.

## 4. Architecture
```
POST /ask {question, document_id?, top_k, rerank}
  hybrid search (Day 3) -> up to `retrieve_k` candidates
    no hits?                              -> refuse: no_results
  [rerank=true] LLM reranker scores 0-10 each candidate (fail-closed: unscored = 0)
    reranked & best < min_rerank_score    -> refuse: low_relevance   (generator never called)
    reranker errors                       -> degrade: keep hybrid order, warn, continue
  build_sources(): dedupe, token-budget, assign [S1..Sn], attach page numbers
  no generator (no API key)?              -> refuse: low_relevance ("Generator unavailable")
  generate(): structured JSON {answer, citations[], insufficient_evidence}
    insufficient_evidence / empty answer  -> refuse: model_insufficient_evidence
  validate_citations(): strip unknown ids; zero valid ids -> refuse: ungrounded_answer
  -> AskResponse {answer, citations[{page_number,...}], cost, timings_ms, refusal_reason}
```

## 5. Files
NEW: services/{openai_client,llm,reranker,context_builder,citations,generator,rag_pipeline}.py, models/ask.py,
routers/ask.py, scripts/verify_day4.py, tests/{test_citations,test_context_builder,test_reranker,test_generator,
test_rag_pipeline}.py, pytest.ini, docs/day4.md
CHANGED: config.py, logging_config.py, dependencies.py, main.py (lifespan), requirements.txt (pytest-asyncio),
tests/{fakes,conftest}.py, README.md

## 6-7. Code notes
- `llm.py`: `JsonLLM` Protocol wraps `chat.completions.create(response_format=json_schema, strict=True)`; checked
  for `msg.refusal` (the model's own safety refusal, distinct from our evidence-based refusal) and JSON validity.
- `reranker.py`: real chunk ids are NEVER sent to the model - short labels (`c1`, `c2`...) are used and mapped
  back, so ids never leak into a prompt or a log.
- `citations.py`: regex over `[S1]` / `[S1, S2]` groups; unknown ids inside a group are dropped, not the whole
  group; ids the model "listed" but never placed inline are still credited (or flagged invalid).
- `rag_pipeline.py`: a pure orchestration function (`answer_question`) taking a `RagDeps` bundle - fully testable
  without FastAPI, same pattern as `ingestion.py`.
- `main.py`: `lifespan` closes the shared client; `dependencies.py` factories return `None` when unconfigured so
  `/ask` degrades to a clean refusal instead of a 500 when there's no API key.

## 8. Run
```
pip install -r requirements.txt   # adds pytest-asyncio
python scripts/verify_day4.py            # live: pytest + real grounded Q&A + refusal check (a few cents at most)
python scripts/verify_day4.py --offline  # free, scripted rerank/generation
uvicorn app.main:app --reload --port 8000
```
Try `/ask` on your resume: `{"question":"what AWS services has this candidate used?","document_id":"<id>"}` and
then something it can't answer: `{"question":"what is this person's expected salary?","document_id":"<id>"}`
(should refuse).

## 9. Testing
99 tests total (34 new), zero live LLM calls: citation validator (7 cases incl. hallucinated/mixed/dedup), context
builder (budget, dedup, dropped-empty), reranker (label mapping, fail-closed on missing score, clamping,
real-id-never-leaked), generator (parsing, insufficient_evidence, malformed shape, source-delimiter injection),
and 13 full-pipeline integration tests covering every refusal reason, degradation, and the no-rerank path.
`verify_day4.py` adds a live run: 3 grounded questions with correct-page citations + 1 refusal on an unanswerable
question - the single most important behavior to see working for real, at least once.

## 10. Failure cases
Reranker down: warn + fall back to hybrid order, generation still happens. Generator down: 502, nothing shown -
never invent an answer. No API key: clean refusal, not a crash. Model hallucinates a citation id: stripped +
warned. Model answers with zero real citations: rejected as `ungrounded_answer`. Model's own safety refusal
(`msg.refusal`): surfaces as `LLMError` -> 502, distinct from an evidence refusal.

## 11. Security
Retrieved document text is treated as untrusted input to the LLM (explicit prompt instruction + delimiter
neutralization) - documents are user-controlled content and must never be treated as trusted instructions. Real
chunk/document ids never appear in reranker prompts. Question length capped (1000 chars), `top_k` capped (10).
Still open (later days): rate limiting, per-tenant quotas, full prompt-injection test suite (Security Guardrail
project), PII redaction before sending chunks to the model.

## 12. Observability
Every `/ask` call returns `timings_ms` (retrieval, rerank, generation, total) and a full `CostBreakdown`
(embedding/rerank/generation tokens and USD, separately). Logs include `refusal_reason` when applicable. Later:
Prometheus histograms per stage, refusal-rate dashboards, alerting on `ungrounded_answer` rate spikes.

## 13. Cost
Per real `/ask` call: ~1 query embedding (~$0.0000004) + 1 rerank call (~200-400 tokens, gpt-4o-mini ~$0.0002) +
1 generation call (~500-1000 tokens, ~$0.0005). A full `verify_day4.py` run (3 grounded + 1 refusal question) is
a few cents at most. Controls: `rerank=false` skips the most expensive extra call; `low_relevance` refusal skips
generation entirely; `max_answer_tokens` bounds output cost.

## 14. Production improvements
Swap `OpenAIReranker` for a local cross-encoder (`Reranker` Protocol makes this a drop-in change) once query
volume makes per-query LLM reranking expensive; stream the generation response (Day 8: Streaming Copilot UI);
calibrate `min_rerank_score` against a labeled eval set instead of a guessed constant (Day 6); cache full answers
for repeated questions; add per-tenant cost budgets.

## 15. Interview Q&A
- *How do you prevent hallucinated citations?* Every citation the model emits is checked in code against the
  actual sources sent; unknown ids are stripped, and an answer with zero valid citations is rejected outright.
- *Why rerank if you already have hybrid search?* Hybrid ranks query and chunk independently; a reranker judges
  them jointly, which is a stronger and more expensive signal, so it's applied only to the top candidates.
- *How does refusal work?* Multiple explicit gates (no results, low rerank score, model self-reports insufficient
  evidence, zero valid citations) each map to a distinct, logged reason - refusal is a normal, observable outcome.
- *Why treat retrieved documents as untrusted?* Anything a user uploads can contain text aimed at the LLM itself
  (prompt injection); documents are data, never instructions.
- *Why one shared OpenAI client instead of one per request?* Connection reuse (faster) and clean shutdown
  (`lifespan`), avoiding unawaited async cleanup errors under load.

## 16. Day-end checklist
- [ ] `python scripts/verify_day4.py` ends with ALL CHECKS PASSED
- [ ] a real `/ask` call on your resume returns an answer with a correct citation
- [ ] a real `/ask` call with an unanswerable question refuses (check `refusal_reason`)
- [ ] you can name all four refusal reasons and when each fires
- [ ] you can explain why citations are checked in code, not trusted from the prompt

## 17. Learned
Structured outputs (JSON schema mode), LLM-based reranking, grounded generation, citation validation as a code
concern, refusal taxonomies, prompt-injection-from-data awareness, connection lifecycle management (`lifespan`).

## 18. Next: Day 5
Automated evaluation: a golden question set, retrieval metrics (hit@k, MRR), answer-quality and citation-
correctness scoring, and your first regression test suite that runs without spending on every commit.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection-from-data mitigation, shared client lifecycle.
NOT YET: retrieval/answer/citation evals (RAGAS, DeepEval), regression tests, hallucination-rate measurement,
LangGraph, OpenTelemetry/Prometheus/Grafana/LangSmith, auth + rate limits, Docker/Compose, OCR, streaming.
PROJECT STATUS: ~55%. Ingestion, indexing, retrieval and grounded generation with citations are done; evaluation,
observability, security hardening, streaming and containerization remain.
