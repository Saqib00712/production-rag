# Day 13: Agent Memory - Multi-Turn Conversations with Query Contextualization

## 1. Objective
Turn the one-shot `/ask` endpoint into a multi-turn conversation: a new `POST /conversations` to start one, an
optional `conversation_id` on `/ask` and `/ask/stream` to continue it, and a `GET /conversations/{id}` to read its
history back. The core mechanism is query contextualization: before retrieval runs, a follow-up question is
rewritten into a standalone search query using the conversation's recent history - the ORIGINAL question is still
what gets generated from and shown back to the user.

## 2. Why this matters
Every day through Day 12 answered exactly one question with exactly one retrieval pass. Real conversations don't
work that way: "What's the warranty?" followed by "What about the second one?" is unanswerable by a retriever
that has never seen turn one - "the second one" matches nothing in a vector index or a BM25 query. Without fixing
this, a multi-turn UI built on top of this API would silently retrieve garbage on every follow-up, no matter how
good the underlying RAG pipeline is. This is also the first day the project persists state ACROSS requests that
isn't a document, a tenant, or a cost ledger - a genuinely new kind of entity in the system.

## 3. Theory
**Why "just add chat history to the generation prompt" doesn't fix this.** It's the obvious first idea, and it's
incomplete: putting history in the prompt helps the MODEL understand context once it already has the right
sources in front of it, but it does nothing for RETRIEVAL, which already ran and already failed, before
generation ever sees a token of history. A follow-up like "what about the second one?" embeds to something close
to "second one" - nowhere near "warranty" or "shipping" in vector space, and no BM25 keyword overlap either.
Hybrid search (Day 3) finds nothing useful, and no amount of context in the generation prompt can retroactively
retrieve the right chunks. The failure happens upstream of generation, so the fix has to happen upstream too.

**The fix: contextualize before you retrieve, not after.** `services/contextualizer.py` asks a small, cheap LLM
call to rewrite the follow-up into something that stands alone, using the recent conversation as context: "what
about the second one?" after a turn about warranty and shipping becomes something like "What is the shipping
policy?" - and THAT text is what gets embedded and searched. This is the single idea that makes multi-turn RAG
actually work, and it is a retrieval-time concern, deliberately separate from (and prerequisite to) anything
happening in the generation prompt.

**Why the user's real question is preserved, not replaced.** The rewritten query is used ONLY for retrieval.
`AskResponse.question` is always the user's actual wording; `AskResponse.search_query` is set only when a
rewrite happened and differs from the question, specifically so a client can show (or log, or debug) what was
actually searched without ever showing the user text they didn't type. `Turn.search_query` in the stored history
keeps the same distinction permanently, not just for one response.

**Cost discipline: skip the LLM call when there's nothing to contextualize.** A conversation's first turn is
already standalone by definition - there is no history to resolve a pronoun or an elided reference against.
`OpenAIContextualizer.contextualize()` checks for this FIRST and returns the question unchanged, paying zero
extra tokens, before ever touching the network. This mirrors a pattern this project has used since Day 4 (rerank
only runs when there's more than one candidate worth ranking) and Day 6 (reuse already-collected scores instead
of a fresh pass): never pay for work a cheap check proves is unnecessary.

**Why conversations are a NEW repository, not bolted onto ChunkRepository or TenantRepository.** Same reasoning
as every prior split in this codebase: different concern, different lifecycle. A conversation is neither document
content (`ChunkRepository`) nor identity/billing (`TenantRepository`) - it's its own thing with its own schema
(`conversations`, `conversation_turns`), persisted to the same SQLite file-per-concern pattern established on Day
2 and Day 10, specifically so it survives a server restart: a conversation a user is in the middle of is exactly
the kind of state that must not evaporate because the process happened to restart between two messages.

**Why ownership uses the same 404-not-403 pattern as documents.** `ConversationRepository.get_owner()` exists
for one reason: so `routers/ask.py` and `routers/conversations.py` can return an IDENTICAL 404 whether a
`conversation_id` doesn't exist at all or belongs to a different tenant - exactly the isolation principle Day 10
established for documents, applied to a new resource type rather than re-derived from scratch.

## 4. Architecture
```
app/repositories/conversation_repository.py
  conversations(conversation_id, tenant_id, created_at)
  conversation_turns(turn_id, conversation_id, turn_index, question, search_query, answer, citations_json,
                      refusal_reason, created_at)
  ConversationRepository: create_conversation, get_owner, append_turn, get_turns(limit=...)

app/services/contextualizer.py
  Contextualizer (Protocol): contextualize(question, history) -> standalone search query
  OpenAIContextualizer: skips the LLM call entirely when history is empty; otherwise one structured-output
    call over the last `contextualizer_max_turns` turns

app/routers/ask.py  (_contextualize / _append_turn helpers, shared by /ask and /ask/stream)
  conversation_id given?
    -> 404 if not owned by caller's tenant (ConversationRepository.get_owner)
    -> load recent history -> Contextualizer.contextualize(question, history) -> search_query
    -> answer_question(question=question, search_query=search_query, ...)   <- question UNCHANGED for generation
    -> on success: ConversationRepository.append_turn(...)
  conversation_id omitted -> unchanged Day 1-12 single-turn behavior, byte-for-byte

app/services/rag_pipeline.py
  _retrieve_and_prepare / answer_question / answer_question_stream now take an optional `search_query`;
  retrieval uses `search_query or question`, generation and the response always use `question`.

app/routers/conversations.py
  POST /conversations          -> create_conversation(tenant_id) -> {conversation_id}
  GET  /conversations/{id}     -> 404 if unowned, else {conversation_id, turns: [...]}
```

## 5. Files
NEW: app/repositories/conversation_repository.py, app/models/conversation.py, app/services/contextualizer.py,
app/routers/conversations.py, scripts/verify_day13.py, docs/day13.md, tests/test_conversation_repository.py,
tests/test_contextualizer.py, tests/test_conversations_endpoint.py, evals/golden/sample_8page.json (restored -
see Section 10)
CHANGED: app/models/ask.py (`AskRequest.conversation_id`, `AskResponse.search_query`), app/config.py
(`conversation_db_path`, `contextualizer_max_turns`, `openai_contextualizer_model`), app/dependencies.py
(`get_conversation_repository`, `get_contextualizer_factory`), app/services/rag_pipeline.py (threaded
`search_query` through both entry points and `_refuse`), app/routers/ask.py (conversation wiring on both /ask and
/ask/stream), app/main.py (mount conversations router, version bump), app/middleware/audit_log.py (one-line log
level fix - see Section 10), tests/conftest.py (`contextualizer_factory` fixture, `ask_client` override,
`conversation_db_path` in the `settings` fixture), tests/fakes.py (`ScriptedContextualizerFactory`),
tests/test_ask_stream_endpoint.py (conversation-aware streaming tests)

## 6-7. Code notes
- `contextualizer.py`'s `contextualize()` checks `if not history: return question` as its FIRST line, before
  building any prompt - the cost-saving short-circuit has to be unconditional and first, not a special case
  bolted on after the "normal" path is written.
- `rag_pipeline.py`: every refusal path (`_refuse`, both early and late) now also carries `search_query`, not
  just the success path - a refused answer still searched for something, and that something is worth showing in
  `AskResponse.search_query` on a refusal too, not only when an answer comes back.
- `routers/ask.py`'s `_contextualize` helper is shared, byte-for-byte, by both `/ask` and `/ask/stream` - the
  404 ownership check and the contextualization call happen identically for both, rather than two
  near-duplicate implementations drifting apart over time.
- `/ask/stream`'s conversation lookup happens BEFORE `StreamingResponse` is constructed, not inside
  `event_source()` - once a streaming response starts, the HTTP status is already committed to 200, so a 404
  for an unowned `conversation_id` has to be raised as a real HTTP error before that point, not as an SSE
  `error` event after the fact.
- `get_contextualizer_factory` fails open exactly like `get_reranker_factory`/`get_generator_factory`: a missing
  `OPENAI_API_KEY` makes the factory return `None` rather than raising, and `_contextualize` falls back to the
  raw question rather than failing the whole `/ask` call over a feature that's naturally skippable.
- `_contextualize` also catches `LLMError` around the actual `contextualize()` call, not just the missing-key
  case above -- caught live while smoke-testing the installer against a simulated Day 12 folder with no real
  network access: a contextualizer call that fails mid-request (provider down, network blip) must fall back to
  the raw question the same way a missing key does, not bubble up as an unhandled 500. Same fail-open principle,
  one more failure mode covered.

## 8. Run
```
python scripts/verify_day13.py            # pytest + a real multi-turn conversation (follow-up included)
python scripts/verify_day13.py --offline  # pytest only

# Manual walkthrough:
curl -X POST localhost:8000/conversations -H "Authorization: Bearer <key>"
# -> {"conversation_id": "..."}
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "What is the warranty?", "conversation_id": "<id>"}'
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "What about shipping instead?", "conversation_id": "<id>"}'
curl localhost:8000/conversations/<id> -H "Authorization: Bearer <key>"
```

## 9. Testing
277 tests total (22 new): the conversation repository (7, create/get_owner/append/get_turns including the
limit/ordering behavior and that two conversations never leak into each other), the contextualizer (4, against a
fake JSON LLM - zero network - covering the no-history skip, a real rewrite, an empty-rewrite fallback, and the
`max_turns` window), and HTTP-level conversation tests (11, across `/conversations`, `/ask`, and `/ask/stream`:
create, empty history, unknown-id 404, a turn actually appended, a real contextualized follow-up end-to-end, the
single-turn path never touching the contextualizer at all, a contextualizer failure falling back to the raw
question instead of a 500, and cross-tenant 404 isolation on both the history endpoint and `/ask`/`/ask/stream`).

## 10. Failure cases
`conversation_id` for a conversation that doesn't exist, or belongs to another tenant: 404, identical either way
(never 403 - see Section 3). No `OPENAI_API_KEY` configured: the contextualizer factory returns `None` and the
raw question is used for retrieval instead of failing the request - a follow-up that needed contextualizing will
retrieve poorly in that case, but the endpoint stays up. A rewrite that comes back empty or whitespace-only: the
original question is used instead of searching for nothing.

Two pre-existing issues, unrelated to Day 13's own code, were found and fixed while rebuilding this project
state for today's work: (1) `scripts/verify_day9.py`'s live check set `LOG_LEVEL=WARNING` to quiet third-party
loggers, which also silently suppressed the audit middleware's own INFO-level line - fixed with one line
(`logger.setLevel(logging.INFO)` in `audit_log.py`) so audit lines are never affected by the app's general log
verbosity; this was a test-script false failure, not a bug in request tracing itself (the retrieval/rerank/
generation spans were always being recorded correctly). (2) `evals/golden/sample_8page.json` was missing from
your reconstructed project folder (dropped during a zip that didn't include the `evals/` directory) and has been
restored to match the Day 3/6 question set exactly - `test_retrieval_meets_the_regression_bar` depends on it.

## 11. Security
Conversation isolation reuses Day 10's exact mechanism: every lookup is scoped to the caller's authenticated
`tenant_id`, and a conversation that exists but isn't yours returns the same 404 as one that doesn't exist at
all. The contextualizer LLM call only ever sees that ONE conversation's own history - no cross-conversation or
cross-tenant data ever enters its prompt, since `get_turns()` is always scoped to a single `conversation_id`
that was already ownership-checked before the call.

## 12. Observability
Contextualization doesn't yet get its own `tracing.span()` - it happens in the router, before `RagDeps`/
`answer_question` are even constructed, one layer outside where the Day 9 span instrumentation currently lives.
Flagged here deliberately rather than silently: a `span("contextualize")` around the router-level call is a small,
natural follow-up, not a gap that was missed by accident.

## 13. Cost
Zero extra cost on a conversation's first turn (the contextualizer never calls the LLM when there's no history -
Section 3). Every FOLLOW-UP turn costs one extra small, cheap structured-output call (`contextualizer_max_turns`
caps how much history goes into it, default 5 turns) on top of the existing rerank + generation calls - this is
not reflected in `AskResponse.cost` yet (a real gap, not a rounding choice: see Section 14).

## 14. Production improvements
Fold the contextualizer's own token usage into `AskResponse.cost` and the Day 10 per-tenant daily budget check
(today it's a real, uncounted cost); add a `tracing.span("contextualize")` so it shows up in the unified trace
breakdown next to retrieval/rerank/generation; a conversation TTL/expiry so abandoned conversations don't grow
`conversation_turns` unboundedly; summarizing old turns instead of a flat recency window once conversations run
long enough that `contextualizer_max_turns` starts losing relevant earlier context; letting the client pass
already-known history inline (for a stateless client) as an alternative to server-side `conversation_id` storage.

## 15. Interview Q&A
- *Why doesn't putting chat history in the generation prompt fix multi-turn RAG?* Because retrieval already ran,
  and already failed, before generation sees any history - a follow-up that doesn't embed/keyword-match to
  anything useful on its own will retrieve garbage no matter what context the generator is later given.
- *What's the one thing query contextualization actually does?* Rewrites a follow-up into a standalone query,
  using recent conversation history, BEFORE retrieval - so the thing that gets embedded/searched is something a
  retriever can actually act on.
- *Why keep the rewritten query separate from the user's real question?* The user asked what they asked; showing
  them a machine-rewritten version of their own words back would be confusing and wrong. The rewrite is an
  internal retrieval detail, exposed separately (`search_query`) for transparency/debugging, not substituted for
  the real thing anywhere user-facing.
- *How do you keep this from costing an LLM call on every single question?* Skip it entirely when there's no
  history to resolve against - a conversation's first turn is already standalone by definition.
- *How do you isolate conversations between tenants the same way you isolate documents?* The exact same
  mechanism: ownership is checked server-side on every access, and "doesn't exist" and "exists but isn't yours"
  produce an identical 404.

## 16. Day-end checklist
- [ ] `python scripts/verify_day13.py` ends with ALL CHECKS PASSED
- [ ] you can explain why putting history in the generation prompt alone doesn't fix multi-turn retrieval
- [ ] you can explain what `AskResponse.search_query` is and why it's `None` on a conversation's first turn
- [ ] you can explain why a conversation's first turn never pays for an extra LLM call
- [ ] you can explain why a wrong `conversation_id` returns 404, never 403

## 17. Learned
Query contextualization as the fix for multi-turn retrieval (and why prompt-stuffing history alone cannot be),
designing a new stateful entity (conversations) that reuses this project's established repository/isolation/
cost-discipline patterns rather than inventing new ones, and the discipline of tracking (not silently absorbing)
a known, real gap - contextualizer cost not yet in the budget ledger - instead of either fixing it hastily or
pretending it isn't there.

## 18. Next: Day 14
Long-term memory: carrying facts and preferences across SEPARATE conversations, not just turns within one -
the natural next step once short-term, single-conversation memory (today) exists, and a good point to compare
server-stored summaries against a vector-indexed memory store built on the same retrieval machinery as the
documents themselves.

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
a realistic benchmark, short-term multi-turn conversation memory with query contextualization.
NOT YET: long-term/cross-conversation memory, contextualizer cost accounted in the budget ledger, real OTel
export, real Prometheus/Grafana deployment, PII redaction mode, RAGAS/DeepEval, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~96%. Ingestion, indexing, retrieval (exact and approximate), grounded generation, evaluation,
tuning, security baseline, observability, multi-tenancy, CI/CD, vector-store scaling, and short-term agent
memory are done; long-term memory and containerization remain as the major pieces.
