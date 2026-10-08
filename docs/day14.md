# Day 14: Long-Term, Cross-Conversation Memory

## 1. Objective
Give the system memory that outlives a single conversation: a new `GET/POST /memories` and `DELETE
/memories/{id}` surface a tenant's accumulated facts, every `/ask` and `/ask/stream` call now feeds that tenant's
remembered facts into generation regardless of `conversation_id`, and after a real (non-refused) exchange an LLM
call decides whether anything durable was stated and, if so, stores it for every future conversation.

## 2. Why this matters
Day 13 solved multi-turn memory WITHIN one conversation: a `conversation_id` ties turns together, and
contextualization resolves a follow-up using that conversation's own history. But that history is gone the
moment a new conversation starts - a user who mentioned "I always want answers in metric units" yesterday gets
no benefit from it in a brand new chat today, because `ConversationRepository.get_turns()` is scoped to exactly
one `conversation_id` and nothing reads across conversations. Real assistants that feel like they "know" a user
do so across sessions, not just within one - that's a genuinely different kind of state than Day 13's, scoped to
the TENANT rather than to any one conversation, and it's the natural next layer once short-term memory exists.

## 3. Theory
**Why this is a different problem from Day 13's, not an extension of it.** Day 13's `Turn` history is read by
`conversation_id` - by construction, it cannot be seen by a different conversation, even for the same tenant.
Long-term memory needs the opposite scoping: readable by every future conversation for that tenant, written once
and re-used indefinitely. These are genuinely different data shapes (a turn-indexed transcript vs. a flat set of
facts) and genuinely different repositories (`ConversationRepository` vs. `MemoryRepository`), not the same table
with a longer retention window.

**Extraction, not storage of everything.** The naive approach - store every question and answer forever, search
it like a document - would work, but it also means every greeting, every "what is the warranty?", every
one-off question becomes "memory," drowning the handful of facts that are actually worth remembering (the
project's "Next" preview in `docs/day13.md` called this out as a server-stored-summaries vs. vector-indexed-store
question). Day 14 takes the summary-extraction path: `services/memory_extractor.py` asks a small, cheap LLM call,
after a turn completes, whether anything DURABLE was stated - a preference, a role, a constraint, an identity
fact - as opposed to that turn's one-off content. Most turns yield nothing, by design: the system prompt
explicitly says "most turns have nothing worth remembering" and instructs an empty list as the default answer.

**Why extraction happens AFTER the turn, not before or during.** The fact (if any) typically only becomes visible
once the full exchange - question AND answer - exists ("I always want answers in metric units, by the way - what
is the refund window?" needs the model to actually recognize the aside as separate from the question being
asked). Running extraction as a genuinely separate, subsequent LLM call - rather than trying to fold "also
extract any facts" into the generation call's own schema - keeps the generation prompt focused purely on
grounded answering (Day 4's contract) and keeps extraction's own prompt focused purely on fact-spotting, each
simpler and more reliable than a combined one would be.

**Why memory notes are never citable, and never passed to retrieval.** `format_memory_notes()` appends
remembered facts to the generation prompt in a clearly separate block with explicit instructions: use ONLY to
interpret the question, never cite as a source, never treat as document fact. This is a hard line, not a
style choice - `services/generator.py`'s whole grounding contract (every factual claim ends in a `[S#]` citation
to actual indexed document text) breaks the moment a long-term memory fact could itself become a cited "source."
A memory fact is context for interpreting what's being asked, exactly like the Day 13 contextualizer uses
conversation history to interpret a follow-up - never evidence for what's true in the documents. For the same
reason, memory notes are NOT fed into `services/contextualizer.py`'s rewrite step this round (see Section 14) -
scope was deliberately kept to generation-time conditioning plus the `/memories` management surface, not every
place conversation history already touches the pipeline.

**Cost + noise discipline: cap and dedupe, don't grow forever.** An LLM extracting "durable facts" after every
turn, indefinitely, would eventually flood the generation prompt with stale or near-duplicate notes - diluting
the few that matter and inflating every future request's token cost (this project's established refrain since
Day 4's single-candidate rerank skip and Day 6's cost-driven tuning). `MemoryRepository.add_memory()` dedupes a
restated fact (case/whitespace-insensitive exact match) rather than storing it twice, and enforces
`memory_max_facts` by evicting the OLDEST fact first whenever a new one pushes the tenant over the cap - keeping
the store small and, in practice, recent: a fact still worth keeping tends to get restated and re-extracted if
it still matters, so dropping a stale one rarely loses anything live.

**Why extraction is skipped entirely on a refusal.** `no_results`, `low_relevance`, `model_insufficient_evidence`,
and `ungrounded_answer` all mean no real exchange of information happened - there is nothing durable to learn
from a request that produced the refusal message. Skipping extraction here is the same "skip work a cheap check
proves is unnecessary" discipline as the Day 13 contextualizer skipping a conversation's first turn.

**Why `/memories` exists as its own surface, not just an internal mechanism.** An automatically extracted fact
can be wrong, stale, or simply something the user doesn't want remembered. `GET /memories` makes what's being
remembered inspectable, `DELETE /memories/{id}` makes it correctable, and `POST /memories` lets a fact be stated
directly without waiting for a future turn to surface it organically - the same transparency-and-control
principle behind any "what does this system remember about me" surface.

## 4. Architecture
```
app/repositories/memory_repository.py
  memories(memory_id, tenant_id, content, created_at)
  MemoryRepository: add_memory (dedupes + FIFO-caps via max_memories), get_owner, list_memories(limit=...),
                     delete_memory

app/services/memory_extractor.py
  MemoryExtractor (Protocol): extract(question, answer, existing) -> list[str] (usually empty)
  OpenAIMemoryExtractor: one structured-output call per real (non-refused) turn

app/services/generator.py
  format_memory_notes(): renders remembered facts as a clearly separate, non-citable prompt block
  Generator.generate / StreamingGenerator.generate_stream: both take an optional memory_notes param now

app/routers/ask.py  (_fetch_memory_notes / _maybe_extract_memories helpers, shared by /ask and /ask/stream)
  every call (conversation_id or not)
    -> memory_notes = MemoryRepository.list_memories(tenant_id, limit=memory_max_facts)
    -> answer_question(..., memory_notes=memory_notes)      <- independent of conversation_id entirely
    -> on a REAL (non-refused) answer: MemoryExtractor.extract(question, answer, existing=memory_notes)
       -> any returned facts: MemoryRepository.add_memory(tenant_id, fact, max_memories=memory_max_facts)

app/services/rag_pipeline.py
  answer_question / answer_question_stream now take an optional `memory_notes`, threaded straight through to
  the generator call (memory notes never affect retrieval or the relevance gate - generation-time only)

app/routers/memories.py
  GET    /memories          -> {memories: [...]}            (this tenant's facts)
  POST   /memories          -> {memory_id, content, created_at}   (manual add, 201)
  DELETE /memories/{id}     -> 404 if unowned, else 204
```

## 5. Files
NEW: app/repositories/memory_repository.py, app/models/memory.py, app/services/memory_extractor.py,
app/routers/memories.py, scripts/verify_day14.py, docs/day14.md, tests/test_memory_repository.py,
tests/test_memory_extractor.py, tests/test_memories_endpoint.py
CHANGED: app/config.py (`memory_db_path`, `memory_max_facts`, `openai_memory_extractor_model`),
app/dependencies.py (`get_memory_repository`, `get_memory_extractor_factory`), app/services/generator.py
(`format_memory_notes`, optional `memory_notes` param on both generator protocols/implementations),
app/services/rag_pipeline.py (threaded `memory_notes` through both entry points), app/routers/ask.py
(`_fetch_memory_notes` / `_maybe_extract_memories` wired into both `/ask` and `/ask/stream`), app/main.py (mount
memories router, version bump), tests/conftest.py (`memory_extractor_factory` fixture, `ask_client` override,
`memory_db_path` in the `settings` fixture), tests/fakes.py (`ScriptedMemoryExtractorFactory`; generator/
streaming-generator fakes updated to accept and record `memory_notes`), tests/test_ask_stream_endpoint.py
(memory-notes-in-generation and extraction-after-stream tests), README.md (Day 14 row, test count, script name,
`/memories` walkthrough step)

## 6-7. Code notes
- `MemoryRepository.add_memory()` dedupes BEFORE inserting (`lower(trim(content))` match), returning the
  existing `memory_id` rather than creating a near-duplicate row - a model re-stating a known fact in slightly
  different casing/whitespace must not grow the store.
- The FIFO cap (`_enforce_cap`) runs AFTER insert, inside the same connection/transaction as the insert itself -
  so a crash between insert and eviction can never leave the tenant permanently over the cap.
- `format_memory_notes()` lives in `generator.py`, not `rag_pipeline.py` or `memory_extractor.py` - it is
  specifically about how the PROMPT renders these facts, which is the generator's concern alone, the same
  reasoning that already put `format_sources()` there on Day 4.
- `routers/ask.py`'s `_fetch_memory_notes` and `_maybe_extract_memories` are shared, byte-for-byte, by `/ask` and
  `/ask/stream` - same discipline as Day 13's `_contextualize`/`_append_turn` helpers, so the two endpoints'
  memory behavior can't silently drift apart.
- `/ask/stream`'s extraction call happens INSIDE `event_source()`, after the "done" payload is built - unlike
  Day 13's conversation 404 check, there's no ownership gate here to resolve before streaming starts (memory is
  always the caller's own tenant), so there's no reason to hoist it earlier the way `_contextualize` had to be.
- `get_memory_extractor_factory` fails open exactly like every other LLM-backed factory in this project
  (`get_reranker_factory`, `get_generator_factory`, `get_contextualizer_factory`): a missing `OPENAI_API_KEY`
  returns `None`, and `_maybe_extract_memories` just skips extraction rather than failing the request. The actual
  `extract()` call is ALSO wrapped in `try/except LLMError`, logging `memory_extraction_failed` and returning,
  for the identical reason Day 13's `_contextualize` added its own `LLMError` catch: a live provider failure
  during the call itself is a different failure mode than a missing key, and both must degrade the same way.
- Extraction is skipped via one guard at the top of `_maybe_extract_memories`: `if refusal_reason is not None:
  return` - checked BEFORE the factory is even called, so a refusal costs nothing extra, not even a wasted
  factory construction.

## 8. Run
```
python scripts/verify_day14.py            # pytest + a real long-term-memory exchange, across conversations
python scripts/verify_day14.py --offline  # pytest only

# Manual walkthrough:
curl -X POST localhost:8000/memories -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"content": "Prefers answers in metric units."}'
curl -X POST localhost:8000/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "What is the refund window?", "document_id": "<id>"}'
# ^ no conversation_id at all -- the memory above is still used in generation
curl localhost:8000/memories -H "Authorization: Bearer <key>"
curl -X DELETE localhost:8000/memories/<memory_id> -H "Authorization: Bearer <key>"
```

## 9. Testing
303 tests total (26 new since Day 13's 277): the memory repository (7: add/list round trip, per-tenant scoping,
dedup-on-restate, get_owner known/unknown, delete, FIFO eviction at the cap, limit returns the most recent), the
memory extractor (4, against a fake JSON LLM - zero network - covering a real extraction, the empty-list default
for an ordinary turn, blank-fact filtering, and that existing facts are passed into the prompt for dedup), and
HTTP-level tests across `/memories`, `/ask`, and `/ask/stream` (15: manual CRUD and cross-tenant 404 isolation on
`/memories`; memory notes reaching the generator with and without any memories present; memory shared identically
across two DIFFERENT conversation_ids for the same tenant - the central Day 14 behavior Day 13 could not provide;
a durable fact extracted and stored after a real answer; extraction skipped on a refusal; extraction failure not
breaking the request; extraction skipped when no API key is configured; the same two behaviors repeated for the
streaming endpoint).

## 10. Failure cases
`memory_id` for a memory that doesn't exist, or belongs to another tenant: 404, identical either way (same
404-not-403 principle as conversations and documents). No `OPENAI_API_KEY` configured: the extractor factory
returns `None` and the turn completes normally with nothing learned - generation itself still uses whatever
facts were ALREADY stored (memory usage and memory extraction fail independently of each other). A turn that
ends in a refusal never reaches the extractor at all, by design (Section 3) - not a failure case, but worth
distinguishing from an actual extractor error. `memory_max_facts: 0` on a manual `POST /memories`: the fact is
inserted and then immediately evicted by its own cap enforcement; the endpoint returns 422 rather than a
misleading 500 or a silent no-op, since "add this" silently doing nothing would be far more confusing than an
explicit error naming the setting responsible.

## 11. Security
Memory isolation reuses the exact 404-not-403 mechanism established for documents (Day 10) and conversations
(Day 13): every `/memories` lookup is scoped to the caller's authenticated `tenant_id`, and a memory that exists
but isn't yours returns the same 404 as one that doesn't exist at all - confirmed by
`test_someone_elses_memory_is_404_not_403`, which checks both the direct delete AND that the fact doesn't leak
into that tenant's own `GET /memories` listing. The memory extractor's prompt only ever sees ONE tenant's own
existing facts (passed in as `existing`) and that tenant's own question/answer pair - no cross-tenant data ever
enters the call, the same containment `_contextualize` already guaranteed for conversation history.

## 12. Observability
Like Day 13's contextualization, memory extraction doesn't yet get its own `tracing.span()` - it runs in the
router, after the main `answer_question`/`answer_question_stream` call has already returned, one layer outside
the Day 9 span instrumentation. Flagged deliberately, not missed by accident: `span("memory_extract")` around
the router-level call, mirroring the already-flagged `span("contextualize")` gap from Day 13, is a natural single
follow-up that would bring both Day 13 and Day 14's extra LLM calls into the same unified trace.

## 13. Cost
Zero extra cost on every refused turn (extraction is skipped entirely - Section 3) and on every turn where the
extractor's own structured-output call returns an empty list (the common case for an ordinary question). Every
REAL, answered turn costs one extra small, cheap structured-output call for extraction, on top of whatever
contextualization (Day 13, on follow-ups) and generation already cost - like Day 13's contextualizer cost, this
is not yet reflected in `AskResponse.cost` or counted against the Day 10 per-tenant daily budget (a real,
tracked gap, not a rounding choice - see Section 14, which now has two uncounted LLM calls instead of one).

## 14. Production improvements
Fold BOTH the contextualizer's (Day 13) and the memory extractor's (Day 14) token usage into `AskResponse.cost`
and the per-tenant daily budget check - today neither is counted, and that gap only grows as more "extra" LLM
calls get added; add `tracing.span("memory_extract")` next to the already-flagged `span("contextualize")` gap so
both show up in the unified trace breakdown; feed long-term memory into `services/contextualizer.py`'s rewrite
too (today memory only affects the generation prompt, not retrieval - a stated preference like "I call it
'warranty', not 'guarantee'" could help resolve a follow-up the same way conversation history already does, but
that's deliberately out of scope for today, see Section 3); compare this server-stored-facts approach against a
vector-indexed memory store (embed each fact, retrieve the top-k relevant ones per question instead of dumping
every stored fact into every prompt) once `memory_max_facts` starts being too small a window for tenants with
many distinct facts; a way to mark a fact "pinned" so cap eviction never removes it; batching extraction instead
of one LLM call per turn, for high-volume tenants where even a cheap call per turn adds up.

## 15. Interview Q&A
- *How is this different from Day 13's conversation memory?* Day 13 is scoped to ONE `conversation_id` and reads
  only that conversation's own turns. Day 14 is scoped to the TENANT and is read by every future conversation,
  including ones with no relationship to where a fact was first learned - it needs its own storage and its own
  read path for exactly that reason.
- *Why extract facts with a separate LLM call instead of asking the generator to also report them?* Keeps each
  prompt doing one job well: generation stays focused on grounded answering from sources (Day 4's contract),
  extraction stays focused on fact-spotting from the completed exchange - a combined schema would complicate both
  without a real benefit.
- *Why can't a remembered fact be cited as a source?* It isn't document content - it's unverified, self-reported
  context about the user. Allowing it as a citable `[S#]` would break the grounding guarantee this whole project
  is built around: every claim traces back to real indexed text.
- *How do you keep the memory store from growing forever?* Dedup on restate (case/whitespace-insensitive) plus a
  hard cap with oldest-first eviction - the same "don't pay for or carry work that's provably unnecessary"
  discipline used everywhere else in this project (rerank skip, contextualizer skip, now memory growth).
- *Why skip extraction on a refusal?* No real exchange happened - there's nothing an LLM call over a refusal
  message could possibly learn that would be worth the cost of making the call.

## 16. Day-end checklist
- [ ] `python scripts/verify_day14.py` ends with ALL CHECKS PASSED
- [ ] you can explain why long-term memory needs its own repository, not a longer Day 13 history window
- [ ] you can explain why memory notes are never cited and never fed into retrieval
- [ ] you can explain why extraction happens AFTER a turn completes, and only on a real (non-refused) answer
- [ ] you can explain how the memory store stays bounded (dedup + FIFO cap) instead of growing forever

## 17. Learned
Designing a second, deliberately DIFFERENT kind of conversational state (tenant-scoped facts vs. Day 13's
conversation-scoped turns) rather than stretching one data shape to cover both; extraction-after-the-fact as the
pattern for turning an ordinary exchange into durable memory without flooding the store with every one-off
question; keeping a hard boundary between "context for interpreting a question" (memory notes, conversation
history) and "evidence for what's true" (cited document sources) even as more context sources get added to the
same prompt; and, again, tracking a known cost/observability gap explicitly (two now-uncounted LLM calls) rather
than letting it quietly compound.

## 18. Next: Day 15
With both short-term (Day 13) and long-term (Day 14) memory in place from server-stored facts/summaries, a
natural next step is comparing that approach against a vector-indexed memory store: embed each remembered fact
and retrieve only the top-k most relevant ones per question using the SAME hybrid search machinery already built
for documents (Day 3), instead of dumping every stored fact into every prompt - directly useful once a tenant's
`memory_max_facts` window starts being too small to hold everything worth remembering about them.

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
cross-conversation memory (tenant-scoped fact extraction, storage, and generation-time use).
NOT YET: vector-indexed memory store (facts retrieved by relevance instead of all injected), contextualizer- and
extractor-cost accounted in the budget ledger, memory fed into query contextualization, real OTel export, real
Prometheus/Grafana deployment, PII redaction mode, RAGAS/DeepEval, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~97%. Ingestion, indexing, retrieval (exact and approximate), grounded generation, evaluation,
tuning, security baseline, observability, multi-tenancy, CI/CD, vector-store scaling, short-term agent memory,
and long-term cross-conversation memory are done; vector-indexed memory retrieval and containerization remain as
the major pieces.
