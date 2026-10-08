# Day 10: Multi-Tenancy and Authentication

## 1. Objective
API key authentication, strict per-tenant document/chunk isolation (even for searches with no document filter),
a second rate limiter keyed by tenant instead of IP, and a per-tenant daily cost budget enforced before spend
happens, not after. `scripts/manage_keys.py` to create/list/revoke keys.

## 2. Why this matters
Every endpoint so far trusted every caller completely: no identity, no isolation, one shared cost budget for the
whole server. That's fine for solo local development and became increasingly wrong as the system grew real
guardrails around it (rate limits, budgets, observability) that only make sense PER CUSTOMER. Today is the day
"the server" becomes "a server serving multiple tenants who must never see each other's data."

## 3. Theory
**API keys as opaque bearer tokens, hashed at rest.** A key is a long random string (`secrets.token_hex`), never
derived from anything guessable. The server stores only its SHA-256 hash and compares hashes at lookup time - the
same principle as password storage, for the same reason: a stolen database should not yield usable credentials.
The plaintext key is shown exactly once, at creation (`scripts/manage_keys.py create`) - there is deliberately no
"show my key again" feature, matching how real API providers handle this.

**Why 404, not 403, for another tenant's document.** A 403 ("Forbidden") confirms the resource EXISTS but you can't
have it. A 404 reveals nothing - from the outside, "it doesn't exist" and "it exists but isn't yours" are
indistinguishable. This is a standard multi-tenant security practice: never let an authenticated-but-unauthorized
caller learn that a specific id is valid.

**The real isolation boundary: tenant_id is never optional in chunk queries.** `document_id` has always been an
OPTIONAL filter (search across one document, or across everything the caller can see). Before today, "everything"
meant everything in the whole database - a search with no `document_id` would have silently searched every
tenant's content. The fix isn't adding a tenant check only when convenient; it's making `tenant_id` a REQUIRED
parameter on every chunk-reading repository method (`keyword_search`, `load_vectors`, `get_chunks_by_ids`,
`list_chunks`), so there is no code path that can forget it. `test_search_never_crosses_tenants_even_with_no_
document_filter` proves this is actually enforced, not just intended.

**What's deliberately still SHARED across tenants.** The embedding cache (keyed by content hash + model) and the
query-embedding cache stay shared - the same text always produces the same vector regardless of who uploaded it,
and a content hash alone reveals nothing about tenant identity or document content. Sharing them is a real cost
saving (two tenants asking "what is the refund policy" share one cached query embedding) with no isolation cost.
This is a deliberate choice, stated explicitly in `chunk_repository.py` - not an oversight that happens to be safe.

**Why a SECOND rate limiter, not replacing Day 7's.** Day 7's per-IP middleware limiter runs BEFORE any route
dependency, including auth - it's the only thing protecting against anonymous/pre-auth abuse (hammering `/ask`
with no key at all still costs CPU to reject). Day 10's per-tenant limiter runs AFTER auth resolves an identity,
as a FastAPI dependency (`enforce_tenant_limits`), because middleware cannot yet know which tenant is calling.
Two layers, two purposes: IP-based is the cheap first line of defense; tenant-based is the real per-customer quota.

**Why the daily budget is a persisted SQLite ledger, not an in-memory counter.** Day 9's `CostTracker` is
explicitly process-memory-only and resets on restart - fine for a live dashboard, wrong for a spending limit
(restarting the server would let anyone's quota silently reset). `tenant_usage` in `tenant_repository.py` persists
to disk specifically because enforcement must survive a crash or redeploy; a dashboard metric and a billing
control have different durability requirements even though both are "just a number going up."

**Why budget is checked BEFORE the call, not after.** Recording spend after a call happens is necessary (you can't
know the real cost until OpenAI returns usage) but insufficient on its own - checking the budget first means a
tenant already over budget is stopped before incurring ANY further cost, rather than always being allowed "one
more" expensive call that pushes them further over.

**Backward compatibility, not a breaking migration.** `DocumentMetadata.tenant_id` and `Chunk.tenant_id` both
default to `"default"`, and `ChunkRepository` self-heals an old `chunks` table with no `tenant_id` column via
`ALTER TABLE` (same self-healing pattern as Day 3's FTS index) - a document or database from Day 1-9 keeps working
under tenant `"default"` rather than breaking or needing a manual migration step.

## 4. Architecture
```
Request -> [Day 7 IP rate limiter, middleware, pre-auth] -> routing
         -> get_current_tenant (Authorization: Bearer <key> or X-API-Key)
              hash key -> TenantRepository.lookup() -> 401 if missing/invalid/revoked
         -> [paid endpoints only] enforce_tenant_limits
              per-tenant sliding-window rate check -> 429
              today's spend >= daily budget?        -> 402
         -> route handler, now holding tenant.tenant_id
              documents: tenant_id stamped into DocumentMetadata at upload;
                         _load_parsed() 404s if the stored tenant_id doesn't match the caller
              search/ask: tenant_id passed to SearchService -> ChunkRepository,
                          filtering chunks/vectors/FTS rows UNCONDITIONALLY
         -> after a paid call completes: record_tenant_spend() -> tenant_usage ledger (persisted)
```

## 5. Files
NEW: app/auth.py, app/repositories/tenant_repository.py, scripts/{manage_keys,verify_day10}.py, tests/test_auth.py,
docs/day10.md
CHANGED: config.py (tenant/auth settings), dependencies.py (`get_tenant_repository`), models/document.py +
models/chunk.py (`tenant_id` fields), repositories/chunk_repository.py (tenant_id column + migration + filtering
everywhere), services/{ingestion,search,rag_pipeline}.py (thread tenant_id through), routers/{documents,search,
ask}.py (require auth, enforce isolation), main.py (second rate limiter, dev-key bootstrap on startup),
tests/conftest.py (default authenticated-tenant override + tenant rate limiter reset), tests/
{test_repository,test_repository_search,test_rag_pipeline_stream}.py (updated call sites for new signatures)

## 6-7. Code notes
- `auth.py`: accepts either `Authorization: Bearer <key>` or `X-API-Key: <key>` - some HTTP tools make custom
  headers easier to set than Bearer auth, and supporting both costs nothing.
- `tenant_repository.py`: `revoke()` matches by hash PREFIX so an operator can act using only the truncated hash
  `manage_keys.py list` shows - the plaintext key is never available again to use for lookup.
- `chunk_repository.py`: schema creation is split into `_CHUNKS_TABLE` (just the table) and `_SCHEMA` (indexes +
  other tables) specifically so the tenant_id migration can run BETWEEN them - the `CREATE INDEX ON
  chunks(tenant_id)` statement would otherwise fail against a pre-Day-10 table that doesn't have the column yet.
- `tests/conftest.py`: the `_default_authenticated_tenant` autouse fixture is what keeps ~200 pre-Day-10 tests
  passing unchanged - they were never testing auth, so auth is transparently satisfied for them, the same pattern
  as overriding `get_settings`/`get_embedder_factory`. Tests that actually exercise auth remove this override.

## 8. Run
```
pip install -r requirements.txt
python scripts/verify_day10.py            # pytest + a real two-tenant isolation check (no OpenAI key needed for this part)
python scripts/manage_keys.py create --tenant your-name --label "my key"
python scripts/manage_keys.py list
uvicorn app.main:app --reload --port 8000
```
On first startup with no keys yet, the server prints a one-time bootstrap key to the console - use that, or create
your own with `manage_keys.py`. Then call any endpoint with `-H "Authorization: Bearer <key>"`.

## 9. Testing
231 tests total (15 new): repository-level tenant isolation and schema migration (3, including a search with no
document filter never crossing tenants, and a simulated pre-Day-10 database upgrading cleanly), and a dedicated
real-auth suite (12: missing/invalid/revoked keys, both header styles, cross-tenant 404s on documents and chunks,
cross-tenant search isolation, per-tenant rate limiting, and per-tenant budget enforcement - including that one
tenant's exhausted budget never blocks a different tenant). `verify_day10.py`'s live step needs no OpenAI key at
all (it uses a fake embedder) since it is purely testing auth and isolation, not retrieval quality.

## 10. Failure cases
A key that matches no tenant, or matches a revoked one: 401, identical response either way. A document_id that
exists but belongs to another tenant: 404, identical to a truly nonexistent id. A tenant over its daily budget:
402 on every paid endpoint until the next UTC day, with no way to self-reset. A pre-Day-10 SQLite file: migrates
automatically on first open rather than crashing or requiring a manual step.

## 11. Security
This IS today's subject. Residual gaps, named plainly: no key rotation/expiry (keys are valid until explicitly
revoked); no scoped permissions (a key can do everything its tenant can do - no read-only keys yet); the
bootstrap dev key is printed to stdout, fine for local dev, wrong for a real deployment's logs; `/metrics` and
`/observability/*` still have no auth at all (flagged already on Day 9, still true).

## 12. Observability
`record_tenant_spend` feeds the same persisted ledger `enforce_tenant_limits` reads from - one source of truth for
"how much has this tenant spent today," not two numbers that could drift. Day 9's process-wide dashboard is
unchanged; a genuine per-tenant usage dashboard is a natural Day 9+10 combination left for later (see Production
Improvements).

## 13. Cost
Auth, isolation, rate limiting, and budget checks are all local computation - $0 marginal cost, and they run
before any paid call, so a blocked request never reaches OpenAI at all. The live verify step intentionally uses a
fake embedder specifically so Day 10 can be fully verified without spending anything.

## 14. Production improvements
Key scopes/permissions (read-only vs. full access); key expiry and rotation reminders; a per-tenant view in the
Day 9 dashboard; moving the bootstrap-key print to a one-time setup command instead of every cold start; replacing
the hand-rolled API-key scheme with OAuth2/JWT if integrating with an existing identity provider becomes a
requirement; moving tenant rate limiting to Redis for multi-instance correctness (same gap Day 7 flagged for the
IP limiter, now doubled).

## 15. Interview Q&A
- *How do you implement multi-tenant data isolation in a shared database?* Every tenant-owned row carries a
  `tenant_id`, and every read path treats it as a REQUIRED filter, not an optional one - the dangerous bug is a
  query that CAN run without it, not forgetting to pass it in one call site.
- *Why 404 instead of 403 for a resource that belongs to someone else?* A 403 confirms the resource exists; 404
  reveals nothing, which is the correct behavior when existence itself is sensitive information.
- *How do you store API keys safely?* Hash them (SHA-256 here) before storage and compare hashes at lookup time,
  identical reasoning to password storage - a stolen database yields no usable credentials.
- *Why two separate rate limiters instead of one?* They protect against different things at different points in
  the request lifecycle: IP-based runs pre-auth as a cheap anonymous-abuse backstop; tenant-based runs post-auth
  as the real per-customer quota, keyed by an identity that doesn't exist yet when the first one runs.
- *Why is a spending budget persisted to disk instead of kept in memory like your other metrics?* A billing
  control must survive a restart; a dashboard metric resetting on deploy is a minor inconvenience, but a quota
  resetting on deploy is a security hole someone could exploit by forcing a crash.

## 16. Day-end checklist
- [ ] `python scripts/verify_day10.py` ends with ALL CHECKS PASSED
- [ ] you've created a key with `manage_keys.py` and used it to call a real endpoint
- [ ] you've confirmed a second tenant gets 404, not 403, touching the first tenant's document
- [ ] you can explain why `tenant_id` must be a required, not optional, repository parameter
- [ ] you can explain why the embedding cache stays shared across tenants while chunks do not

## 17. Learned
API key authentication with hashed storage, tenant isolation as a required-not-optional query parameter, the
403-vs-404 information-disclosure distinction, layered rate limiting (pre-auth vs. post-auth), and persisted vs.
in-memory state as a durability decision tied to what the state is FOR (dashboard vs. billing control).

## 18. Next: Day 11
CI/CD: wiring the test suite, the offline retrieval regression gate (Day 6), and a cost-bounded live eval run into
a GitHub Actions pipeline - so "did this change break something" becomes an automatic check on every push, not a
manual `pytest` run someone has to remember to do.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection mitigation + test suite, shared client lifecycle, golden datasets, retrieval
metrics, citation-based answer correctness, LLM-as-judge, eval CLI + reports, data-driven tuning, offline
retrieval regression testing, rate limiting, audit logging, PII-aware ingestion, streaming responses (SSE),
unified request tracing, Prometheus metrics, live observability dashboard, API key auth, multi-tenant data
isolation, per-tenant rate limits + persisted daily cost budgets, key management CLI.
NOT YET: CI/CD pipeline, key scopes/expiry, real OTel export, real Prometheus/Grafana deployment, PII redaction
mode, RAGAS/DeepEval, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~90%. Ingestion, indexing, retrieval, grounded generation, evaluation, tuning, security baseline,
observability, and multi-tenancy are done; CI/CD and containerization remain as the major pieces.
