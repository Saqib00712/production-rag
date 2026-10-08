# Production RAG with Citations

| Day | Status | What it adds |
|-----|--------|--------------|
| 1 | done | FastAPI skeleton, PDF upload, page-preserving extraction |
| 2 | done | Text cleaning, page-aware chunking, OpenAI embeddings, cost tracking, SQLite storage |
| 3 | done | BM25 keyword search, vector search, hybrid (RRF), `/search` endpoint |
| 4 | done | LLM reranking, grounded generation, citation validation, refusal on insufficient evidence, `/ask` endpoint |
| 5 | done | Golden eval sets, retrieval metrics (hit@k, MRR, recall@k), citation-based answer correctness, `scripts/run_eval.py` |
| 6 | done | Data-driven chunk size / retrieve_k / rerank threshold tuning (`scripts/tune.py`), offline retrieval regression test |
| 7 | done | Rate limiting, audit logging, PII-aware ingestion, prompt-injection test suite |
| 8 | done | Streaming `/ask` responses over Server-Sent Events (`POST /ask/stream`) |
| 9 | done | Unified request tracing, Prometheus metrics (`/metrics`), live dashboard (`/observability/dashboard`) |
| 10 | done | API key auth, multi-tenant isolation, per-tenant rate limits + persisted daily cost budgets |
| 11 | done | Two-tier CI/CD (GitHub Actions): free test gate always-on, cost-bounded live eval on `main` |
| 12 | done | Pluggable ANN vector indexing (HNSW) with per-tenant caching, realistic benchmark (`scripts/bench_vector_index.py`) |
| 13 | done | Short-term agent memory: multi-turn conversations (`/conversations`) with query contextualization before retrieval |
| 14 | done | Long-term, cross-conversation memory: tenant-scoped facts (`/memories`), auto-extracted after each turn and used in every future conversation |
| 15 | done | Vector-indexed memory retrieval: only the `memory_top_k` facts most relevant to the question are used, not every stored fact |
| 16 | done | Full cost + trace accounting: contextualizer, memory extraction, and memory embedding costs now counted in `AskResponse.cost`, the per-tenant daily budget, and request tracing |
| 17 | done | `/observability/summary` + eval reports now break cost down by stage (`cost_by_stage`), including the Day 16 categories -- the dashboard's "total spend" no longer undercounts conversation/memory-bearing requests |
| 18 | done | Containerized: `Dockerfile` (multi-stage) + `docker-compose.yml` with a persistent storage volume, a zero-dependency `/health` healthcheck, and no code changes needed to get there |
| 19 | done | Question-embedding cache for memory retrieval: the same question asked twice (or embedded for both memory lookup AND retrieval in one request) now only pays for one real OpenAI embedding call, sharing the Day 3 document-search query cache |
| 20 | done | `requirements.txt` split into prod/dev; CI gains a third job publishing the Docker image to GHCR on pushes to `main`, gated behind both quality gates passing; `hnswlib` (needs a C++ compiler on Windows) moved out of the hard requirements into an optional `requirements-hnsw.txt`, since the default `brute_force` backend never needed it; fixed `verify_day16.py`/`verify_day19.py`'s live checks, which asked about "warranty" against a PDF that never mentioned it |
| 21 | done | Optional real OpenTelemetry export mirroring Day 9's tracing spans (`OTEL_ENABLED`); `PII_MODE=redact` actually masks detected PII before chunking/embedding instead of only warning; vision-based OCR (`OCR_ENABLED`) for scanned PDFs via an OpenAI vision model, capped per document. All three are off by default and fail open if their optional dependency isn't installed |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # requirements.txt alone is production-only (no pytest/httpx/fpdf2) -- see Day 20
# Optional extras -- none of these are needed for the defaults to work:
#   pip install -r requirements-hnsw.txt   # HNSW vector-index backend (default is brute_force)
#   pip install -r requirements-ocr.txt    # vision OCR for scanned PDFs (default: OCR_ENABLED=false)
#   pip install -r requirements-otel.txt   # real OpenTelemetry export (default: OTEL_ENABLED=false)
cp .env.example .env        # then put your real OPENAI_API_KEY in .env
pytest -v                        # 361 tests, no OpenAI calls, $0
python scripts/verify_day21.py   # tests + offline PII/OCR/OTel checks + a real PII-redact round trip
python scripts/bench_vector_index.py --n-chunks 50000   # see brute-force vs HNSW at real scale
python scripts/manage_keys.py create --tenant you --label "my key"   # get an API key first
python scripts/install_git_hooks.py   # optional: run tests before every git push
python scripts/run_eval.py --pdf your.pdf --golden evals/golden/your_set.json   # evaluate YOUR document
python scripts/tune.py --pdf your.pdf --golden evals/golden/your_set.json      # tune chunk size / retrieve_k / threshold
uvicorn app.main:app --reload --port 8000
```

## Or run it in Docker
```bash
cp .env.example .env        # then put your real OPENAI_API_KEY in .env (optional -- /health works without one)
docker compose up --build   # builds the image, starts the API on http://localhost:8000
docker compose exec api python scripts/manage_keys.py create --tenant you --label "my key"
docker compose down          # stop (add -v to also delete the rag_storage volume and start fresh)

# To run the test suite INSIDE a container (the default image is production-only, see Day 20):
docker build --build-arg DEPS_FILE=requirements-dev.txt -t production-rag:dev .
docker run --rm production-rag:dev pytest
```
Every SQLite DB + uploaded PDF lives in the `rag_storage` named volume (see `docker-compose.yml`), so stopping,
restarting, or rebuilding the image never loses data. On a push to `main`, CI also builds and publishes this
same image to `ghcr.io/<owner>/<repo>` (see "CI" below) -- `docker pull` from there needs no build step at all.
Open http://localhost:8000/docs then:
1. `POST /documents/upload` -> copy `document_id`
2. `POST /documents/{id}/index?dry_run=true` -> chunks, tokens, cost (free)
3. `POST /documents/{id}/index` -> real embeddings (fractions of a cent)
4. `GET /documents/{id}/chunks` -> inspect chunks, page numbers, `has_embedding`
5. `POST /search` -> `{"query":"...","mode":"hybrid"}` (keyword | vector | hybrid)
6. `POST /ask` -> `{"question":"...","document_id":"<id>"}` -> cited, grounded answer (or a refusal)
6b. `POST /ask/stream` -> same body, streamed token-by-token over Server-Sent Events
6c. Open `/observability/dashboard` in a browser, or `curl /metrics` / `curl /observability/summary` -- the
    cost table now has a $ column per stage (`cost.usd_by_stage`), including `contextualize`, `memory_extract`,
    and `memory_embedding` once a conversation or memory is involved
6d. Every call above now needs `-H "Authorization: Bearer <your-api-key>"`
7. Write `evals/golden/your_set.json` (see `evals/README.md`) and run `scripts/run_eval.py` against it
8. `POST /conversations` -> copy `conversation_id`, then pass it as `"conversation_id"` on `/ask` or
   `/ask/stream` to get contextualized, remembered multi-turn Q&A. `GET /conversations/{id}` reads the history.
9. `GET /memories` -> facts this tenant's past turns taught the system (used automatically on every future
   `/ask`, with no `conversation_id` needed; only the `memory_top_k` most RELEVANT ones are used per question,
   not every stored fact). `POST /memories` adds one by hand; `DELETE /memories/{id}` removes one.
10. `AskResponse.cost` now has a full breakdown -- `contextualizer_cost_usd`, `memory_extraction_cost_usd`, and
    `memory_embedding_cost_usd` alongside the original `embedding_cost_usd`/`rerank_cost_usd`/`generation_cost_usd`
    -- and `total_cost_usd` is the true sum, counted against the per-tenant daily budget.
11. `scripts/run_eval.py` now prints a `By stage` line and `EvalReport.cost_by_stage` holds the same breakdown,
    read straight off the same in-process cost tracker the dashboard uses -- no separate accounting to keep in sync.
12. The whole app also runs in Docker now (`docker compose up --build`) -- see "Or run it in Docker" above.
13. Asking the exact same question twice (or asking it once with existing memories, which embeds it for both
    memory lookup and retrieval) now costs one real embedding call instead of two -- see `cost.memory_embedding_
    tokens` / `cost.embedding_tokens` on the second identical `/ask`.
14. `IndexReport.pii_mode` shows whether PII was off/warned-about/redacted for that request; with
    `PII_MODE=redact`, `IndexReport.warnings` says so and the raw values never reach `/documents/{id}/chunks`.
15. A scanned page (no text layer) uploaded with `OCR_ENABLED=true` comes back with `page.status == "ocr"` and
    real transcribed text instead of `"empty"`; `ParsedDocument.metadata.ocr_pages_used`/`ocr_cost_usd` show
    what that cost, capped by `OCR_MAX_PAGES_PER_DOCUMENT`.
16. With `OTEL_ENABLED=true` (and `pip install -r requirements-otel.txt`), every `tracing.span(...)` call
    ALSO opens a real OpenTelemetry span -- printed to the console by default, or shipped to a collector via
    `OTEL_EXPORTER_OTLP_ENDPOINT`.

See `docs/PROJECT_GUIDE.md` (repo rules) and `docs/day2.md` .. `docs/day21.md` (lessons).

## CI
Push this repo to GitHub and the Actions tab will run `.github/workflows/ci.yml` automatically: a free test gate
on every push/PR, and an optional cost-bounded live eval on pushes to `main` if you add an `OPENAI_API_KEY` repo
secret. See `docs/day11.md`.
