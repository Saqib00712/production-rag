# Day 2: Cleaning, Chunking, Embeddings, Cost Tracking

## 1. Objective
Turn Day 1's per-page text into embedded, page-tagged chunks stored in SQLite, with token/cost reporting and a
free dry-run mode. No search yet (Day 3).

## 2. Why this matters
Retrieval quality is capped by chunk quality. Bad chunks (cut mid-sentence, no page tag, noisy text) can't be fixed
by a better model later. Also, this is the first paid API call: cost control must exist from the first call.

## 3. Theory
**Text cleaning.** Definition: deterministic normalization of extracted text. Why: PDFs leak layout artifacts
(`latency\nmonitoring`, `infor-\nmation`, ligatures, control chars) that pollute embeddings. How: unicode NFKC,
strip control chars, fix hyphen wraps, join lowercase-continuation lines. Mistake: over-cleaning (joining every
newline destroys headings/lists); we only join when the previous line has no sentence-ending punctuation and the next
starts lowercase. Known limit: `well-\nknown` becomes `wellknown`. Interview: "Cleaning is cheap, deterministic and
testable; I do it before the expensive step."

**Tokens.** Models see tokens, not characters (~4 chars per token in English). Embedding limits and billing are
in tokens, so chunk size is measured in tokens (tiktoken, cl100k_base for text-embedding-3-*). Mistake: sizing by
characters or words.

**Chunking.** Why: an embedding is one vector per text; a whole page blurs many topics, a tiny fragment loses
context. Our strategy: sentence-packed chunks of ~350 tokens, 50 token overlap, never crossing pages. Trade-offs:
bigger chunks = more context, less precise retrieval, more tokens sent to the LLM later; smaller = precise but
fragmented. Overlap protects answers that straddle a boundary but costs ~15% extra tokens. Never crossing pages
means each chunk has exactly one citation page. Interview: "Chunk size is a hyperparameter I tune with a retrieval
eval (Day 6+), not a guess."

**Embeddings.** A vector representing meaning; similar meaning = nearby vectors. Model: text-embedding-3-small
(1536 dims, cheap, strong baseline). Same model must embed documents AND queries. Mistake: mixing models or not
recording which model produced a vector (we key vectors by model).

**Embedding cache.** Vectors keyed by (sha256(text), model). Same text again = zero API calls. Re-uploads and
re-indexing are free.

**Reliability.** SDK retries 429/5xx with exponential backoff and a 30s timeout; failures become `EmbeddingError`
(HTTP 502). Storage is atomic: nothing is saved unless every embedding succeeded.

## 4. Architecture
```
POST /documents/{id}/index[?dry_run=true]
  load ParsedDocument JSON (id validated as UUID)
    -> clean_text per page (skip EMPTY/error pages, warn)
    -> Chunker: sentences -> token-packed chunks + overlap  (page_number kept)
    -> guards: max chunks, max tokens
    -> SQLite lookup: embeddings by (hash, model)  -> cache hits
    -> [not dry_run] OpenAI embeddings for misses only (batched)
    -> atomic save: chunks + new embeddings
    -> IndexReport {chunks, cache hits, tokens, cost_usd}
GET /documents/{id}/chunks  -> inspect chunks, page_number, has_embedding
```

## 5. Files (NEW / CHANGED)
NEW: dependencies.py, models/chunk.py, services/{text_cleaner,chunker,embedder,cost,ingestion}.py,
repositories/chunk_repository.py, tests/{fakes,test_text_cleaner,test_chunker,test_cost,test_repository,test_index_endpoint}.py, docs/
CHANGED: config.py, logging_config.py, routers/documents.py, main.py, tests/conftest.py,
tests/test_documents_endpoint.py, requirements.txt, .env.example, .gitignore, README.md

## 6-7. Code notes
- `chunker.py`: units = sentences/lines with a separator; greedy pack to `chunk_size`; step back over trailing units
  up to `overlap` tokens; always advance at least one unit (guarantees termination even for a 10,000-char "word").
- `embedder.py`: `Embedder` Protocol, so tests inject `HashEmbedder`; the pipeline never imports the OpenAI SDK.
- `ingestion.py`: order is cheap-to-expensive: clean, chunk, guards, cache lookup, then (only then) API call.
- `dependencies.py`: embedder is a FACTORY, so dry runs and fully cached re-indexes need no API key.
- `chunk_repository.py`: one short-lived connection per call (thread-safe under FastAPI's threadpool).

## 8. Run
```bash
pip install -r requirements.txt
# put OPENAI_API_KEY in .env
pytest -v
uvicorn app.main:app --reload --port 8000
```
Then in /docs: upload PDF -> `index?dry_run=true` -> `index` -> `chunks`.

## 9. Testing
38 tests, zero OpenAI calls: cleaner (5), chunker incl. overlap/oversize/termination (8), cost, repository,
endpoint flows (dry run writes nothing, cache makes re-index free, 502 saves nothing, 413 budget guard, 503 no key).
Real-API check (manual, ~$0.00002 for a 1-page resume): run `index` once and confirm `has_embedding: true`, `embedding_dim: 1536`.

## 10. Failure cases
OpenAI down/rate-limited: retries, then 502, nothing saved, safe to retry. Missing key: 503, dry run still works.
Scanned page: skipped + warning. Huge PDF: 413 before any spend. Duplicate chunk text: embedded once.
Provider returns wrong vector count: rejected. tiktoken can't load its encoding file (offline first run): falls
back to ~4 chars/token and reports a warning.

## 11. Security
UUID path params (no path traversal); token/chunk caps stop a large upload from burning your budget (cost-abuse);
API key only from env, never logged (only exception type is logged); document text with special tokens such as
`<|endoftext|>` is treated as plain text. Still missing: auth and per-user quotas (later days).

## 12. Observability
Every index call logs request_id, document_id, duration_ms, chunks, tokens, cost_usd. `IndexReport` returns the
same numbers to the caller. Later: Prometheus counters/histograms for tokens, cost, cache hit rate.

## 13. Cost
text-embedding-3-small ~ $0.02 per 1M tokens (verify on OpenAI's pricing page; configurable in `.env`).
A 1-page resume ~ 1k tokens ~ $0.00002. A 300-page book ~ 150k tokens ~ $0.003. Controls: dry run, cache,
dedupe, `MAX_INDEX_TOKENS`. Never embed at upload time automatically.

## 14. Production improvements
Async job queue for big documents; concurrent batches with rate-limit awareness; store vectors in a real vector
index (Day 3); OCR for empty pages; per-tenant budgets; embedding model versioning/migration job.

## 15. Interview Q&A
- *Why token-based chunking?* Limits and billing are in tokens; characters vary by language and content.
- *Why not embed on upload?* Cost and failure isolation: parsing is cheap and reliable, embedding is paid and can
  fail; separate steps allow dry runs, retries and budgets.
- *How do you avoid paying twice?* Content-hash + model keyed embedding cache.
- *What if the provider fails halfway?* Atomic save: all-or-nothing, retry is safe and cheap because of the cache.
- *Chunk size?* Start 300-500 tokens, ~10-15% overlap, then tune against a retrieval eval.

## 16. Day-end checklist
- [ ] `pytest -v`: 38 passed
- [ ] `.env` has real `OPENAI_API_KEY`
- [ ] dry run shows chunks/tokens/cost, nothing saved
- [ ] real index: `has_embedding: true`, dim 1536 in `/chunks`
- [ ] second index call: `chunks_from_cache` = chunk_count, tokens 0
- [ ] you can explain overlap, why chunks don't cross pages, and the cache key

## 17. Learned
Cleaning, tokens, chunking trade-offs, embeddings, content-hash caching, cost guards, atomic writes, DI for testability.

## 18. Next: Day 3
Search: SQLite FTS5 keyword search (BM25), vector similarity search over stored embeddings, and hybrid retrieval
with Reciprocal Rank Fusion.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning,
tokens, chunking, embeddings (OpenAI), embedding cache, cost tracking, cost guards, SQLite storage.
NOT YET: vector search, BM25/keyword, hybrid, reranking, context building, generation + citations, insufficient-
evidence refusal, RAGAS/DeepEval, LangGraph, OpenTelemetry/Prometheus/Grafana/LangSmith, auth/rate limits, Docker.
PROJECT STATUS: ~22% (ingestion + indexing done; retrieval, generation, evals, observability, deploy remain).
