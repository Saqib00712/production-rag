# Day 3: Retrieval - BM25 keyword, vector, hybrid (RRF)

## 1. Objective
Turn stored chunks into an answerable index: `POST /search` with `mode` = keyword | vector | hybrid, returning
page-numbered hits with scores. No LLM answer yet (Day 5). Plus `scripts/verify_day3.py`: one command that runs
every test and a live retrieval benchmark.

## 2. Why this matters
Generation can only be as good as retrieval: if the right chunk is not retrieved, no prompt can save the answer.
Keyword search misses paraphrases ("send the item back" vs "refund"); vector search can miss exact identifiers
("ERR-4471") and rare names. Production systems combine both.

## 3. Theory
**BM25 (keyword).** Ranks chunks by query-term matches, weighting each term by rarity (IDF: rare terms count more),
saturating term frequency (the 10th "refund" adds little), and normalizing by chunk length. Implemented by SQLite FTS5
(`bm25()`, negative = better, we flip the sign). Porter stemming makes "refunds" match "refund". Mistake: passing raw
user text to MATCH: operators/quotes cause syntax errors or query manipulation. We tokenize to `\w+`, drop stopwords,
quote every term, join with OR.

**Vector search (semantic).** Embed the query with the SAME model as the chunks, compare with cosine similarity
(angle between vectors; 1 = same direction). We do exact brute-force numpy search: perfect recall, no native
extension, fine up to tens of thousands of chunks. Approximate indexes (HNSW/IVF) trade recall for speed at scale
(later: Vector Database at Scale). Cosine values are model specific (text-embedding-3-small is typically 0.2-0.6 for
relevant text), so a refusal threshold must be calibrated with an eval, not guessed (Day 5-6).

**Hybrid + RRF.** BM25 scores are unbounded, cosine is in [-1,1]: never add them. RRF fuses RANKS:
score = sum 1/(60 + rank). A chunk found by both retrievers accumulates score and rises. k=60 is the standard constant.
Each retriever fetches `max(top_k*4, 20)` candidates so fusion has material to work with.

**Graceful degradation.** If the embedding API fails, hybrid returns keyword-only results with `degraded: true` and a
warning; pure vector mode fails loudly (502/503). Interview line: "Search should degrade, not disappear."

**Query embedding cache.** Repeat queries cost 0 tokens (table `query_embeddings`, bounded to 5,000 rows so
user-controlled input can not grow storage forever).

**Self-healing keyword index.** On startup the repository compares `chunks` and `chunks_fts`; if they differ (your Day 2
data predates FTS) it rebuilds. No re-index needed, no OpenAI cost.

## 4. Architecture
```
POST /search {query, mode, top_k, document_id?}
  validate (1-500 chars, top_k 1-20)
  keyword: build_match_query -> FTS5 BM25 top-N          (free, ~ms)
  vector : cache lookup -> [embed query] -> load vectors -> cosine top-N
  hybrid : both -> RRF fuse (vector failure => keyword-only, degraded=true)
  -> top_k hits {page_number, text, score, keyword_rank/score, vector_rank/score}
  -> tokens, cost_usd, timings_ms
```

## 5. Files
NEW: models/search.py, services/{search,fusion,vector_math}.py, routers/search.py, scripts/verify_day3.py,
tests/{test_search_units,test_repository_search,test_search_endpoint}.py, docs/day3.md
CHANGED: repositories/chunk_repository.py (FTS5 table, keyword_search, load_vectors, query cache), config.py,
logging_config.py, main.py, tests/fakes.py (BowEmbedder), requirements.txt (numpy), README.md

## 6-7. Code notes
- `build_match_query`: the only path from user text to FTS5. Tests throw injection strings at it.
- `SearchService`: keyword first (free), vector second (paid); hits carry both ranks so you can SEE why a chunk won.
- `BowEmbedder` (tests): synonyms share a dimension, so "money" is near "refund" with no shared word, which proves
  vector search finds what keyword search can not, without spending on OpenAI.

## 8. Run
```
pip install -r requirements.txt
python scripts/verify_day3.py            # everything, live (costs < $0.001)
python scripts/verify_day3.py --offline  # free, keyword only
uvicorn app.main:app --reload --port 8000
```
Try `POST /search` in /docs against your resume: `{"query":"which project used OCR?","mode":"hybrid"}`.

## 9. Testing
65 tests total (27 new): fusion math, cosine edge cases (zero vector, empty), query sanitizer, BM25 ranking/stemming,
re-index replaces FTS rows, FTS self-heal, document filter, query cache + prune, vector-beats-keyword, hybrid rescue,
degradation, 502/503 paths, hostile queries, validation. The verify script adds a live hit@1/hit@3 benchmark.

## 10. Failure cases
Provider down: hybrid degrades, vector 502. No key: keyword works, vector 503. Query with only stopwords: empty +
warning. Nothing indexed: empty hits, not an error. Model/dimension mismatch: 409 "re-index". FTS5 missing in Python's
SQLite: clear startup error.

## 11. Security
FTS query injection neutralized; query length capped (500 chars) and top_k capped (20) to bound cost/CPU; query
cache bounded; document_id is a validated UUID. Still open: auth, per-user rate limits (later days), and prompt
injection inside retrieved text (Day 7 guardrails) - retrieved chunks are untrusted data.

## 12. Observability
Each search logs request_id, mode, hits, duration_ms, tokens, cost_usd. Response includes timings_ms (keyword,
embed_query, vector, total). Later: Prometheus histograms for retrieval latency, cache hit rate, degraded rate.

## 13. Cost
Keyword = $0. Vector/hybrid = one query embedding (~10-20 tokens ~ $0.0000004), 0 on repeats. The verify script's
whole live run is a few hundred tokens.

## 14. Production improvements
Cache vectors in memory / ANN index at scale; per-tenant filters pushed into the index; BM25 tuning (k1, b, field
weights); stopword lists per language; query rewriting; async concurrent keyword+vector; reranking (Day 4).

## 15. Interview Q&A
- *Why hybrid?* Lexical and semantic retrieval fail on different queries; fusion covers both.
- *Why RRF instead of adding scores?* Incompatible scales; ranks are comparable and need no tuning.
- *Why brute-force vectors?* Exact baseline, zero dependencies, fine at this scale; move to ANN when measured latency demands it.
- *How do you evaluate retrieval?* Labeled questions -> hit@k / MRR / recall@k (Day 6); the verify script is the seed.
- *What if the embedding service is down?* Degrade to BM25 and flag it.
- *Is FTS MATCH safe with user input?* Not raw: tokenize, quote, OR-join.

## 16. Day-end checklist
- [ ] `python scripts/verify_day3.py` ends with ALL CHECKS PASSED
- [ ] `/search` works on your resume in all 3 modes
- [ ] you can explain BM25 vs cosine vs RRF and why we fuse ranks
- [ ] you saw a query where keyword misses and vector/hybrid finds the page
- [ ] you can explain what `degraded: true` means

## 17. Learned
BM25, FTS5, cosine similarity, exact vs approximate search, RRF, graceful degradation, injection-safe query building.

## 18. Next: Day 4
Cross-encoder reranking: re-scoring the top candidates with a stronger model (OpenAI-based, as you chose OpenAI only),
plus retrieval-quality metrics groundwork.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25 keyword search, vector search,
hybrid retrieval (RRF), graceful degradation, query cache, injection-safe search.
NOT YET: reranking, context construction, generation + page citations, insufficient-evidence refusal, hallucination
detection, retrieval/answer/citation evals (RAGAS, DeepEval), regression tests, LangGraph, OpenTelemetry/Prometheus/
Grafana/LangSmith, auth + rate limits, Docker/Compose, OCR.
PROJECT STATUS: ~38%. Ingestion, indexing and retrieval are done; reranking, generation, evals, observability, security
hardening and containerization remain.
