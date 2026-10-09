# Production RAG with Citations

A fully production-grade Retrieval-Augmented Generation system built over 21 days — covering hybrid search, LLM reranking, grounded generation with citations, streaming, multi-tenancy, observability, security guardrails, agent memory, containerization, and more. Built entirely on the OpenAI API stack.

---

## What This Project Does

Most RAG tutorials stop at "embed → retrieve → generate." This project builds every layer a real production system needs — from chunking strategy and hybrid search all the way to PII redaction, multi-tenant auth, CI/CD pipelines, Docker deployment, and a full Streamlit monitoring interface.

```
PDF Upload
    ↓
Clean + Page-aware Chunking + OpenAI Embeddings + SQLite Storage
    ↓
Hybrid Search — BM25 Keyword + Vector + RRF Fusion
    ↓
LLM Reranker — scores chunks for true relevance
    ↓
Grounded Generation — answers with inline citations
    ↓
Streaming SSE response — token by token
    ↓
Multi-tenant isolation — per-tenant keys, budgets, rate limits
    ↓
Full observability — request tracing, Prometheus metrics, live dashboard
```

---

## Built Day by Day — 21 Days

| Day | Feature | Tests |
|-----|---------|-------|
| 1 | Project scaffold, PDF ingestion, basic RAG pipeline | — |
| 2 | Page-aware chunking, OpenAI embeddings, SQLite storage, cost tracking | — |
| 3 | BM25 keyword + vector + hybrid RRF search, /search endpoint | 65 |
| 4 | LLM reranking, grounded generation, citation validation, refusal taxonomy, /ask endpoint | 99 |
| 5 | Golden eval sets, retrieval metrics, run_eval.py | — |
| 6 | Chunk size / retrieve_k / rerank threshold tuning via tune.py | 142 |
| 7 | Rate limiting, audit logging, PII-aware ingestion, prompt injection test suite | 174 |
| 8 | Streaming /ask via SSE, correction event on ungrounded streamed answers | 189 |
| 9 | Unified request tracing via contextvars, Prometheus metrics, live /observability/dashboard | 216 |
| 10 | API key auth, multi-tenant document isolation, per-tenant rate limits + daily cost budgets | 231 |
| 11 | GitHub Actions CI/CD — test gate + cost-bounded live-eval gate on main | 235 |
| 12 | Pluggable ANN indexing via hnswlib, per-tenant index caching with invalidation | 256 |
| 13 | Short-term agent memory, conversation threads, query contextualization | 277 |
| 14 | Long-term cross-conversation memory — fact extraction, tenant-scoped storage | 303 |
| 15 | Vector-indexed memory retrieval — cosine similarity instead of flat injection | 315 |
| 16 | Full cost accounting — contextualizer, memory extraction, memory embedding | 332 |
| 17 | Cost breakdown in observability dashboard and eval reports | 336 |
| 18 | Docker + docker-compose, persistent storage volume, /health healthcheck | 336 |
| 19 | Question-embedding cache for memory retrieval | 338 |
| 20 | Prod/dev dependency split, GHCR Docker image publish via CI, Windows fixes | 338 |
| 21 | Optional OpenTelemetry export, PII redaction mode, vision-based OCR for scanned PDFs | 361 |

**Final: 361 tests — 100% of scoped checklist complete.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION RAG SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│  Interfaces                                                 │
│  ├── FastAPI REST API  (upload, /ask, /ask/stream, /search) │
│  ├── Browser Console   (upload, ask, search, chunks)        │
│  └── Streamlit App     (full pipeline trace + monitoring)   │
│                                                             │
│  Core Pipeline                                              │
│  ├── PDF Ingestion → Page-aware Chunking → Embeddings       │
│  ├── Hybrid Search: BM25 + Vector + RRF Fusion              │
│  ├── LLM Reranker → Grounded Generation → Citation Check    │
│  └── Streaming SSE with correction event                    │
│                                                             │
│  Memory Layer                                               │
│  ├── Short-term: conversation threads + query rewriting     │
│  └── Long-term: fact extraction + vector-indexed retrieval  │
│                                                             │
│  Security & Multi-tenancy                                   │
│  ├── API key auth + per-tenant document isolation           │
│  ├── Per-tenant rate limits + daily cost budgets            │
│  ├── PII detection + optional redaction ([REDACTED_*])      │
│  └── Prompt injection test suite                            │
│                                                             │
│  Observability                                              │
│  ├── Unified request tracing (contextvars + trace spans)    │
│  ├── Prometheus metrics at /metrics                         │
│  ├── Live dashboard at /observability/dashboard             │
│  └── Optional OpenTelemetry export                          │
│                                                             │
│  Infrastructure                                             │
│  ├── Docker + docker-compose + persistent storage volume    │
│  ├── GitHub Actions: test gate + live-eval + GHCR publish   │
│  ├── Pluggable ANN: hnswlib or brute force                  │
│  └── Optional vision OCR for scanned PDFs                  │
│                                                             │
│  Storage: SQLite — chunks, conversations, memory, budgets   │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

### Search & Retrieval
- **BM25 keyword search** — exact term matching with TF-IDF scoring
- **Vector search** — OpenAI embeddings with cosine similarity
- **Hybrid RRF fusion** — Reciprocal Rank Fusion combines both signals into one ranked list
- **LLM reranker** — second-pass LLM scoring for true relevance beyond embedding similarity
- **Pluggable ANN indexing** — hnswlib for scale, brute force as zero-dependency default
- **Query-embedding cache** — reuses embeddings within and across requests to cut costs

### Generation & Citations
- **Grounded generation** — answers only from retrieved context, never from model memory
- **Inline citations** — every claim linked to source document + page number
- **Refusal taxonomy** — structured refusal with reason code when question is unanswerable
- **Citation validator** — verifies every cited chunk actually exists before returning answer
- **Streaming SSE** — token-by-token streaming with a correction event on ungrounded answers

### Memory
- **Short-term memory** — conversation threads with query contextualization (rewrites follow-up questions like "what about its price?" into standalone search queries)
- **Long-term memory** — cross-conversation fact extraction, tenant-scoped, vector-indexed retrieval via cosine similarity so only the most relevant facts enter each answer
- **Fail-open** — flat memory injection fallback if vector retrieval fails

### Security & Multi-tenancy
- **API key auth** — per-tenant key management via scripts/manage_keys.py
- **Tenant isolation** — complete document and chunk separation per tenant at the DB level
- **Rate limiting** — per-tenant request throttling
- **Audit logging** — every request logged with tenant, cost, outcome
- **PII-aware ingestion** — detect PII before chunking; optional redaction replaces with [REDACTED_*] markers
- **Prompt injection test suite** — validates guardrails hold against known attacks

### Observability & Cost
- **Request tracing** — unified trace ID across all pipeline stages with named spans
- **Prometheus metrics** — scrapable at /metrics
- **Live dashboard** — at /observability/dashboard
- **Full cost accounting** — embeddings, reranking, generation, contextualizer, memory extraction all tracked per request and per tenant daily budget
- **Cost-bounded CI** — live-eval gate on main enforces a spending ceiling before deploying
- **Optional OTel export** — send traces to any OpenTelemetry collector

### Infrastructure
- **Docker + docker-compose** — multi-stage build, persistent storage volume, /health healthcheck
- **GitHub Actions CI/CD** — test gate on every PR + cost-bounded live-eval gate on main + Docker image published to GHCR
- **Prod/dev dependency split** — requirements.txt vs requirements-dev.txt
- **Optional OCR** — vision-based OCR for scanned PDFs via OpenAI vision model, capped per document

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20%2B%20Embeddings-black?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-Storage-orange?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Container-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Monitoring-red?style=flat-square)

- **FastAPI** — async REST API with SSE streaming and CORS
- **OpenAI** — GPT-4o for generation and reranking, text-embedding-3-small for embeddings, vision model for OCR
- **SQLite + SQLAlchemy** — chunks, conversations, memory, budgets, audit logs
- **rank_bm25** — BM25 keyword search
- **hnswlib** — optional ANN vector indexing at scale
- **Prometheus** — metrics export
- **Docker + GitHub Actions** — containerization and CI/CD with GHCR image publish
- **Streamlit** — monitoring interface and full pipeline trace viewer

---

## Project Structure

```
production-rag/
│
├── app/
│   ├── main.py                      # FastAPI app + CORS + routes
│   ├── config.py                    # Settings, env vars, CORS origins
│   ├── models/                      # SQLAlchemy ORM models
│   ├── repositories/
│   │   ├── chunk_repository.py      # Vector + BM25 + hybrid + cache
│   │   └── memory_repository.py     # Fact storage + vector retrieval
│   ├── services/
│   │   ├── ingestion.py             # PDF → clean → chunk → embed → store
│   │   ├── search.py                # BM25 + vector + RRF fusion
│   │   ├── reranker.py              # LLM reranking
│   │   ├── generator.py             # Grounded generation + citations
│   │   ├── contextualizer.py        # Query rewriting for follow-ups
│   │   ├── memory_extractor.py      # Long-term fact extraction
│   │   ├── ocr.py                   # Vision OCR (optional)
│   │   └── pii.py                   # PII detection + redaction
│   ├── observability/
│   │   ├── tracing.py               # Unified request tracing
│   │   ├── metrics.py               # Prometheus metrics
│   │   └── otel_export.py           # OpenTelemetry export (optional)
│   └── eval/
│       └── cost_gate.py             # CI cost ceiling enforcement
│
├── scripts/
│   ├── tune.py                      # Chunk size / retrieve_k tuning sweep
│   ├── run_eval.py                  # Golden eval + cost breakdown reports
│   ├── manage_keys.py               # Multi-tenant key management
│   ├── bench_vector_index.py        # ANN indexing benchmark
│   └── ci_eval.py                   # CI live-eval gate
│
├── frontend/
│   ├── index.html                   # Browser console UI
│   └── streamlit_app.py             # Streamlit monitoring interface
│
├── tests/                           # 361 tests across all 21 days
├── Dockerfile                       # Multi-stage production build
├── docker-compose.yml               # Persistent storage + healthcheck
├── .github/workflows/ci.yml         # Test + eval + GHCR publish
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Dev/test dependencies
├── requirements-hnsw.txt            # Optional: hnswlib ANN indexing
├── requirements-otel.txt            # Optional: OpenTelemetry export
├── requirements-ocr.txt             # Optional: vision OCR (PyMuPDF)
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/production-rag.git
cd production-rag
```

### 2. Create environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the API
```bash
uvicorn app.main:app --reload
```

### 6. Open browser console
```
http://localhost:8000
```

### 7. Run Streamlit monitoring interface
```bash
streamlit run frontend/streamlit_app.py
```

### 8. Or run everything with Docker
```bash
docker-compose up
```

---

## Optional Features

| Feature | Install | Enable via .env |
|---------|---------|-----------------|
| ANN indexing | `pip install -r requirements-hnsw.txt` | `VECTOR_INDEX_BACKEND=hnsw` |
| OpenTelemetry export | `pip install -r requirements-otel.txt` | `OTEL_ENDPOINT=your_endpoint` |
| Vision OCR for scanned PDFs | `pip install -r requirements-ocr.txt` | `OCR_ENABLED=true` |
| PII redaction | built-in | `PII_MODE=redact` |

All optional features are **fail-open** — the system continues without them if the dependency is not installed.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload and index a PDF |
| POST | `/ask` | Grounded answer with inline citations |
| POST | `/ask/stream` | Streaming SSE answer with correction event |
| GET | `/search` | Hybrid BM25 + vector search |
| GET | `/chunks` | Inspect indexed chunks |
| POST | `/conversations` | Create conversation thread |
| GET | `/memories` | List long-term memory facts |
| DELETE | `/memories/{id}` | Delete a memory fact |
| GET | `/metrics` | Prometheus metrics |
| GET | `/observability/dashboard` | Live monitoring dashboard |
| GET | `/observability/summary` | Cost breakdown by stage |
| GET | `/health` | Health check |

---

## Key Concepts Covered

- **Page-aware chunking** — chunks respect page boundaries so every citation points to a real page number
- **Hybrid RRF fusion** — combining sparse BM25 and dense vector scores via Reciprocal Rank Fusion
- **LLM reranking** — second-pass scoring that consistently outperforms embedding similarity alone
- **Citation validation** — verifying every cited chunk exists in the DB before returning the answer
- **Refusal taxonomy** — structured "I don't know" with reason code instead of hallucinating
- **SSE streaming with correction** — stream early; emit a correction event if answer later proves ungrounded
- **Query contextualization** — rewriting "what about its price?" into "what is the price of X?" using conversation history
- **Vector-indexed memory** — embedding facts and retrieving only the most relevant per question, not injecting everything
- **Multi-tenant isolation** — complete document and chunk separation per API key at the DB level
- **Cost accounting** — every LLM/embedding call tracked by stage, per request, per tenant daily budget
- **Pluggable ANN** — swap between brute-force and hnswlib without changing application code
- **Fail-open design** — every optional component degrades gracefully if its dependency is missing

---

## Related Certifications

Built applying skills from the IBM **RAG for Generative AI Applications Specialization**, **Advanced RAG with Vector Databases and Retrievers**, and **Building AI Agents and Agentic Workflows Specialization** on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-RAG%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-Advanced%20RAG-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)
