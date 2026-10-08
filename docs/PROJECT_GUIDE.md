# Project Guide: how to manage this repo across all days

## Rules
1. ONE repo (`production-rag`), grown every day. Never start a new folder per day.
2. Layers: `routers/` = HTTP only, `services/` = logic (no FastAPI imports), `repositories/` = database only,
   `models/` = Pydantic contracts, `dependencies.py` = wiring. Logic never goes in routers.
3. Every new feature ships with tests in `tests/`. Tests must not call OpenAI (use `tests/fakes.py`).
4. Secrets only in `.env` (git-ignored). New settings go in `config.py` AND `.env.example`.
5. `app/storage/` is DATA (uploads, sqlite). Never put code there, never commit it.

## Applying a new day's files (overlay method)
1. Extract the day zip INTO your existing `production-rag` folder; choose "overwrite / replace" on conflicts.
   The zip contains only new/changed files, so your `.env` and `app/storage/` are untouched.
2. Add any new variables from `.env.example` to your `.env` (Day 2: only `OPENAI_API_KEY` is required).
3. `pip install -r requirements.txt` then `pytest -v`.

## Git workflow (one commit + tag per day)
```bash
git init                       # first time only
git add . && git commit -m "Day 1: PDF ingestion"   # if Day 1 was never committed
git tag day-1
# after applying Day 2:
git add . && git commit -m "Day 2: chunking + embeddings" && git tag day-2
```
Check `git status` shows no `.env`, no `*.db`, no `app/storage/raw/*.json` (your resume contains personal data).

## Where future days will land (planned map)
```
app/
  routers/      documents.py (Day1-2)  search.py (D3)  chat.py (D4-5)  admin/metrics (D9+)
  services/     pdf_parser, text_cleaner, chunker, embedder (D1-2)
                keyword_search + vector_search + hybrid (D3), reranker (D4),
                context_builder + generator + citations (D5), guardrails (D7)
  repositories/ chunk_repository (D2) -> vector index / FTS5 additions (D3)
  observability/ metrics + tracing (D9+)
evals/          golden datasets, retrieval + answer + citation evals (D6-8)
docker/ + docker-compose.yml  (D10+)
```
This map is a plan; days may move, the layering rules above will not.
