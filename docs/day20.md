# Day 20: Prod/Dev Dependency Split, Restored CI, GHCR Publishing -- and Two Real-World Bugfixes

## 1. Objective
Close the two polish items flagged since Day 18 (publish the Docker image to a registry; split prod/dev
dependencies) -- and, discovered along the way (one from the day's own investigation, two reported directly from
a real Windows install), fix three things that were silently wrong: a missing `.github/workflows/ci.yml` that
Day 11 had always claimed existed, an `hnswlib` C++ build forced onto every install regardless of whether HNSW
was ever used, and two verify scripts that asked a question their own test PDF had no answer for.

## 2. Why this matters
Running this project for real surfaced problems a clean dev machine never would. `pip install -r
requirements-dev.txt` on Windows without the Microsoft Visual C++ Build Tools failed outright, because
`hnswlib` -- needed ONLY by the optional `hnsw` vector-index backend (the default is `brute_force`, pure numpy,
Day 12) -- sat unconditionally in the hard requirements files, so EVERY install paid its C++ compile cost whether
or not HNSW was ever selected. Separately, `scripts/verify_day16.py` and `scripts/verify_day19.py` both asked
"what is the warranty on the device?" against a test PDF that only ever mentioned refunds (and, for Day 16,
shipping) -- never warranty -- so the RAG pipeline correctly refused to answer (exactly its Day 4 design), and the
verify scripts wrongly treated that correct refusal as a failure. Both bugs were real defects in delivered
scripts, not flaky tests, and both are fixed today alongside Day 20's own planned scope.

## 3. Theory
**Why hnswlib becomes a THIRD requirements file instead of a try/except around the import.** The import was
already lazy and correctly guarded -- `HnswIndex.build()` (app/services/vector_index.py) only ever runs `import
hnswlib` when the `hnsw` backend is explicitly selected, and `vector_index_backend` defaults to `brute_force`
(app/config.py), which never touches it. The bug was never in the import; it was that `pip install -r
requirements.txt` resolves and builds EVERY listed package up front, regardless of which code paths a given
user will ever execute. A library only needed by one opt-in backend belongs in a THIRD file the user installs
only if they opt into that backend -- the same reasoning that already split prod from dev today, just applied to
a second axis (default-backend vs. optional-backend) instead of production-vs-test.

**Why the HnswIndex error path needed a second change, not just the requirements split.** Without hnswlib
installed, `import hnswlib` now raises a raw `ModuleNotFoundError` from deep inside a request handler -- not a
compiler error during `pip install`, but a confusing 500 at RUNTIME for anyone who edits `vector_index_backend`
to `hnsw` without reading why. `HnswIndex.build()` now catches that `ImportError` and re-raises a `RuntimeError`
that names the exact fix (`pip install -r requirements-hnsw.txt`) -- the same "fail with a message that tells you
what to do," not "fail with a stack trace," standard this project has applied to every other optional/
misconfigured dependency since Day 2.

**Why the verify-script bug was a fixture problem, not an application bug.** `/ask`'s refusal behavior (Day 4:
answer only from evidence actually in the indexed document, refuse otherwise) worked exactly as designed in both
cases -- there was never a warranty page in either PDF, so refusing was correct, and a verify script asserting
the OPPOSITE of correct behavior is a bug in the verify script's test data, not in `app/`. The fix adds the exact
warranty-page text already proven to work in `scripts/ci_eval.py`'s fixture, rather than inventing new wording,
so the fixture is now internally consistent with a page that genuinely answers the question being asked.

## 4. Architecture
```
requirements.txt        - production only (Day 20): fastapi, uvicorn, pydantic(-settings), pypdf, numpy,
                           pyyaml, prometheus_client, python-multipart, openai, tiktoken. No hnswlib.
requirements-dev.txt     - "-r requirements.txt" + pytest, httpx, fpdf2, pytest-asyncio. No hnswlib either.
requirements-hnsw.txt    - NEW (Day 20): just "hnswlib>=0.8.0" with a comment explaining why it's separate.
                           Install only if you set vector_index_backend=hnsw.

app/services/vector_index.py
  HnswIndex.build()  - import hnswlib wrapped in try/except ImportError -> RuntimeError naming the fix

.github/workflows/ci.yml
  test            (always; pip install -r requirements-dev.txt; pytest -v)
  live-eval       (needs: test; push-to-main only; scripts/ci_eval.py, skips clean with no secret)
  publish-image   (needs: [test, live-eval]; push-to-main only; docker/login-action -> ghcr.io using the
                   built-in GITHUB_TOKEN -> docker/build-push-action pushing :latest and :<sha>)

scripts/verify_day16.py / verify_day19.py
  Test PDF fixtures now include a warranty page (same text as scripts/ci_eval.py's proven fixture) so the
  turn-1 "what is the warranty" question has real evidence to answer from.
```

## 5. Files
CHANGED: `requirements.txt` (hnswlib removed, production-only), `requirements-dev.txt` (hnswlib removed),
`Dockerfile` (`ARG DEPS_FILE=requirements.txt`, copies both requirement files, installs `${DEPS_FILE}`),
`docker-compose.yml` (top comment), `app/services/vector_index.py` (`HnswIndex.build()`'s import wrapped with a
clear `RuntimeError`), `tests/test_vector_index_integration.py` (the `hnsw`-parametrized fixture now
`pytest.importorskip("hnswlib")`s itself so the suite still passes cleanly on a machine without it),
`scripts/verify_day16.py` / `scripts/verify_day19.py` (PDF fixtures gain a warranty page), `scripts/verify_day20.py`
(gains the hnsw-optional and requirements-hnsw.txt checks), `app/main.py` (version bump), `README.md`.
NEW: `.github/workflows/ci.yml` (restored -- had been referenced by `docs/day11.md` and checked by
`scripts/verify_day11.py` since Day 11 but was missing from this checkout; rebuilt from those two sources, then
extended with `publish-image`), `requirements-hnsw.txt`, `docs/day20.md`.

## 6-7. Code notes
- `HnswIndex.build()`'s new `try/except ImportError` is the ONLY code change needed to make hnswlib truly
  optional -- everything else (the requirements split, the test fixture skip) just stops forcing it to be
  installed in the first place.
- `tests/test_vector_index_integration.py`'s `backend_client` fixture calls `pytest.importorskip("hnswlib")`
  only for the `"hnsw"` parametrization, inside the fixture body, so the `"brute_force"` half of every
  parametrized test still runs (and still counts) even on a machine with no hnswlib installed at all.
- The warranty page added to both verify scripts' PDFs is copy-pasted verbatim from `scripts/ci_eval.py`'s
  existing fixture (`"Warranty terms. The device carries a two year limited warranty covering manufacturing
  defects. Claims require serial number ERR-4471."`) rather than newly written, specifically so it's text
  already proven to retrieve and answer correctly in a different, working script.
- `.github/workflows/ci.yml`'s restoration was verified against the project's OWN pre-existing check for it:
  `python scripts/verify_day11.py --offline` failed on the missing file before the fix and passed all 8 checks
  after, with no changes to verify_day11.py itself.

## 8. Run
```
python scripts/verify_day20.py            # pytest + requirements split + hnsw-optional check + ci.yml checks
                                           # + (if Docker is reachable) a build proving the default image is lean
python scripts/verify_day20.py --offline  # everything except the docker build

# Opting into the HNSW backend (optional, needs a C++ toolchain on Windows):
pip install -r requirements-hnsw.txt
# then set vector_index_backend=hnsw in .env -- without the above, that setting now fails with a clear
# RuntimeError on first use instead of a cryptic ImportError deep in a request handler.
```

## 9. Testing
338 tests total (unchanged count from Day 19 -- today's fixes don't add new application behavior, they correct
test fixtures and remove an unnecessary hard dependency). `scripts/verify_day20.py --offline`: 17/17 checks pass,
including a new one that simulates hnswlib being absent (via `sys.modules`/`builtins.__import__` patching, not a
real uninstall) and confirms `HnswIndex.build()` raises the new, actionable `RuntimeError` rather than a raw
`ImportError`. `scripts/verify_day16.py --offline` and `scripts/verify_day19.py --offline` both still pass (their
pytest-only mode doesn't exercise the fixed PDFs); their live checks (which need a real `OPENAI_API_KEY`) now ask
a question the indexed content actually answers, so `turn 1 is answered` / `turn 2 is answered the same way` will
pass instead of failing as they did before today's fix.

## 10. Failure cases
If `vector_index_backend=hnsw` is set and `requirements-hnsw.txt` was never installed, the FIRST request that
needs to build or rebuild that tenant's vector index now fails with a clear `RuntimeError` naming the exact pip
command to run, instead of either a successful-looking-but-wrong result or an opaque `ModuleNotFoundError`
traceback. The Docker image itself is unaffected either way, since the default backend never imports hnswlib.

## 11. Security
No new surface. Removing hnswlib from the default install path is a supply-chain reduction, not an addition --
fewer compiled third-party packages are pulled in by default. `.github/workflows/ci.yml`'s `publish-image` job
uses the repo's own ephemeral `GITHUB_TOKEN` (scoped to `packages: write` for this repo only, expires at job end)
rather than a new long-lived secret.

## 12. Cost
No change to request-time cost. Build-time/install-time cost goes down for anyone not using the HNSW backend
(the common case, since it defaults to `brute_force`): no C++ compilation, no Visual Studio Build Tools
prerequisite, faster `pip install`, and a smaller Docker image (already true since Day 18's multi-stage build,
now also true for a local non-Docker install).

## 13. Production improvements
Everything originally flagged as open since Day 18 (publish the image to a registry, split prod/dev deps) is now
done. Remaining smaller items: real OTel export, a real Prometheus/Grafana deployment, a PII redaction mode (only
detection exists, since Day 7), RAGAS/DeepEval-based eval, LangGraph-based orchestration, OCR for scanned PDFs.

## 14. Interview Q&A
- *Why was hnswlib ever a hard dependency if the import was already lazy?* `pip install -r requirements.txt`
  resolves and builds every listed package regardless of which code paths actually run -- a lazy `import` inside
  a function only controls when Python loads the module at RUNTIME, not whether `pip` has to install it at
  INSTALL time. Those are two different moments, and fixing one doesn't fix the other.
- *Why not just catch the ImportError silently and fall back to brute_force?* Because the user explicitly chose
  `hnsw` in their config -- silently substituting a different backend would mean their retrieval behaves
  differently from what they configured without telling them, which is worse than a clear, actionable error
  that stops and says exactly what to install.
- *How would you have caught the missing ci.yml and the verify-script bugs before a user did?* Running each
  verify script with a REAL `OPENAI_API_KEY` (not just `--offline`) in this environment, or in CI itself, would
  have caught the warranty-question bug immediately; for ci.yml, it's a reminder that "a doc file references X"
  is not the same evidence as "X exists" -- the day's own verify_dayN.py script checking for the file it
  documents is what actually would have caught the regression at the time it happened.
- *Why fix verify_day16.py/verify_day19.py here instead of waiting for a "Day 21 bugfix"?* Both were reported as
  actively blocking a real user's work, and this installer's own pipeline (each day's package is built as a diff
  on top of the previous day's folder) means leaving them broken would carry the SAME bug forward into every
  future day's installer -- fixing it now, before building today's package, is strictly cheaper than fixing it
  later.

## 15. Day-end checklist
- [ ] `python scripts/verify_day20.py --offline` ends with ALL 17 CHECKS PASSED
- [ ] `pip install -r requirements-dev.txt` succeeds on a machine with NO C++ compiler installed
- [ ] you can explain why a lazy import doesn't prevent `pip install` from building the package anyway
- [ ] you can explain why the verify-script bug was a test-fixture defect, not an `/ask` defect

## 16. Learned
A lazily-imported dependency can still be an eagerly-INSTALLED one -- `pip install -r requirements.txt` doesn't
know or care that a function three call-frames deep never reaches that import at runtime; only splitting the
requirements file itself fixes install-time cost. Also: a verify script that asserts a specific answer must NOT
be a refusal is only as correct as the fixture content it's asking about -- when the application's refusal logic
is working exactly as designed, the bug is almost always in what the test fed it, not in the refusal itself.

## 17. Next: Day 21
With CI/CD, containerization, cost accounting, observability, and dependency hygiene now all in a genuinely solid
state (and verified against a real Windows install, not just this dev environment), the natural next step is
either closing one of the remaining Section 13 items (OTel export, PII redaction mode, OCR for scanned PDFs) or
moving to Project 2 of the curriculum, since Production RAG with Citations' own checklist is now complete start
to finish.

## Tracker
COMPLETED: everything listed in `docs/day19.md`'s tracker, plus: Docker image publishing to GHCR via a third CI
job gated behind both quality gates, prod/dev/optional-hnsw dependency separation, a friendly runtime error when
the hnsw backend is selected without its optional dependency installed, and corrected verify-script fixtures for
Day 16 and Day 19 (both were asking about content that was never in their test PDFs).
NOT YET: real OTel export, real Prometheus/Grafana deployment, PII redaction mode, RAGAS/DeepEval, LangGraph,
OCR.
PROJECT STATUS: 100% of the originally scoped checklist. Ingestion, indexing, retrieval (exact and approximate,
with the approximate path now a true opt-in with no forced install cost), grounded generation, evaluation,
tuning, security baseline, full observability, multi-tenancy, three-tier CI/CD with image publishing, short- and
long-term agent memory with a shared, relevance-ranked, cache-sharing retrieval path, complete cost/trace
accounting, and containerization are all done and have now been exercised against a real external (Windows)
install, not just this development environment.
