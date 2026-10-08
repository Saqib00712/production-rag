# Day 11: CI/CD for AI Systems

## 1. Objective
A GitHub Actions pipeline with two deliberately different tiers: a $0, no-API-key, always-on test gate (every push,
every PR), and an optional, cost-bounded live eval gate (pushes to `main` only, hard-capped spend). Plus a local
pre-push hook for an even earlier signal, and `app/eval/cost_gate.py` as the single, tested chokepoint that
prevents any automated run from becoming an uncapped bill.

## 2. Why this matters
Every previous day ended with "run this script yourself." That's fine solo, and a real liability the moment more
than one person touches the code, or you forget to run it yourself before a bad commit ships. CI/CD turns "did I
remember to test this" into a property of the repository, not a property of your memory. For an AI system
specifically, the extra wrinkle is cost: a CI pipeline that calls a paid API on every commit, unbounded, is a
production incident waiting to happen - today's design treats that as a first-class constraint, not an
afterthought.

## 3. Theory
**Two tiers, not one pipeline.** The 235-test pytest suite needs zero API keys and costs $0 - it should run on
EVERY push and EVERY pull request, full stop, including from a fork with no secrets configured. A live eval against
the real OpenAI API costs real money on every run - running it on every PR (including ones from strangers on a
public repo) is a spend an attacker could trivially abuse by opening PRs. So it's gated to pushes to `main` only,
and even then, hard-capped. This mirrors a real distinction production teams make: fast, free, deterministic
checks run unconditionally; slow, paid, real-dependency checks run more conservatively.

**Why `live-eval` doesn't fail when no secret is configured.** `scripts/ci_eval.py` checks for a real key and exits
0 with an explanation if there isn't one, rather than failing. This means forking the repo, or a contributor
without access to the secret, never sees a red X they can't fix - the ABSENCE of a paid check is not a failure,
only a FAILED paid check is.

**The cost ceiling is a tested, standalone function, not inline script logic.** `app/eval/cost_gate.py`'s
`check_cost_ceiling()` is the one place that can stop an automated run from overspending, regardless of what
upstream code produced the cost - a bug in chunking, an accidentally huge golden set, a retry storm. Making it a
small, pure, directly-tested function (not a few lines buried in a script) means the safety property itself has
a test, not just the script that happens to use it.

**Why `ci_eval.py` builds its own synthetic document instead of needing a committed fixture file.** CI environments
shouldn't depend on binary test fixtures living in the repo if they can be generated deterministically in code -
same reasoning as every `verify_dayN.py` script in this project. It reuses the exact 8-page document and golden
set already used throughout Days 3-9, so a CI failure here means something in the core pipeline broke, not that a
CI-only fixture drifted from what's actually tested locally.

**Local pre-push hook: a convenience, not a second source of truth.** `scripts/install_git_hooks.py` runs the same
`pytest -q` locally, before a push leaves your machine - purely to save the round trip of pushing, waiting for
Actions, and seeing a failure you could have caught in three seconds locally. It is bypassable (`git push
--no-verify`) and is NOT where correctness is enforced - GitHub Actions remains the authoritative gate, since a
local hook can be skipped, forgotten to install, or run on a machine with stale dependencies.

**What CI validates about itself.** `verify_day11.py`'s step 2 doesn't just check that `ci.yml` exists - it parses
it as YAML and checks the actual job structure and conditions (does `live-eval` really only run on `main`?). A
workflow file with a typo that silently never runs its gate is a worse failure mode than a workflow that's simply
missing, because it LOOKS like protection without providing any.

## 4. Architecture
```
git push  ->  (optional local pre-push hook: pytest -q, bypassable)
           ->  GitHub Actions:
                 job: test        (always, no secrets needed)
                   checkout -> setup Python (pip cache) -> pip install -r requirements.txt -> pytest -v
                 job: live-eval   (needs: test; only on push to main)
                   if no OPENAI_API_KEY secret -> exit 0, explain why, job shows green
                   else -> scripts/ci_eval.py --max-cost-usd 0.05
                             build synthetic 8-page doc -> index -> run_eval() against sample_8page.json
                             check_cost_ceiling()        -> FAIL if spend > ceiling
                             hit@3 / answer_accuracy     -> FAIL if below thresholds
```

## 5. Files
NEW: .github/workflows/ci.yml, app/eval/cost_gate.py, scripts/{ci_eval,install_git_hooks,verify_day11}.py,
tests/test_cost_gate.py, docs/day11.md
CHANGED: requirements.txt (pyyaml made an explicit, not merely transitive, dependency - verify_day11.py parses
YAML directly)

## 6-7. Code notes
- `cost_gate.py`: `>` not `>=` for the ceiling comparison - spending EXACTLY the ceiling is allowed, which matters
  at the boundary (a `0.05` ceiling with a `0.05` actual cost should pass, not fail on a rounding coincidence).
- `ci_eval.py`: mirrors `verify_day9.py`'s live-check pattern almost exactly (temp storage, synthetic fixture,
  real pipeline calls) - reusing a proven pattern rather than inventing a new one for CI specifically.
- `ci.yml`: `needs: test` on the `live-eval` job means a broken test suite short-circuits the whole pipeline before
  any money is spent - the free gate always runs first and the paid gate never runs if it already failed.
- `install_git_hooks.py`: the hook script itself is a POSIX shell script (`#!/bin/sh`) even though the installer
  is Python - Git for Windows ships Git Bash specifically to execute hooks like this, so the same hook file works
  across macOS, Linux, and Windows without a separate Windows-specific version.

## 8. Run
```
pip install -r requirements.txt
python scripts/verify_day11.py            # pytest + CI config validation + a real cost-bounded eval
python scripts/verify_day11.py --offline  # pytest + CI config validation only
python scripts/install_git_hooks.py       # optional: local pre-push test gate
```
To actually activate the pipeline: push this repo to GitHub, then add a repo secret named `OPENAI_API_KEY`
(Settings -> Secrets and variables -> Actions) if you want the `live-eval` job to run for real on pushes to `main`.
Without the secret, `test` still runs and protects every push and PR for free.

## 9. Testing
235 tests total (4 new): the cost ceiling's boundary behavior (under, exactly-at, over, zero). Everything else
Day 11 adds is infrastructure (a workflow file, two scripts) validated by `verify_day11.py` rather than unit
tests, which is itself a deliberate choice: CI config correctness is checked by actually parsing and inspecting
the YAML, the same principle as testing code instead of eyeballing it.

## 10. Failure cases
A PR from a fork with no access to secrets: `test` runs and gates the PR normally; `live-eval` doesn't run at all
(wrong branch), so nothing breaks. A main-branch push with a temporarily invalid/expired API key: `ci_eval.py`'s
underlying calls raise the same `EmbeddingConfigError`/`LLMError` already handled elsewhere in this project,
surfacing as a clear CI failure rather than a silent skip (only a MISSING key skips cleanly - an invalid one still
fails loudly, which is correct: a configured-but-broken key is worth knowing about).

## 11. Security
`OPENAI_API_KEY` lives only as a GitHub Actions secret, never in the repo or workflow file - standard practice,
and GitHub masks secret values in logs automatically. The live-eval job's branch restriction (`main` only) is
itself a security control against spend abuse via PRs from untrusted forks, not just a cost optimization.

## 12. Observability
CI failures are visible as a red X on the commit/PR in GitHub's UI, with full `pytest -v` output and
`ci_eval.py`'s printed cost/accuracy breakdown in the job log - the existing verify-script print statements
(by design, throughout this project) double as CI log output with no changes needed.

## 14. Production improvements
Add a status badge to the README once the repo is on GitHub (the badge URL needs your actual `OWNER/REPO`, which
is why it isn't hardcoded here); add a lint/type-check job (ruff/mypy) as a third, equally free tier; cache the
`tiktoken` encoding download in CI if network restrictions ever make it flaky (today's fallback to approximate
counting already prevents this from breaking anything, which is itself worth noting as a resilience property
built back on Day 2, paying off here); gate merges to `main` on required status checks via GitHub branch protection
settings (a repo setting, not a file this project can commit).

## 15. Interview Q&A
- *How do you design CI for a system with paid API dependencies?* Split into a free, always-on tier (deterministic
  tests, fakes, no network) that gates every change, and a paid, cost-bounded tier that runs less often (e.g. only
  on merges to main) with a hard spend ceiling enforced in code, not just hoped for.
- *How do you prevent a CI pipeline from running up an unbounded bill?* A single, tested function that checks
  actual spend against a ceiling after every live run and fails the job if exceeded - not a convention or a
  comment, an enforced check.
- *Why have both a local pre-push hook and GitHub Actions?* The hook is a fast, bypassable, local convenience;
  Actions is the actual, unbypassable gate — redundancy in favor of the developer's feedback loop, not a
  replacement for the real gate.
- *What should a CI job do when a secret it depends on is missing?* Depends on context - here, skip cleanly and
  exit 0 with an explanation, because the free primary gate already protects the repo and a missing OPTIONAL
  secret (e.g. on a fork) shouldn't block anyone.

## 16. Day-end checklist
- [ ] `python scripts/verify_day11.py` ends with ALL CHECKS PASSED
- [ ] the repo is pushed to GitHub and the Actions tab shows the `test` job running
- [ ] (optional) an `OPENAI_API_KEY` secret is added and a push to `main` shows `live-eval` running too
- [ ] you can explain why `live-eval` is restricted to `main` and gated behind a secret check
- [ ] you can explain what `check_cost_ceiling` protects against that a code review alone would not

## 17. Learned
Two-tier CI design for cost-sensitive systems, enforcing a spend ceiling as a tested function rather than a
convention, branch-gating paid operations for both cost and security reasons, and local pre-push hooks as a
convenience layered in front of (never instead of) a real CI gate.

## 18. Next: Day 12
Vector database at scale: moving beyond the brute-force numpy cosine search (fine at this project's scale, a real
limit past tens of thousands of chunks) into an actual ANN index, what recall/latency trade-off that introduces,
and how to evaluate whether an upgrade was actually worth it using the Day 5/6 eval tooling already in place.

## Tracker
COMPLETED: project structure, config, logging, PDF extraction, page numbers, upload security, cleaning, tokens,
chunking, embeddings, embedding cache, cost tracking/guards, SQLite storage, BM25, vector search, hybrid (RRF),
graceful degradation, query cache, injection-safe search, LLM reranking, grounded generation, citation validation,
refusal taxonomy, prompt-injection mitigation + test suite, shared client lifecycle, golden datasets, retrieval
metrics, citation-based answer correctness, LLM-as-judge, eval CLI + reports, data-driven tuning, offline
retrieval regression testing, rate limiting, audit logging, PII-aware ingestion, streaming responses (SSE),
unified request tracing, Prometheus metrics, live observability dashboard, API key auth, multi-tenant data
isolation, per-tenant rate limits + persisted daily cost budgets, key management CLI, two-tier CI/CD with a
tested cost ceiling, local pre-push test hook.
NOT YET: ANN vector indexing at scale, real OTel export, real Prometheus/Grafana deployment, PII redaction mode,
RAGAS/DeepEval, LangGraph, Docker/Compose, OCR.
PROJECT STATUS: ~93%. Ingestion, indexing, retrieval, grounded generation, evaluation, tuning, security baseline,
observability, multi-tenancy, and CI/CD are done; vector-store scaling and containerization remain as the major
pieces.
