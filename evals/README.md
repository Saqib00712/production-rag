# Golden evaluation sets

Each file is a JSON array of questions:
```json
{"id": "q1", "question": "...", "expected_pages": [1], "expected_keywords": ["word1", "word2"]}
```
- `expected_pages`: page(s) that should be cited in a correct answer. **Empty array = the question is
  intentionally unanswerable** — the system should refuse, and refusing counts as correct.
- `expected_keywords`: optional, a weak lexical sanity check (substring match on the answer). Not semantic —
  a real evaluation framework (RAGAS/DeepEval) is the upgrade path once you need that.

`sample_8page.json` matches the synthetic 8-page document `scripts/verify_day3.py`/`verify_day4.py` build. Copy
this file as a template and write one for your own resume or documents: pick real questions, note the real page,
and include a couple of unanswerable ones — that is the check most tutorials skip.

Reports from `scripts/run_eval.py` are written to `evals/reports/`.
