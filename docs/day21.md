# Day 21: Optional Real OTel Export, PII Redaction, and Vision OCR for Scanned PDFs

## 1. Objective
Close the three remaining "not yet" items that have sat in every tracker since Day 19: real OpenTelemetry
export, a PII redaction mode (Day 7 only ever detected and warned), and OCR for scanned PDFs (flagged as
deferred since Day 1). All three ship together today, each strictly opt-in and fail-open, following the exact
pattern Day 20 established for `hnswlib`: an optional capability belongs in its own requirements file, never in
the hard dependencies, and its absence must degrade gracefully, never crash a request.

## 2. Why this matters
These three features share one deployment reality: they're each valuable to SOME production deployments and
irrelevant (or actively costly) to others. A team with no OTel collector doesn't want OpenTelemetry's dependency
tree. A document-Q&A system over resumes WANTS emails visible, not redacted -- a legal/compliance deployment over
contracts may want the opposite. OCR is the most expensive single operation in the whole pipeline (a
vision-capable chat completion per page vs. fractions of a cent for everything else) and must never fire
without an explicit, capped opt-in. Treating all three as off-by-default, capability-gated, fail-open add-ons
-- rather than always-on features -- is what makes them safe to ship into a project real users are already
running in production-like conditions.

## 3. Theory
**Why OTel is additive, not a replacement, for Day 9's tracing.** `app/observability/tracing.py`'s `span()`
already answers "where did this request's time go" with zero infrastructure, and every existing caller and test
depends on its exact shape (`Span`, `get_spans()`). Rewriting it to BE OpenTelemetry would force every user of
this project to run a collector just to see per-request timing locally. Instead, `app/observability/otel_export.py`
adds a second, independent effect inside the SAME `span()` call: when `otel_enabled=true` and the SDK is
installed, a REAL OTel span opens alongside the lightweight one; otherwise `otel_span()` is a true no-op. One
call site, two optional outputs.

**Why PII redaction happens on CLEANED text, before chunking -- not after, and not on raw page text.**
`detect_pii` (Day 7) only ever counts; it was never meant to touch content. `redact_pii` (today) returns a new
string with detected spans replaced by `[REDACTED_<CATEGORY>]` markers. For this to be a real guarantee ("the
raw value never reaches OpenAI"), it has to run BEFORE chunking, so the chunks that get embedded and stored are
already redacted -- running it after chunking, or only in the warning message, would mean the real value was
already sent to the embeddings API by the time anyone saw a warning.

**Why credit_card is redacted before the broader phone pattern, not in the dict's natural iteration order.**
This surfaced as a real bug while writing today's tests: the phone regex is broad enough to also match pieces of
a 16-digit card number. Redacting phone-shaped spans first would shred a credit card into several
`[REDACTED_PHONE]` fragments before the credit-card pattern ever saw the whole, contiguous number. `redact_pii`
fixes this with an explicit category order (credit_card, ssn, email, phone -- most structurally specific first),
not the dict's insertion order `detect_pii` safely uses (detection only ever counts against the untouched
original text, so overlap never mattered there).

**Why OCR is OpenAI vision, not Tesseract or another local OCR library.** This curriculum has been
OpenAI-API-only since Day 1 specifically so there is exactly one provider, one client, one retry/cost-accounting
path to maintain (see docs/PROJECT_GUIDE.md). A vision-capable chat model reuses that exact path; a traditional
OCR library would need its own system binary, its own error handling, and its own cost model (free, but with a
completely different quality/dependency trade-off) bolted on next to it. The trade-off this buys: OCR here costs
real money per page and needs no local install beyond a pure-Python renderer (PyMuPDF) to turn a page into an
image OpenAI can read.

**Why OCR is capped per document (`ocr_max_pages_per_document`), unlike other per-document guards.** Every other
cost guard in this project (`max_chunks_per_document`, `max_index_tokens`) exists to catch ABUSE -- a reasonable
document should never hit them. OCR's cap exists to bound NORMAL, EXPECTED cost: a 300-page scanned PDF
uploaded in good faith would otherwise trigger 300 real vision calls in one request. The cap silently leaves
pages beyond it `EMPTY` rather than failing the whole upload -- partial OCR is strictly better than none.

## 4. Architecture
```
app/config.py
  pii_mode: "off" | "warn" (default) | "redact"
  otel_enabled, otel_service_name, otel_exporter_otlp_endpoint ("" = console exporter)
  ocr_enabled, openai_ocr_model, ocr_max_pages_per_document, ocr_price_input/output_per_million

app/services/pii.py
  redact_pii(text) -> text with [REDACTED_<CATEGORY>] markers, credit_card/ssn/email/phone order

app/services/ingestion.py
  index_document(): clean -> (pii_mode: warn=count only, redact=count AND rewrite `pages`) -> chunk -> embed
  IndexReport gains pii_mode (what was actually applied)

app/observability/otel_export.py   -- NEW
  init_otel(settings)   called once from app/main.py's lifespan; no-op unless otel_enabled + SDK installed
  otel_span(name, **attrs)   context manager; true no-op when disabled/unavailable
app/observability/tracing.py
  span() now wraps its body in otel_span(name, **attributes) -- ADDITIVE, Span/get_spans() unchanged

app/services/ocr.py   -- NEW
  Ocr (Protocol) / OpenAIOcr -- a vision chat completion, same AsyncOpenAI client every other service shares
  render_page_to_png(file_bytes, page_index) -- lazy `import pymupdf as fitz`, raises ImportError if missing
app/services/pdf_parser.py
  parse_pdf() is now ASYNC; optional ocr_factory/ocr_max_pages/ocr_price_* kwargs (all default to "OCR off")
  _run_ocr_pass(): for each EMPTY page up to the cap, render -> transcribe -> on success, status becomes OCR
app/dependencies.py
  get_ocr_factory(): returns None unless ocr_enabled + a real API key -- same fail-open shape as every other
  *_factory in this file (get_reranker_factory, get_generator_factory, ...)
app/routers/documents.py
  upload_document() awaits parse_pdf(..., ocr_factory=...), bills OCR cost to cost_tracker + tenant spend
```

## 5. Files
NEW: `app/observability/otel_export.py`, `app/services/ocr.py`, `requirements-otel.txt`, `requirements-ocr.txt`,
`tests/test_otel.py`, `tests/test_ocr.py`, `scripts/verify_day21.py`, `docs/day21.md`.
CHANGED: `app/config.py` (`pii_mode` replaces `warn_on_pii`; new otel_*/ocr_* settings), `app/services/pii.py`
(`redact_pii`), `app/services/ingestion.py` (redact-before-chunk, `pii_mode` on IndexReport), `app/models/chunk.py`
(`IndexReport.pii_mode`), `app/models/document.py` (`PageStatus.OCR`, `DocumentMetadata.ocr_pages_used/ocr_tokens/
ocr_cost_usd`), `app/observability/tracing.py` (wraps `span()` with `otel_span()`), `app/services/pdf_parser.py`
(async; OCR pass), `app/dependencies.py` (`get_ocr_factory`), `app/routers/documents.py` (awaits `parse_pdf`,
bills OCR cost), `app/main.py` (calls `init_otel`, version bump), `tests/fakes.py` (`ScriptedOcrFactory`),
`tests/test_pii.py`, `tests/test_index_endpoint.py`, `tests/test_pdf_parser.py` (now async),
`tests/test_documents_endpoint.py`, `.env.example`, `README.md`.

## 6-7. Code notes
- `parse_pdf` becoming `async` is the one call-site-visible breaking change today -- every caller (the router,
  every test) now awaits it. With no `ocr_factory` passed (its default), behavior is byte-for-byte identical to
  Day 1-20; `tests/test_pdf_parser.py` has an explicit regression test for exactly this.
- `_run_ocr_pass` fails open at TWO granularities: a single page's render/transcribe failure just skips THAT
  page (`continue`); PyMuPDF being entirely missing (`ImportError`) aborts the whole pass immediately, since
  every subsequent page would fail identically -- no point burning vision-call budget on pages 2-5 after page 1
  already proved the renderer isn't installed.
- `redact_pii`'s category ordering bug (phone pattern shredding a credit card) was caught by its own test suite,
  not discovered in production -- `test_redact_pii_only_masks_luhn_valid_credit_cards` failed on the first
  implementation attempt, which is exactly what that test existed to catch.
- `otel_export.reset_for_tests()` exists ONLY for test isolation (a module-level `_tracer` singleton, same
  rationale as `cost_tracker`/`latency_recorder`'s autouse reset fixture) -- application code never calls it.

## 8. Run
```
python scripts/verify_day21.py             # pytest + offline PII/OCR/OTel checks + a real PII-redact round trip
python scripts/verify_day21.py --offline   # skip the one real OpenAI call entirely
python scripts/verify_day21.py --with-ocr  # ALSO makes one real, billed vision call to prove OCR end-to-end

# Try PII redaction:
#   PII_MODE=redact in .env, then upload a document containing an email/SSN/card number --
#   GET /documents/{id}/chunks will show [REDACTED_EMAIL] etc., never the original value.

# Try OCR (needs: pip install -r requirements-ocr.txt, OCR_ENABLED=true in .env):
#   upload a scanned (image-only) PDF -- ParsedDocument.pages[i].status == "ocr" instead of "empty",
#   and metadata.ocr_pages_used/ocr_cost_usd show what it cost.

# Try OTel (needs: pip install -r requirements-otel.txt, OTEL_ENABLED=true in .env):
#   uvicorn app.main:app --reload, make any request that uses span() (e.g. POST /documents/{id}/index) --
#   spans print to the console (or ship to OTEL_EXPORTER_OTLP_ENDPOINT if set).
```

## 9. Testing
361 tests total (23 new since Day 20's 338: PII redaction x6 in `tests/test_pii.py`, PII mode wiring x2 in
`tests/test_index_endpoint.py`, OCR mechanism x7 in `tests/test_ocr.py`, OCR router wiring x2 in
`tests/test_documents_endpoint.py`, OTel x5 in `tests/test_otel.py`, plus a `parse_pdf`-is-now-async regression
guard added to `tests/test_pdf_parser.py`). The OCR
tests exercise REAL PyMuPDF rendering against a fake (scripted) transcriber -- only the OpenAI vision call
itself is faked, so the rendering path is genuinely proven, not just the plumbing around it. The OTel tests use
OpenTelemetry's own `InMemorySpanExporter`, so a real `TracerProvider`/span lifecycle runs with nothing leaving
the process.

## 10. Failure cases
PII: `pii_mode="redact"` with no PII detected is a no-op (nothing to rewrite); a PII-detection false positive
(e.g. an invoice number flagged as a possible credit card) degrades to redacting that span too -- the cost of a
probabilistic detector, unchanged from Day 7's own false-positive profile. OCR: no OCR factory, a factory
returning `None`, PyMuPDF missing, a render error, or a vision-call error all leave the affected page exactly
`EMPTY`, logged, never raised past the upload endpoint. OTel: `otel_enabled=true` with the SDK missing logs one
warning at startup and leaves tracing exactly as it was Day 9-20; a bad `otel_exporter_otlp_endpoint` falls back
to the console exporter rather than failing startup.

## 11. Security
PII redaction is a net security IMPROVEMENT with one honest caveat: it's regex-based (Day 7's existing
detector), so it inherits that detector's false-negative profile -- it catches the PATTERNS it was built for
(email, phone, SSN, Luhn-valid card), not every possible PII shape. OCR sends a rendered page IMAGE to OpenAI,
same data-handling posture as every other OpenAI call this project already makes with document text -- nothing
new leaves the process that wasn't already leaving it in text form for `OK`-status pages. OTel span attributes
are filtered to primitive types before being attached, so an unexpected object/dict passed to `span(**kwargs)`
is dropped rather than raising or leaking an unintended repr into exported trace data.

## 12. Cost
PII redaction and OTel export have no per-request cost of their own (redaction is pure-Python regex; OTel
span export is local/async). OCR is the one real new cost center in this project: a vision-capable chat
completion per scanned page, billed through the exact same `cost_tracker`/per-tenant daily budget path as every
other OpenAI call, and hard-capped per document via `ocr_max_pages_per_document` (default 5) specifically
because it's materially more expensive than anything else here.

## 13. Production improvements
With real OTel export, PII redaction, and OCR now shipped, every item that had been sitting in this project's
"NOT YET" tracker since Day 19 is closed. Remaining future directions are genuinely new scope, not backlog:
RAGAS/DeepEval-based evaluation, LangGraph-based orchestration, a real Prometheus/Grafana deployment (vs. Day
9's `/metrics` endpoint, which is real and scrapeable but has nothing scraping it today).

## 14. Interview Q&A
- *Why does OCR need its own requirements file instead of just using PyMuPDF if it's already a dependency
  elsewhere?* It isn't a dependency elsewhere -- this project has never needed to RENDER a PDF page as an image
  before (pypdf extracts text, it doesn't rasterize). Keeping it in `requirements-ocr.txt` means the default
  install never pays PyMuPDF's cost for a capability that's off by default.
- *Why is PII redaction applied to cleaned text before chunking rather than redacting each final chunk?*
  Functionally there's no difference in WHAT gets redacted (the same text ends up chunked either way), but
  redacting before chunking means a PII span that happens to straddle a chunk boundary is still redacted as one
  contiguous match, instead of being split and potentially missed by the regex on either side of the cut.
- *What's the actual risk of the regex-based phone pattern over-matching into a redacted credit card, as the
  test caught?* None, once the ordering fix is in: credit_card is checked and redacted FIRST, so by the time the
  phone pattern runs, a Luhn-valid card number is already replaced by a single `[REDACTED_CREDIT_CARD]` token
  that the phone pattern can't partially match into.
- *Why is OTel's attribute filtering (`isinstance(v, (str, bool, int, float))`) done in `otel_span()` rather than
  fixing every `span(...)` call site to only ever pass OTel-safe values?* `span()` is called from a dozen
  call sites across routers and services that were all written before OTel existed and pass whatever's
  convenient (ints, floats, the occasional list). Filtering centrally in one place is one small, obviously
  correct guard instead of auditing and constraining every existing call site for a feature most users won't
  enable.

## 15. Day-end checklist
- [ ] `python scripts/verify_day21.py --offline` ends with ALL CHECKS PASSED
- [ ] you can explain why OCR needed `parse_pdf` to become `async`
- [ ] you can explain why credit_card is redacted before the phone pattern runs
- [ ] you can explain why OTel is additive to Day 9's tracing rather than replacing it

## 16. Learned
Fail-open isn't one pattern, it's a property every optional feature needs to be independently re-derived for:
OTel fails open by checking `_tracer is None`, PII redaction fails open by simply never being invoked unless
`pii_mode` opts in, and OCR fails open at two different granularities (per-page vs. whole-pass) depending on
whether the failure is local to one page or systemic. Also: writing the "happy path" test for a regex-based
transformation (`redact_pii`) is often what SURFACES the ordering/overlap bug that counting-only code
(`detect_pii`) never exposed, because mutation is where pattern overlap actually bites.

## 17. Next: Day 22
With the full checklist (ingestion, retrieval, generation, evaluation, security, observability, multi-tenancy,
CI/CD, containerization, agent memory, dependency hygiene, and now OTel/PII redaction/OCR) complete, Project 1
-- Production RAG with Citations -- is done end to end. The natural next step is Project 2 of the curriculum.

## Tracker
COMPLETED: everything listed in `docs/day20.md`'s tracker, plus: optional real OpenTelemetry export additive to
Day 9's local tracing, a PII redaction mode that actually removes detected values before they're chunked/
embedded/stored (not just a warning), and vision-based OCR for scanned PDFs via an OpenAI vision model, capped
per document and fully fail-open.
NOT YET: RAGAS/DeepEval-based evaluation, LangGraph-based orchestration, a real Prometheus/Grafana deployment.
PROJECT STATUS: 100% of the original checklist AND every item added along the way. Production RAG with
Citations is feature-complete: ingestion (including OCR for scanned PDFs), retrieval (exact and approximate),
grounded generation, evaluation, security (including PII redaction), full observability (local and, optionally,
real OTel), multi-tenancy, three-tier CI/CD with image publishing, agent memory, and dependency hygiene are all
done and tested.
