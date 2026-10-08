"""
One-command Day 21 verification.

    python scripts/verify_day21.py             # pytest + offline PII/OCR/OTel checks + a real PII-redact round trip
    python scripts/verify_day21.py --offline    # everything except the real OpenAI call
    python scripts/verify_day21.py --with-ocr   # ALSO makes one real (billed) vision call to prove OCR end-to-end

All three Day 21 features (PII redaction, OTel export, OCR) are OFF by
default and fail open if their optional dependency isn't installed, so
most of this script runs with zero cost and zero extra installs. The one
real OpenAI call in the default live check is a cheap embedding/generation
round trip proving PII_MODE=redact actually keeps raw PII out of storage;
the vision OCR call is real money (a vision-capable chat completion per
page) so it's opt-in via --with-ocr, not run by default.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOG_LEVEL", "WARNING")

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def step_pytest() -> None:
    print("\n[1/4] Test suite (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes (includes PII redaction, OCR, and OTel suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_pii_offline() -> None:
    print("\n[2/4] PII redaction (offline -- pure functions, no API needed)")
    from app.services.pii import redact_pii

    out = redact_pii("Email jane@example.com, SSN 123-45-6789, card 4111 1111 1111 1111")
    check("redact_pii masks email", "jane@example.com" not in out and "[REDACTED_EMAIL]" in out)
    check("redact_pii masks ssn", "123-45-6789" not in out and "[REDACTED_SSN]" in out)
    check("redact_pii masks a Luhn-valid credit card", "[REDACTED_CREDIT_CARD]" in out)


def step_ocr_offline() -> None:
    print("\n[3/4] OCR mechanism (offline -- fake transcriber, real PyMuPDF render if installed)")
    import asyncio

    from app.models.document import PageStatus
    from app.services.ocr import OcrResult

    class _FakeOcr:
        async def transcribe(self, image_bytes: bytes) -> OcrResult:
            return OcrResult(text="offline-verify transcription", prompt_tokens=100, completion_tokens=10)

    async def _run():
        from fpdf import FPDF

        from app.services.pdf_parser import parse_pdf

        pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", size=12)  # a genuinely blank page -> EMPTY
        pdf_bytes = bytes(pdf.output())
        return await parse_pdf(
            file_bytes=pdf_bytes, filename="t.pdf", content_type="application/pdf", request_id="verify-day21",
            ocr_factory=lambda: _FakeOcr(), ocr_max_pages=5,
        )

    try:
        parsed = asyncio.run(_run())
    except ImportError as exc:
        print(f"  Skipped: {exc} (pip install -r requirements-ocr.txt to exercise the real render path)")
        return

    check("a scanned (empty) page is picked up by the OCR pass", parsed.pages[0].status == PageStatus.OCR)
    check("the transcribed text replaces the empty page's text", "offline-verify" in parsed.pages[0].text)
    check("ocr_pages_used is counted on DocumentMetadata", parsed.metadata.ocr_pages_used == 1)


def step_otel_offline() -> None:
    print("\n[4/4] OTel export (offline -- disabled is a no-op; a real span if the SDK is installed)")
    from app.config import Settings
    from app.observability import otel_export
    from app.observability.tracing import span

    otel_export.reset_for_tests()
    otel_export.init_otel(Settings(otel_enabled=False))
    check("disabled by default: is_enabled() is False", otel_export.is_enabled() is False)
    with span("verify_span", x=1):
        pass  # must not raise even though nothing is listening

    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        print("  SDK not installed -- skipping the real-span check (pip install -r requirements-otel.txt).")
        otel_export.reset_for_tests()
        return

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_export._tracer = provider.get_tracer("verify")
    with span("verify_real_span", chunks=3):
        pass
    spans = exporter.get_finished_spans()
    check("a real OTel span is recorded when the SDK is installed and enabled",
          len(spans) == 1 and spans[0].name == "verify_real_span")
    otel_export.reset_for_tests()


def step_live(with_ocr: bool) -> None:
    print("\n[live] Real check: PII_MODE=redact keeps raw PII out of storage" +
          (" + a real (billed) OCR vision call" if with_ocr else ""))
    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.auth import generate_api_key, get_current_tenant
    from app.config import Settings, get_settings
    from app.main import app

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(
            raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db",
            tenant_db_path=f"{tmp}/tenants.db", conversation_db_path=f"{tmp}/conversations.db",
            memory_db_path=f"{tmp}/memories.db", pii_mode="redact",
            ocr_enabled=with_ocr,
        )
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides.pop(get_current_tenant, None)
        app.state.rate_limiter.reset(); app.state.tenant_rate_limiter.reset()
        client = TestClient(app)

        from app.repositories.tenant_repository import TenantRepository

        repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        repo.create_key(key, "verify-day21", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Contact our support team at support@example.com for refund questions.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        idx = client.post(f"/documents/{doc}/index", headers=headers).json()

        check("the document was reported to contain PII", idx["pii_detected"].get("email") == 1)
        check("pii_mode=redact is reflected in the response", idx["pii_mode"] == "redact")
        chunks = client.get(f"/documents/{doc}/chunks", headers=headers).json()["chunks"]
        all_text = " ".join(c["text"] for c in chunks)
        check("the raw email never reached storage", "support@example.com" not in all_text)
        check("the redaction marker IS in storage", "[REDACTED_EMAIL]" in all_text)

        if with_ocr:
            scanned = FPDF(); scanned.add_page(); scanned.set_font("Helvetica", size=12)  # genuinely blank
            up2 = client.post("/documents/upload", files={"file": ("scan.pdf", bytes(scanned.output()), "application/pdf")}, headers=headers)
            meta = up2.json()["metadata"]
            check("a real vision call OCR'd the blank page", meta["ocr_pages_used"] == 1)
            check("the OCR cost was billed", meta["ocr_cost_usd"] > 0)

        app.dependency_overrides.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the real OpenAI call entirely")
    ap.add_argument("--with-ocr", action="store_true", help="also make one real (billed) vision OCR call")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()
    step_pii_offline()
    step_ocr_offline()
    step_otel_offline()

    if args.offline:
        print("\n[live] Skipped (--offline): no real OpenAI call.")
    else:
        from app.config import Settings

        key = Settings().openai_api_key
        if not key or key.startswith("sk-your"):
            print("\n[live] Skipped: no OPENAI_API_KEY (use --offline to skip this deliberately)")
        else:
            step_live(with_ocr=args.with_ocr)

    failed = [r for r in RESULTS if not r[0]]
    print("\n" + "=" * 60)
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED:")
        for _, name, detail in failed:
            print(f"  - {name} {detail}")
        return 1
    print(f"ALL {len(RESULTS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
