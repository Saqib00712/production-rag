"""
One-command Day 9 verification.

    python scripts/verify_day9.py            # pytest + a real /ask call, then inspect the trace/metrics it produced
    python scripts/verify_day9.py --offline  # pytest only
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
    print("\n[1/2] Test suite (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes (includes tracing + metrics + trace-unification suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: one real /ask call, then inspect what it left behind")
    import logging

    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.config import Settings, get_settings
    from app.main import app

    logs: list[dict] = []

    class Capture(logging.Handler):
        def emit(self, record):
            logs.append(record.__dict__)

    handler = Capture()
    logging.getLogger().addHandler(handler)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db")
        app.dependency_overrides[get_settings] = lambda: settings
        app.state.rate_limiter.reset()
        client = TestClient(app)

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within 30 days of purchase.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")})
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index")

        logs.clear()
        r = client.post("/ask", json={"question": "How many days for a refund?", "document_id": doc})
        header_id = r.headers.get("X-Request-ID")
        check("real /ask call returns an X-Request-ID header", bool(header_id))

        service_ids = {rec["request_id"] for rec in logs if hasattr_dict(rec, "request_id") and rec.get("name", "").startswith("app.services")}
        audit_lines = [rec for rec in logs if rec.get("name") == "audit"]
        check("audit log line was emitted for the /ask call", len(audit_lines) >= 1)
        if audit_lines:
            check("audit log's request_id matches the response header", audit_lines[-1].get("request_id") == header_id)
        check("every internal service log for this request shares the SAME trace id",
              service_ids == {header_id} or service_ids == set(),
              f"found ids: {service_ids}")

        trace_span_names = []
        for rec in audit_lines:
            trace_span_names.extend(s["name"] for s in rec.get("spans", []))
        print(f"  spans recorded for this request: {trace_span_names}")
        check("the trace recorded at least a retrieval span", "retrieval" in trace_span_names)

        summary = client.get("/observability/summary").json()
        print(f"  /observability/summary: {summary}")
        check("summary shows at least one recorded request", summary["requests"]["total"] >= 1)
        check("summary shows non-zero cost after a real /ask call", summary["cost"]["total_usd"] > 0)

        metrics_text = client.get("/metrics").text
        check("/metrics exposes Prometheus text format", "http_requests_total" in metrics_text)

        dash = client.get("/observability/dashboard")
        check("/observability/dashboard serves HTML", dash.status_code == 200 and "text/html" in dash.headers["content-type"])

        app.dependency_overrides.clear()
    logging.getLogger().removeHandler(handler)


def hasattr_dict(d: dict, key: str) -> bool:
    return key in d and d[key] is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()

    if args.offline:
        print("\n[2/2] Skipped (--offline): live check needs a real OPENAI_API_KEY.")
    else:
        from app.config import Settings

        key = Settings().openai_api_key
        if not key or key.startswith("sk-your"):
            print("\n[2/2] Skipped: no OPENAI_API_KEY (use --offline to skip this deliberately)")
        else:
            step_live()

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
