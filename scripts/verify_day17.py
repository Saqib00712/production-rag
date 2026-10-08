"""
One-command Day 17 verification.

    python scripts/verify_day17.py            # pytest + a real cost-visible dashboard check
    python scripts/verify_day17.py --offline   # pytest only

The live check proves the actual Day 17 promise with a REAL OpenAI call:
`/observability/summary`'s `cost.usd_by_stage` now has nonzero
`contextualize`, `memory_extract`, and `memory_embedding` entries after a
conversation+memory exchange, and `cost.total_usd` reflects them -- not
just embedding/rerank/generation like before today.
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
    check("pytest passes (includes the new cost_by_stage suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: /observability/summary now counts every stage")
    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.auth import generate_api_key, get_current_tenant
    from app.config import Settings, get_settings
    from app.main import app
    from app.observability.metrics import cost_tracker, latency_recorder
    from app.observability.tracing import reset as reset_tracing
    from app.repositories.tenant_repository import TenantRepository

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(
            raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db",
            tenant_db_path=f"{tmp}/tenants.db", conversation_db_path=f"{tmp}/conversations.db",
            memory_db_path=f"{tmp}/memories.db",
        )
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides.pop(get_current_tenant, None)
        app.state.rate_limiter.reset(); app.state.tenant_rate_limiter.reset()
        latency_recorder.reset(); cost_tracker.reset(); reset_tracing()
        client = TestClient(app)

        repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        repo.create_key(key, "verify-day17", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within thirty days of purchase.")
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Shipping information. Standard shipping takes five to seven business days.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index", headers=headers)

        client.post("/memories", json={"content": "Always wants answers in metric units."}, headers=headers)
        conv = client.post("/conversations", headers=headers)
        cid = conv.json()["conversation_id"]

        client.post("/ask", json={
            "question": "What is the warranty on the device?", "document_id": doc, "conversation_id": cid,
        }, headers=headers)
        r2 = client.post("/ask", json={
            "question": "What about shipping instead?", "document_id": doc, "conversation_id": cid,
        }, headers=headers)
        check("turn 2 is answered using the contextualized follow-up",
              r2.status_code == 200 and not r2.json().get("insufficient_evidence"))

        summary = client.get("/observability/summary", headers=headers).json()
        usd_by_stage = summary["cost"]["usd_by_stage"]
        print(f"  usd_by_stage: {usd_by_stage}")
        check("dashboard now counts contextualize", usd_by_stage.get("contextualize", 0) > 0)
        check("dashboard now counts memory_extract", usd_by_stage.get("memory_extract", 0) > 0)
        check("dashboard now counts memory_embedding", usd_by_stage.get("memory_embedding", 0) > 0)
        check("total_usd includes every stage, not just embedding/rerank/generation",
              summary["cost"]["total_usd"] >= sum(usd_by_stage.values()) - 1e-9)

        app.dependency_overrides.clear()


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
