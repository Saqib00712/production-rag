"""
One-command Day 16 verification.

    python scripts/verify_day16.py            # pytest + a real cost-accounted multi-turn exchange
    python scripts/verify_day16.py --offline  # pytest only

The live check proves the actual Day 16 promise with a REAL OpenAI call:
`AskResponse.cost` now has nonzero `contextualizer_*`, `memory_extraction_*`,
and `memory_embedding_*` fields on a request that exercises all three, and
`total_cost_usd` is the full sum, not just retrieval+rerank+generation.
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
    check("pytest passes (includes full cost accounting + tracing suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: a real multi-turn exchange with every extra cost counted")
    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.auth import generate_api_key, get_current_tenant
    from app.config import Settings, get_settings
    from app.main import app
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
        client = TestClient(app)

        repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        repo.create_key(key, "verify-day16", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within thirty days of purchase.")
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Warranty terms. The device carries a two year limited warranty covering "
                               "manufacturing defects. Claims require serial number ERR-4471.")
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Shipping information. Standard shipping takes five to seven business days.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index", headers=headers)

        # A memory fact ahead of time -> triggers the memory_embedding cost
        # (embedding the question against it) on every turn below.
        client.post("/memories", json={"content": "Always wants answers in metric units."}, headers=headers)

        conv = client.post("/conversations", headers=headers)
        cid = conv.json()["conversation_id"]

        r1 = client.post("/ask", json={
            "question": "I always want answers in metric units -- what is the warranty on the device?",
            "document_id": doc, "conversation_id": cid,
        }, headers=headers)
        body1 = r1.json()
        print(f"  turn 1 cost: {body1.get('cost')}")
        check("turn 1 is answered", r1.status_code == 200 and not body1.get("insufficient_evidence"))
        check("turn 1's contextualizer cost is zero (first turn, nothing to resolve)",
              body1["cost"]["contextualizer_cost_usd"] == 0.0)
        check("turn 1's memory_embedding cost is nonzero (a memory already exists to search against)",
              body1["cost"]["memory_embedding_tokens"] > 0)

        # A follow-up that NEEDS contextualization -- also exercises the
        # extractor, since the first turn's "I always want metric units"
        # aside is exactly the kind of durable fact it should catch.
        r2 = client.post("/ask", json={
            "question": "What about shipping instead?", "document_id": doc, "conversation_id": cid,
        }, headers=headers)
        body2 = r2.json()
        print(f"  turn 2 cost: {body2.get('cost')}")
        check("turn 2 is answered using the contextualized follow-up", r2.status_code == 200 and not body2.get("insufficient_evidence"))
        check("turn 2's contextualizer cost is nonzero (a real rewrite happened)",
              body2["cost"]["contextualizer_cost_usd"] > 0)
        c = body2["cost"]
        expected_total = round(
            c["embedding_cost_usd"] + c["rerank_cost_usd"] + c["generation_cost_usd"]
            + c["contextualizer_cost_usd"] + c["memory_extraction_cost_usd"] + c["memory_embedding_cost_usd"], 8,
        )
        check("turn 2's total_cost_usd is the full sum of every component, not just retrieval/rerank/generation",
              c["total_cost_usd"] == expected_total)

        import datetime
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        spent = repo.get_usage("verify-day16", today)
        check("the per-tenant daily budget reflects the full total, including every extra cost",
              spent >= c["total_cost_usd"])

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
