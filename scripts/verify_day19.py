"""
One-command Day 19 verification.

    python scripts/verify_day19.py            # pytest + a real cache-hit check with two identical questions
    python scripts/verify_day19.py --offline  # pytest only

The live check proves the actual Day 19 promise with REAL OpenAI calls:
asking the exact same question twice (no conversation_id) pays for the
memory-lookup embedding only on the FIRST call -- the second is a pure
cache hit (zero tokens, zero cost) -- and even on that FIRST call,
document retrieval's own embedding of the same text is already free,
because memory lookup ran first and populated the shared cache.
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
    check("pytest passes (includes the new question-embedding-cache suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: the same question asked twice only pays for one real embed call")
    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.auth import generate_api_key, get_current_tenant
    from app.config import Settings, get_settings
    from app.main import app

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

        from app.repositories.tenant_repository import TenantRepository

        repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        repo.create_key(key, "verify-day19", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within thirty days of purchase.")
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Warranty terms. The device carries a two year limited warranty covering "
                               "manufacturing defects. Claims require serial number ERR-4471.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index", headers=headers)

        # A memory fact ahead of time -> memory lookup actually embeds the
        # question (otherwise it short-circuits before ever calling the
        # embedder -- see _fetch_memory_notes's memory_repo.count() == 0 check).
        client.post("/memories", json={"content": "Always wants answers in metric units."}, headers=headers)

        question = "What is the warranty on the device?"
        r1 = client.post("/ask", json={"question": question, "document_id": doc}, headers=headers)
        r2 = client.post("/ask", json={"question": question, "document_id": doc}, headers=headers)
        body1, body2 = r1.json(), r2.json()
        print(f"  turn 1 cost: {body1.get('cost')}")
        print(f"  turn 2 cost: {body2.get('cost')}")

        check("turn 1 is answered", r1.status_code == 200 and not body1.get("insufficient_evidence"))
        check("turn 1's memory_embedding cost is nonzero (first time asking -- a real call happened)",
              body1["cost"]["memory_embedding_tokens"] > 0)
        check("turn 1's retrieval embedding cost is ALREADY zero (memory lookup ran first and cached it)",
              body1["cost"]["embedding_tokens"] == 0)
        check("turn 2 is answered the same way", r2.status_code == 200 and not body2.get("insufficient_evidence"))
        check("turn 2's memory_embedding cost is zero (pure cache hit -- the exact same question)",
              body2["cost"]["memory_embedding_tokens"] == 0)

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
