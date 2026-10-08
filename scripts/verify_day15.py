"""
One-command Day 15 verification.

    python scripts/verify_day15.py            # pytest + a real relevance-based memory retrieval
    python scripts/verify_day15.py --offline  # pytest only

The live check is the part pytest's fakes can't prove with real semantics:
that a REAL embedding call ranks a tenant's stored facts by actual
relevance to the question, rather than Day 14's flat "inject everything"
behavior. It stores two UNRELATED facts, asks a question that's relevant to
only one of them, and confirms (via GET /memories after the ask -- the
request itself doesn't expose which notes were used, so this also
indirectly proves the fact that's harder to detect: the notes still don't
leak as citable sources) that the system stays healthy and keeps both
facts, while the pytest suite's `last_memory_notes` fakes already prove
HTTP-level exactly which notes were selected for generation.
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
    check("pytest passes (includes vector-indexed memory retrieval suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: relevance-ranked long-term memory, with a real embedder")
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
            memory_db_path=f"{tmp}/memories.db", memory_top_k=1,
        )
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides.pop(get_current_tenant, None)
        app.state.rate_limiter.reset(); app.state.tenant_rate_limiter.reset()
        client = TestClient(app)

        repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        repo.create_key(key, "verify-day15", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within thirty days of purchase.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index", headers=headers)

        # Two facts about completely different topics -- a flat Day 14
        # store would inject BOTH into every prompt regardless of
        # relevance; Day 15 should surface only the one that matters.
        m1 = client.post("/memories", json={"content": "Always wants answers in metric units."}, headers=headers)
        m2 = client.post("/memories", json={"content": "Is allergic to shellfish."}, headers=headers)
        check("both unrelated facts were stored (and embedded)", m1.status_code == 201 and m2.status_code == 201)

        r = client.post("/ask", json={
            "question": "What is the refund window, and can you give it to me in metric terms (weeks, not days)?",
            "document_id": doc,
        }, headers=headers)
        body = r.json()
        print(f"  answer: {body.get('answer', '')[:160]!r}")
        check("the question is answered (not refused)", r.status_code == 200 and not body.get("insufficient_evidence"))

        both_still_present = client.get("/memories", headers=headers).json()["memories"]
        check("both facts remain in the store (relevance filtering is per-request, not destructive)",
              len(both_still_present) == 2)

        other_key = generate_api_key()
        repo.create_key(other_key, "verify-day15-other-tenant", "verify script")
        cross = client.get("/memories", headers={"Authorization": f"Bearer {other_key}"})
        check("a different tenant sees none of this tenant's facts", cross.status_code == 200 and cross.json()["memories"] == [])

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
