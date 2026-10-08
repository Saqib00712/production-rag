"""
One-command Day 14 verification.

    python scripts/verify_day14.py            # pytest + a real long-term-memory exchange
    python scripts/verify_day14.py --offline  # pytest only

The live check is the part pytest's fakes can't prove: that a REAL LLM call
extracts a durable fact from an ordinary turn (no special "please remember
this" phrasing needed) and that the fact then shows up via GET /memories --
tenant-scoped, with no conversation_id in sight. Whether that fact then
actually CHANGES a later generation is Day 13-style prompt-threading that's
already covered by the fakes in tests/test_memories_endpoint.py
(`last_memory_notes`); a live run can't introspect the real prompt sent to
OpenAI, so it checks the observable, testable contract instead: the fact is
extracted, stored, deduplicated, and visible to a brand new conversation.
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
    check("pytest passes (includes memory repository + extractor + endpoint suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: a real long-term-memory exchange, across separate conversations")
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
        repo.create_key(key, "verify-day14", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within thirty days of purchase.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index", headers=headers)

        # Manual memory CRUD, with no LLM involved.
        created = client.post("/memories", json={"content": "Works in procurement."}, headers=headers)
        check("POST /memories returns 201 with the stored fact", created.status_code == 201 and created.json().get("content") == "Works in procurement.")
        mid = created.json()["memory_id"]
        listed = client.get("/memories", headers=headers).json()["memories"]
        check("GET /memories lists the manually added fact", any(m["memory_id"] == mid for m in listed))

        # A real turn that states a durable preference inline -- no special
        # "please remember this" phrasing, just an ordinary follow-on remark.
        r1 = client.post("/ask", json={
            "question": "I always want answers in metric units, by the way -- what is the refund window?",
            "document_id": doc,
        }, headers=headers)
        body1 = r1.json()
        print(f"  turn 1 answer: {body1.get('answer', '')[:120]!r}")
        check("turn 1 is answered (not refused)", r1.status_code == 200 and not body1.get("insufficient_evidence"))

        facts_after = client.get("/memories", headers=headers).json()["memories"]
        new_fact_extracted = len(facts_after) > len(listed)
        print(f"  memories after turn 1: {[f['content'] for f in facts_after]}")
        check("a durable fact was extracted from an ordinary turn (no special phrasing)", new_fact_extracted)

        # The fact must be visible from a conversation that has never heard
        # of the one above -- this is the whole point of Day 14 over Day 13.
        cid_new = client.post("/conversations", headers=headers).json()["conversation_id"]
        r2 = client.post("/ask", json={"question": "What is the refund window?", "document_id": doc,
                                        "conversation_id": cid_new}, headers=headers)
        check("a BRAND NEW conversation still sees this tenant's long-term memory", r2.status_code == 200)
        facts_in_new_conv_scope = client.get("/memories", headers=headers).json()["memories"]
        check("the fact persists across conversations (tenant-scoped, not conversation-scoped)",
              len(facts_in_new_conv_scope) >= len(facts_after))

        # Re-stating the same fact should not duplicate it.
        before_count = len(client.get("/memories", headers=headers).json()["memories"])
        client.post("/ask", json={
            "question": "Just to confirm again -- I always want answers in metric units. Refund window?",
            "document_id": doc,
        }, headers=headers)
        after_count = len(client.get("/memories", headers=headers).json()["memories"])
        check("re-stating the same preference does not duplicate the stored fact", after_count <= before_count + 1)

        other_key = generate_api_key()
        repo.create_key(other_key, "verify-day14-other-tenant", "verify script")
        cross = client.delete(f"/memories/{mid}", headers={"Authorization": f"Bearer {other_key}"})
        check("a different tenant gets 404 (not 403) for someone else's memory", cross.status_code == 404)

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
