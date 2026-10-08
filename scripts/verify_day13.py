"""
One-command Day 13 verification.

    python scripts/verify_day13.py            # pytest + a real multi-turn /ask conversation
    python scripts/verify_day13.py --offline  # pytest only

The live check is the whole point of Day 13: it asks a FOLLOW-UP question
that is unanswerable on its own ("What about the second one?") and checks
that the contextualizer rewrites it into something retrieval can actually
act on, using the real conversation history -- not just that the plumbing
returns 200.
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
    check("pytest passes (includes conversation repository + contextualizer suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: a real multi-turn conversation, including a pronoun follow-up")
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
        )
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides.pop(get_current_tenant, None)
        app.state.rate_limiter.reset(); app.state.tenant_rate_limiter.reset()
        client = TestClient(app)

        repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        repo.create_key(key, "verify-day13", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Warranty terms. The device carries a two year limited warranty covering defects.")
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Shipping information. Standard shipping takes five to seven business days.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index", headers=headers)

        conv = client.post("/conversations", headers=headers)
        check("POST /conversations returns 201 with a conversation_id", conv.status_code == 201 and bool(conv.json().get("conversation_id")))
        cid = conv.json()["conversation_id"]

        r1 = client.post("/ask", json={"question": "What is the warranty on the device?", "document_id": doc,
                                        "conversation_id": cid}, headers=headers)
        body1 = r1.json()
        print(f"  turn 1 answer: {body1.get('answer', '')[:120]!r}")
        check("turn 1 (first-ever turn) answers without a search_query rewrite", r1.status_code == 200 and body1.get("search_query") is None)

        # The canonical multi-turn failure mode: this question is meaningless
        # to a retriever on its own -- it only makes sense given turn 1.
        r2 = client.post("/ask", json={"question": "What about shipping instead?", "document_id": doc,
                                        "conversation_id": cid}, headers=headers)
        body2 = r2.json()
        print(f"  turn 2 answer: {body2.get('answer', '')[:120]!r}")
        print(f"  turn 2 search_query (contextualized): {body2.get('search_query')!r}")
        check("turn 2 is answered (not refused) using history to resolve the follow-up",
              r2.status_code == 200 and not body2.get("insufficient_evidence"))
        check("turn 2's search_query was rewritten (differs from the raw question)",
              bool(body2.get("search_query")) and body2["search_query"] != body2["question"])

        hist = client.get(f"/conversations/{cid}", headers=headers)
        turns = hist.json().get("turns", [])
        check("GET /conversations/{id} shows both turns, in order", len(turns) == 2 and turns[0]["turn_index"] == 0)

        other_key = generate_api_key()
        repo.create_key(other_key, "verify-day13-other-tenant", "verify script")
        cross = client.get(f"/conversations/{cid}", headers={"Authorization": f"Bearer {other_key}"})
        check("a different tenant gets 404 (not 403) for someone else's conversation", cross.status_code == 404)

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
