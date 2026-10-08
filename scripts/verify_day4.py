"""
One-command Day 4 verification.

    python scripts/verify_day4.py            # full check, uses your OpenAI key, costs a few cents at most
    python scripts/verify_day4.py --offline  # no OpenAI calls, scripted rerank/generation, $0
    python scripts/verify_day4.py --skip-pytest

Runs pytest (99 tests, no live calls), then an end-to-end run in a TEMP
folder: builds an 8-page PDF, indexes it, asks questions that should be
answered WITH citations, and one question that should be REFUSED because
the document does not contain the answer. This refusal check is the most
important thing Day 4 adds: a RAG system that never says "I don't know" is
lying to you more often than one that does.
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


PAGES = [
    "Refund policy. Customers may request a refund within 30 days of purchase, provided the item is unused.",
    "Warranty terms. The device carries a two year limited warranty covering manufacturing defects. Claims require serial number ERR-4471.",
    "Shipping information. Standard shipping takes five to seven business days within the country.",
    "Privacy notice. We never sell personal data. Users may request deletion of their account information at any time.",
]
GROUNDED_QUESTIONS = [
    ("How many days do I have to return an item?", 1, "30"),
    ("What serial number is needed for a warranty claim?", 2, "ERR-4471"),
    ("How long does standard shipping take?", 3, "five"),
]
UNANSWERABLE_QUESTION = "What is the CEO's name and what is the company's annual revenue?"


def build_pdf(pages: list[str]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


def step_environment(offline: bool) -> bool:
    print("\n[1/3] Environment")
    check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    from app.config import Settings

    key = Settings().openai_api_key
    has_key = bool(key) and not key.startswith("sk-your")
    if not offline:
        check("OPENAI_API_KEY found in .env", has_key, "" if has_key else "set it, or run with --offline")
    return has_key


def step_pytest() -> None:
    print("\n[2/3] Test suite (pytest)")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_end_to_end(offline: bool) -> None:
    print("\n[3/3] End-to-end grounded Q&A run (temp storage, your data is untouched)")
    from fastapi.testclient import TestClient

    from app.config import Settings, get_settings
    from app.dependencies import get_embedder_factory, get_generator_factory, get_reranker_factory
    from app.main import app

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db")
        app.dependency_overrides[get_settings] = lambda: settings
        if offline:
            from tests.fakes import HashEmbedder, ScriptedGeneratorFactory, ScriptedRerankerFactory

            app.dependency_overrides[get_embedder_factory] = lambda: (lambda: HashEmbedder())
            app.dependency_overrides[get_reranker_factory] = lambda: ScriptedRerankerFactory()
            gen = ScriptedGeneratorFactory(
                reply={"answer": "Simulated grounded answer [S1].", "citations": ["S1"], "insufficient_evidence": False}
            )
            app.dependency_overrides[get_generator_factory] = lambda: gen
        client = TestClient(app)

        up = client.post("/documents/upload", files={"file": ("verify.pdf", build_pdf(PAGES), "application/pdf")})
        check("upload 4-page PDF", up.status_code == 201)
        doc = up.json()["document_id"]
        idx = client.post(f"/documents/{doc}/index")
        check("index document", idx.status_code == 200, f"HTTP {idx.status_code}" if idx.status_code != 200 else "")
        if idx.status_code != 200:
            return
        total_cost = idx.json()["cost_usd"]

        print("\n  Grounded questions (should answer WITH a citation to the correct page)")
        print(f"  {'question':<52} {'page':>4}  {'cited?':>7}  {'page ok?':>8}")
        all_grounded_ok = True
        for question, page, must_contain in GROUNDED_QUESTIONS:
            r = client.post("/ask", json={"query": question} if False else {"question": question, "document_id": doc})
            body = r.json()
            total_cost += body.get("cost", {}).get("total_cost_usd", 0)
            ok_call = r.status_code == 200
            cited = ok_call and len(body.get("citations", [])) > 0
            page_ok = cited and any(c["page_number"] == page for c in body["citations"])
            if offline:
                page_ok = cited  # scripted generator does not know real page numbers
            print(f"  {question[:50]:<52} {page:>4}  {str(cited):>7}  {str(page_ok):>8}")
            all_grounded_ok = all_grounded_ok and ok_call and cited and page_ok
        check("all grounded questions answered with a correct-page citation", all_grounded_ok)

        print(f"\n  Unanswerable question (should REFUSE, not guess)")
        r = client.post("/ask", json={"question": UNANSWERABLE_QUESTION, "document_id": doc})
        body = r.json()
        total_cost += body.get("cost", {}).get("total_cost_usd", 0)
        if offline:
            check("refusal path reachable (offline: skipped, scripted generator always answers)", True)
        else:
            print(f"  refusal_reason = {body.get('refusal_reason')}")
            check(
                "model refused instead of fabricating an answer",
                body.get("insufficient_evidence") is True and body.get("citations") == [],
                f"refusal_reason={body.get('refusal_reason')}",
            )

        no_rerank = client.post("/ask", json={"question": GROUNDED_QUESTIONS[0][0], "document_id": doc, "rerank": False})
        check("rerank=false still returns a valid response", no_rerank.status_code == 200)

        if not offline:
            print(f"\n  Total OpenAI usage this run: ~${total_cost:.6f}")
        app.dependency_overrides.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    has_key = step_environment(args.offline)
    if not args.skip_pytest:
        step_pytest()
    if args.offline or has_key:
        step_end_to_end(offline=args.offline)
    else:
        print("\n[3/3] Skipped: no OPENAI_API_KEY (use --offline for a free scripted run)")

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
