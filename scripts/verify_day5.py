"""
One-command Day 5 verification.

    python scripts/verify_day5.py            # pytest + live eval run against a real document (a few cents)
    python scripts/verify_day5.py --offline  # pytest + scripted eval run, $0

Builds the same 8-page document as verify_day3/4.py, indexes it, and runs
evals/golden/sample_8page.json (8 answerable questions + 1 intentionally
unanswerable one) through the full retrieval+rerank+generation pipeline,
using scripts/run_eval.py's exact machinery. Prints the same summary a real
eval run would, and checks it against thresholds.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
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
    "Payment options. We accept Visa, Mastercard and PayPal. Bank transfers are available for orders above five hundred dollars.",
    "Account help. To reset a forgotten password, click the reset link on the sign in page and follow the emailed instructions.",
    "Product specifications. The rechargeable battery provides up to twelve hours of continuous use. Charging takes two hours.",
    "Customer support. Our support team is available Monday to Friday from nine to five by telephone and live chat.",
]


def build_pdf() -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    for text in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


def step_pytest() -> None:
    print("\n[1/2] Test suite (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


async def _run(offline: bool):
    from fastapi.testclient import TestClient

    from app.config import Settings, get_settings
    from app.dependencies import get_embedder_factory, get_generator_factory, get_reranker_factory
    from app.eval.answer_metrics import AskLikeResult
    from app.eval.loader import load_golden_set
    from app.eval.runner import run_eval
    from app.main import app
    from app.services.rag_pipeline import RagDeps, answer_question
    from app.services.search import SearchService
    from app.repositories.chunk_repository import ChunkRepository
    from app.services.chunker import TokenCounter

    golden = load_golden_set("evals/golden/sample_8page.json")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db")
        app.dependency_overrides[get_settings] = lambda: settings
        if offline:
            from tests.fakes import BowEmbedder, ScriptedGeneratorFactory, ScriptedRerankerFactory

            embedder = BowEmbedder()
            app.dependency_overrides[get_embedder_factory] = lambda: (lambda: embedder)
            reranker_factory = ScriptedRerankerFactory()
            app.dependency_overrides[get_reranker_factory] = lambda: reranker_factory
            generator_factory = ScriptedGeneratorFactory()
            app.dependency_overrides[get_generator_factory] = lambda: generator_factory
        client = TestClient(app)

        up = client.post("/documents/upload", files={"file": ("verify.pdf", build_pdf(), "application/pdf")})
        check("upload 8-page PDF", up.status_code == 201)
        doc = up.json()["document_id"]
        idx = client.post(f"/documents/{doc}/index")
        check("index document", idx.status_code == 200)
        if idx.status_code != 200:
            return

        repo = ChunkRepository(settings.sqlite_path)
        counter = TokenCounter()
        embedder_factory = get_embedder_factory(settings) if not offline else (lambda: embedder)
        search_service = SearchService(repo, embedder_factory, settings)

        if offline:
            reranker, generator = reranker_factory(), generator_factory()
        else:
            from app.dependencies import get_json_llm_factory
            llm_factory = get_json_llm_factory(settings)
            reranker = get_reranker_factory(settings, llm_factory)()
            generator = get_generator_factory(settings, llm_factory)()

        deps = RagDeps(search=search_service, reranker=reranker, generator=generator, counter=counter, settings=settings)

        async def search_fn(query, document_id, top_k):
            r = await search_service.search(query=query, mode="hybrid", top_k=top_k,
                                              document_id=document_id, request_id=str(uuid.uuid4()))
            return [h.page_number for h in r.hits]

        async def ask_fn(question, document_id):
            r = await answer_question(question=question, document_id=document_id, top_k=5,
                                       use_rerank=True, deps=deps, request_id=str(uuid.uuid4()))
            return AskLikeResult(refused=r.insufficient_evidence, refusal_reason=r.refusal_reason,
                                  answer_text=r.answer, cited_pages=[c.page_number for c in r.citations])

        report = await run_eval(golden, doc, search_fn, ask_fn, retrieval_k=5)
        app.dependency_overrides.clear()
        return report


def step_eval(offline: bool) -> None:
    print("\n[2/2] End-to-end evaluation run (temp storage, your data is untouched)")
    import asyncio

    report = asyncio.run(_run(offline))
    if report is None:
        return

    print(f"\n  {'ID':<5} {'ok?':<5} {'hit@1':<6} {'MRR':<6} {'refused':<8}  question")
    print("  " + "-" * 80)
    for q in report.questions:
        print(f"  {q.question_id:<5} {'YES' if q.answer.is_correct else 'NO':<5} "
              f"{str(q.retrieval.hit_at_1):<6} {q.retrieval.mrr:<6.2f} {str(q.answer.refused):<8}  {q.question[:45]}")

    print(f"\n  Retrieval  hit@1={report.retrieval_hit_at_1:.2%}  hit@3={report.retrieval_hit_at_3:.2%}  "
          f"MRR={report.retrieval_mrr:.3f}")
    print(f"  Answers    accuracy={report.answer_accuracy:.2%}  false_refusal={report.false_refusal_rate:.2%}  "
          f"hallucinated_citation={report.hallucinated_citation_rate:.2%}")

    if offline:
        check("scripted pipeline produced a report for all 9 questions", report.total_questions == 9)
        check("unanswerable question (q9) handling is scoreable", any(q.question_id == "q9" for q in report.questions))
    else:
        print(f"  Cost       ~${report.cost_usd:.6f}")
        check("retrieval hit@3 >= 8/9", report.retrieval_hit_at_3 >= 8 / 9, f"{report.retrieval_hit_at_3:.2%}")
        check("the unanswerable question (q9) was correctly refused",
              next(q for q in report.questions if q.question_id == "q9").answer.is_correct)
        check("false refusal rate == 0", report.false_refusal_rate == 0.0, f"{report.false_refusal_rate:.2%}")
        check("hallucinated citation rate == 0", report.hallucinated_citation_rate == 0.0,
              f"{report.hallucinated_citation_rate:.2%}")
        check("answer accuracy >= 8/9", report.answer_accuracy >= 8 / 9, f"{report.answer_accuracy:.2%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()

    if args.offline:
        step_eval(offline=True)
    else:
        from app.config import Settings

        key = Settings().openai_api_key
        if not key or key.startswith("sk-your"):
            print("\n[2/2] Skipped: no OPENAI_API_KEY (use --offline for a free scripted run)")
        else:
            step_eval(offline=False)

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
