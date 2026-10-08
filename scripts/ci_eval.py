"""
A self-contained, cost-bounded live eval run meant for CI (GitHub Actions),
not local development (use scripts/run_eval.py / scripts/tune.py for that,
against your own documents).

Builds the same synthetic 8-page document + golden set used throughout this
project's verify_dayN.py scripts (so CI needs no fixture files committed to
the repo), runs it through the full real pipeline exactly once, and FAILS
the job (non-zero exit) if:
  - the cost exceeds --max-cost-usd (a hard ceiling -- see app/eval/cost_gate.py)
  - answer accuracy or hit@3 falls below the --min-* thresholds

This is deliberately separate from the $0 pytest suite, which is the
PRIMARY gate and runs unconditionally on every push with no API key needed
at all. This script is the SECONDARY, optional gate: it only runs when an
OPENAI_API_KEY secret is configured, and typically only on pushes to main
(not every PR) -- see .github/workflows/ci.yml for exactly when.

    python scripts/ci_eval.py                          # defaults: $0.05 ceiling, hit@3 >= 0.85, accuracy >= 0.85
    python scripts/ci_eval.py --max-cost-usd 0.10
"""

import argparse
import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOG_LEVEL", "WARNING")

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


async def run(max_cost_usd: float, min_hit_at_3: float, min_accuracy: float) -> int:
    from app.auth import generate_api_key
    from app.config import Settings, get_settings
    from app.dependencies import (
        get_embedder_factory, get_generator_factory, get_json_llm_factory, get_reranker_factory,
    )
    from app.eval.answer_metrics import AskLikeResult
    from app.eval.cost_gate import CostCeilingExceeded, check_cost_ceiling
    from app.eval.loader import load_golden_set
    from app.eval.runner import run_eval
    from app.repositories.chunk_repository import ChunkRepository
    from app.repositories.tenant_repository import TenantRepository
    from app.services.chunker import TokenCounter
    from app.services.ingestion import index_document
    from app.services.pdf_parser import parse_pdf
    from app.services.rag_pipeline import RagDeps, answer_question
    from app.services.search import SearchService

    settings = Settings()
    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
        print("No OPENAI_API_KEY available -- skipping live eval (this is expected on PRs; "
              "the $0 pytest suite is the gate that always runs).")
        return 0

    golden = load_golden_set("evals/golden/sample_8page.json")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/ci.db", tenant_db_path=f"{tmp}/tenants.db")
        tenant_repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        tenant_repo.create_key(key, tenant_id="ci", label="ci_eval.py")

        repo = ChunkRepository(settings.sqlite_path)
        counter = TokenCounter()
        embedder_factory = get_embedder_factory(settings)
        request_id = str(uuid.uuid4())
        parsed = parse_pdf(build_pdf(), "ci.pdf", "application/pdf", request_id)
        parsed.metadata.tenant_id = "ci"
        idx_report = await index_document(parsed, repo=repo, embedder_factory=embedder_factory,
                                           counter=counter, settings=settings, request_id=request_id)
        print(f"Indexed: {idx_report.chunk_count} chunks, {idx_report.tokens} tokens (~${idx_report.cost_usd:.6f})")

        search_service = SearchService(repo, embedder_factory, settings)
        llm_factory = get_json_llm_factory(settings)
        reranker = get_reranker_factory(settings, llm_factory)()
        generator = get_generator_factory(settings, llm_factory)()
        deps = RagDeps(search=search_service, reranker=reranker, generator=generator, counter=counter, settings=settings)
        running_cost = idx_report.cost_usd

        async def search_fn(query, document_id, top_k):
            r = await search_service.search(query=query, mode="hybrid", top_k=top_k, document_id=document_id,
                                              tenant_id="ci", request_id=str(uuid.uuid4()))
            nonlocal running_cost
            running_cost += r.query_cost_usd
            return [h.page_number for h in r.hits]

        async def ask_fn(question, document_id):
            nonlocal running_cost
            r = await answer_question(question=question, document_id=document_id, tenant_id="ci", top_k=5,
                                       use_rerank=True, deps=deps, request_id=str(uuid.uuid4()))
            running_cost += r.cost.total_cost_usd
            return AskLikeResult(refused=r.insufficient_evidence, refusal_reason=r.refusal_reason,
                                  answer_text=r.answer, cited_pages=[c.page_number for c in r.citations])

        report = await run_eval(golden, parsed.document_id, search_fn, ask_fn, retrieval_k=5,
                                 cost_fn=lambda: running_cost)

    print(f"\nRetrieval  hit@3={report.retrieval_hit_at_3:.2%}  MRR={report.retrieval_mrr:.3f}")
    print(f"Answers    accuracy={report.answer_accuracy:.2%}  false_refusal={report.false_refusal_rate:.2%}  "
          f"hallucinated_citation={report.hallucinated_citation_rate:.2%}")
    print(f"Cost       ~${report.cost_usd:.6f}  (ceiling: ${max_cost_usd:.6f})")

    exit_code = 0
    try:
        check_cost_ceiling(report.cost_usd, max_cost_usd)
    except CostCeilingExceeded as exc:
        print(f"FAIL: {exc}")
        exit_code = 1

    if report.retrieval_hit_at_3 < min_hit_at_3:
        print(f"FAIL: retrieval hit@3 {report.retrieval_hit_at_3:.2%} is below the required {min_hit_at_3:.2%}")
        exit_code = 1
    if report.answer_accuracy < min_accuracy:
        print(f"FAIL: answer accuracy {report.answer_accuracy:.2%} is below the required {min_accuracy:.2%}")
        exit_code = 1

    if exit_code == 0:
        print("\nPASS: live eval within cost ceiling and quality thresholds.")
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cost-usd", type=float, default=0.05)
    ap.add_argument("--min-hit-at-3", type=float, default=0.85)
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    args = ap.parse_args()
    return asyncio.run(run(args.max_cost_usd, args.min_hit_at_3, args.min_accuracy))


if __name__ == "__main__":
    sys.exit(main())
