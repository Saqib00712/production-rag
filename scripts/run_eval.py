"""
Run the golden-question evaluation against a REAL indexed document using
your real OpenAI key. This is what you run after every meaningful change
(chunk size, rerank threshold, prompt wording) to see whether it helped or
hurt -- a regression test with numbers, not a vibe check.

Usage:
    python scripts/run_eval.py --pdf path/to/your.pdf --golden evals/golden/your_set.json
    python scripts/run_eval.py --document-id <existing-id> --golden evals/golden/your_set.json
    python scripts/run_eval.py --pdf ... --golden ... --no-judge   # skip the optional LLM judge (cheaper)
    python scripts/run_eval.py --pdf ... --golden ... --no-rerank  # evaluate without reranking, for comparison

Writes a full JSON report to evals/reports/<timestamp>.json and prints a
summary table with per-question pass/fail so a single failing case is easy
to spot, not buried in an average.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOG_LEVEL", "WARNING")


async def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="PDF to upload and index before evaluating")
    src.add_argument("--document-id", help="Evaluate an already-indexed document instead")
    ap.add_argument("--golden", required=True, help="Path to a golden question JSON file")
    ap.add_argument("--top-k", type=int, default=5, help="Retrieval depth for metrics (default 5)")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    from app.config import get_settings
    from app.dependencies import get_embedder_factory, get_generator_factory, get_reranker_factory, get_repository
    from app.eval.answer_metrics import AskLikeResult
    from app.eval.loader import GoldenSetError, load_golden_set
    from app.eval.runner import run_eval
    from app.models.document import ParsedDocument
    from app.services.ingestion import index_document
    from app.services.chunker import TokenCounter
    from app.services.embedder import EmbeddingConfigError
    from app.services.judge import OpenAIJudge
    from app.services.llm import OpenAIJsonLLM
    from app.services.openai_client import close_openai_clients, get_openai_client
    from app.services.pdf_parser import parse_pdf
    from app.services.rag_pipeline import RagDeps, answer_question
    from app.services.search import SearchService

    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
        print("ERROR: OPENAI_API_KEY is not set in .env. This script needs a real key.")
        return 1

    try:
        golden = load_golden_set(args.golden)
    except GoldenSetError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Loaded {len(golden)} questions from {args.golden}")

    repo = get_repository(settings)
    counter = TokenCounter()
    embedder_factory = get_embedder_factory(settings)

    if args.pdf:
        pdf_bytes = Path(args.pdf).read_bytes()
        request_id = str(uuid.uuid4())
        parsed = parse_pdf(pdf_bytes, Path(args.pdf).name, "application/pdf", request_id)
        Path(settings.raw_storage_dir).mkdir(parents=True, exist_ok=True)
        (Path(settings.raw_storage_dir) / f"{parsed.document_id}.json").write_text(
            parsed.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Uploaded '{args.pdf}' -> document_id={parsed.document_id} ({parsed.metadata.page_count} pages)")
        report = await index_document(
            parsed, repo=repo, embedder_factory=embedder_factory, counter=counter,
            settings=settings, request_id=request_id,
        )
        document_id = parsed.document_id
        print(f"Indexed: {report.chunk_count} chunks, {report.tokens} tokens (~${report.cost_usd:.6f})")
    else:
        document_id = args.document_id

    from app.observability.metrics import cost_tracker

    search_service = SearchService(repo, embedder_factory, settings)
    llm_factory = lambda: OpenAIJsonLLM(get_openai_client(settings))
    try:
        reranker = None if args.no_rerank else get_reranker_factory(settings, llm_factory)()
    except EmbeddingConfigError:
        reranker = None
    generator = get_generator_factory(settings, llm_factory)()
    judge = None if args.no_judge else OpenAIJudge(llm_factory(), settings.openai_chat_model)

    deps = RagDeps(search=search_service, reranker=reranker, generator=generator, counter=counter, settings=settings)
    running_cost = 0.0

    # Day 17: snapshot cost_tracker's per-stage $ AFTER indexing (so that
    # one-time indexing cost doesn't get mixed into "per-question" spend,
    # same reasoning running_cost already follows) and diff it at
    # aggregation time -- answer_question/_retrieve_and_prepare already
    # feed this same singleton (see app/observability/metrics.py), so this
    # is free: no new accounting, just reading what's already recorded.
    cost_by_stage_start = dict(cost_tracker.cost_by_stage)

    def cost_by_stage_fn() -> dict[str, float]:
        return {
            stage: round(total - cost_by_stage_start.get(stage, 0.0), 8)
            for stage, total in cost_tracker.cost_by_stage.items()
            if round(total - cost_by_stage_start.get(stage, 0.0), 8) > 0
        }

    async def search_fn(query: str, doc_id: str, top_k: int) -> list[int]:
        result = await search_service.search(query=query, mode="hybrid", top_k=top_k, document_id=doc_id,
                                               request_id=str(uuid.uuid4()))
        nonlocal running_cost
        running_cost += result.query_cost_usd
        return [h.page_number for h in result.hits]

    async def ask_fn(question: str, doc_id: str) -> AskLikeResult:
        nonlocal running_cost
        resp = await answer_question(question=question, document_id=doc_id, top_k=args.top_k,
                                      use_rerank=not args.no_rerank, deps=deps, request_id=str(uuid.uuid4()))
        running_cost += resp.cost.total_cost_usd
        return AskLikeResult(refused=resp.insufficient_evidence, refusal_reason=resp.refusal_reason,
                              answer_text=resp.answer, cited_pages=[c.page_number for c in resp.citations])

    async def judge_fn(question, answer, g):
        nonlocal running_cost
        score = await judge.score(question, answer, g)
        running_cost += 0.0  # judge tokens are small; folded into the printed total below for simplicity
        return score

    print("\nRunning evaluation..." + ("" if not args.no_judge else " (judge disabled)")
          + ("" if not args.no_rerank else " (rerank disabled)"))
    report = await run_eval(golden, document_id, search_fn, ask_fn, retrieval_k=args.top_k,
                             judge_fn=judge_fn if judge else None, cost_fn=lambda: running_cost,
                             cost_by_stage_fn=cost_by_stage_fn)

    print(f"\n{'ID':<5} {'ok?':<5} {'hit@1':<6} {'hit@3':<6} {'MRR':<6} {'refused':<8} {'judge':<6}  question")
    print("-" * 100)
    for q in report.questions:
        print(f"{q.question_id:<5} {'YES' if q.answer.is_correct else 'NO':<5} "
              f"{str(q.retrieval.hit_at_1):<6} {str(q.retrieval.hit_at_3):<6} {q.retrieval.mrr:<6.2f} "
              f"{str(q.answer.refused):<8} {('%.1f' % q.answer.judge_score) if q.answer.judge_score is not None else '-':<6}  "
              f"{q.question[:50]}")

    print(f"\nRetrieval  hit@1={report.retrieval_hit_at_1:.2%}  hit@3={report.retrieval_hit_at_3:.2%}  "
          f"hit@5={report.retrieval_hit_at_5:.2%}  MRR={report.retrieval_mrr:.3f}  "
          f"recall@5={report.retrieval_recall_at_5:.2%}")
    print(f"Answers    accuracy={report.answer_accuracy:.2%}  false_refusal={report.false_refusal_rate:.2%}  "
          f"hallucinated_citation={report.hallucinated_citation_rate:.2%}"
          + (f"  avg_judge={report.avg_judge_score:.1f}/10" if report.avg_judge_score is not None else ""))
    print(f"Cost       ~${report.cost_usd:.6f}   Duration  {report.duration_ms:.0f}ms")
    if report.cost_by_stage:
        breakdown = "  ".join(f"{stage}=${v:.6f}" for stage, v in report.cost_by_stage.items())
        print(f"By stage   {breakdown}")

    out_dir = Path("evals/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")

    await close_openai_clients()
    return 0  # informational script: exit 0 regardless, numbers speak for themselves in the printed report


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
