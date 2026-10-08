"""
Data-driven hyperparameter tuning for chunk size, retrieve_k, and the
rerank threshold -- using scripts/run_eval.py's own golden-set machinery,
against YOUR real document and questions.

Usage:
    python scripts/tune.py --pdf your.pdf --golden evals/golden/your_set.json
    python scripts/tune.py --pdf your.pdf --golden evals/golden/your_set.json --skip-threshold  # cheapest run

What it does (see docs/day6.md for the full reasoning):
  1. CHUNK SIZE sweep: re-indexes your PDF at a few chunk sizes (in ISOLATED
     temp storage -- your real indexed data is never touched) and measures
     retrieval-only hit@k/MRR for each. No rerank/generation calls: this
     isolates the effect of chunking from everything downstream.
  2. RETRIEVE_K sweep: re-uses ONE index (no re-embedding) and just asks for
     more or fewer candidates per query, showing the cheapest k that still
     reaches good retrieval.
  3. RERANK THRESHOLD analysis: runs the REAL /ask pipeline exactly ONCE per
     question (this is the only step that spends on rerank+generation), then
     replays the recorded scores against several candidate thresholds in
     plain Python -- zero extra API calls for this step.
Writes a full report to evals/reports/tune-<timestamp>.json.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOG_LEVEL", "WARNING")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[200, 350, 500])
    ap.add_argument("--retrieve-ks", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0, 6.0])
    ap.add_argument("--skip-threshold", action="store_true", help="skip the rerank-threshold step (saves the /ask calls)")
    args = ap.parse_args()

    from app.config import Settings, get_settings
    from app.dependencies import get_embedder_factory, get_generator_factory, get_json_llm_factory, get_reranker_factory, get_repository
    from app.eval.loader import GoldenSetError, load_golden_set
    from app.eval.retrieval_metrics import compute_page_ranks, hit_at_k, reciprocal_rank
    from app.eval.tuning import (
        ChunkSweepPoint, QuestionScore, RetrieveKSweepPoint,
        best_chunk_point, recommend_threshold, simulate_thresholds, smallest_k_reaching,
    )
    from app.repositories.chunk_repository import ChunkRepository
    from app.services.chunker import TokenCounter
    from app.services.ingestion import index_document
    from app.services.pdf_parser import parse_pdf
    from app.services.rag_pipeline import RagDeps, answer_question
    from app.services.search import SearchService

    base_settings = get_settings()
    if not base_settings.openai_api_key or base_settings.openai_api_key.startswith("sk-your"):
        print("ERROR: OPENAI_API_KEY is not set in .env. This script needs a real key.")
        return 1

    try:
        golden = load_golden_set(args.golden)
    except GoldenSetError as exc:
        print(f"ERROR: {exc}")
        return 1
    answerable = [g for g in golden if not g.is_unanswerable]
    if not answerable:
        print("ERROR: golden set has no answerable questions (all expected_pages are empty) -- nothing to tune retrieval against.")
        return 1
    print(f"Loaded {len(golden)} questions ({len(answerable)} answerable) from {args.golden}")
    pdf_bytes = Path(args.pdf).read_bytes()
    pdf_name = Path(args.pdf).name

    # ---------------- 1) chunk size sweep ----------------
    print(f"\n=== Chunk size sweep: {args.chunk_sizes} ===")
    chunk_points: list[ChunkSweepPoint] = []
    for size in args.chunk_sizes:
        overlap = max(1, round(size * 0.15))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            s = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/t.db",
                         chunk_size_tokens=size, chunk_overlap_tokens=overlap)
            repo = ChunkRepository(s.sqlite_path)
            counter = TokenCounter()
            embedder_factory = get_embedder_factory(s)
            request_id = str(uuid.uuid4())
            parsed = parse_pdf(pdf_bytes, pdf_name, "application/pdf", request_id)
            idx = await index_document(parsed, repo=repo, embedder_factory=embedder_factory, counter=counter,
                                        settings=s, request_id=request_id)
            search = SearchService(repo, embedder_factory, s)
            ranks_per_q = []
            for g in answerable:
                r = await search.search(query=g.question, mode="hybrid", top_k=5, document_id=parsed.document_id,
                                         request_id=str(uuid.uuid4()))
                pages = [h.page_number for h in r.hits]
                ranks = compute_page_ranks(pages, g.expected_pages)
                ranks_per_q.append(ranks)
            n = len(answerable)
            point = ChunkSweepPoint(
                chunk_size=size, overlap=overlap,
                hit_at_1=sum(hit_at_k(r, 1) for r in ranks_per_q) / n,
                hit_at_3=sum(hit_at_k(r, 3) for r in ranks_per_q) / n,
                hit_at_5=sum(hit_at_k(r, 5) for r in ranks_per_q) / n,
                mrr=sum(reciprocal_rank(r) for r in ranks_per_q) / n,
                recall_at_5=sum(hit_at_k(r, 5) for r in ranks_per_q) / n,
                chunk_count=idx.chunk_count, index_tokens=idx.tokens, index_cost_usd=idx.cost_usd,
            )
            chunk_points.append(point)
            print(f"  size={size:>4} overlap={overlap:>3}  chunks={point.chunk_count:>3}  "
                  f"hit@1={point.hit_at_1:.0%}  hit@3={point.hit_at_3:.0%}  MRR={point.mrr:.3f}  "
                  f"(~${point.index_cost_usd:.6f})")

    best_chunks = best_chunk_point(chunk_points)
    current_note = " (current default)" if best_chunks.chunk_size == base_settings.chunk_size_tokens else ""
    print(f"  -> best: chunk_size={best_chunks.chunk_size} (hit@3={best_chunks.hit_at_3:.0%}, MRR={best_chunks.mrr:.3f}){current_note}")

    # ---------------- 2) retrieve_k sweep (re-uses ONE index at the best chunk size) ----------------
    print(f"\n=== retrieve_k sweep: {args.retrieve_ks} (using chunk_size={best_chunks.chunk_size}) ===")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        s = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/t.db",
                     chunk_size_tokens=best_chunks.chunk_size, chunk_overlap_tokens=best_chunks.overlap)
        repo = ChunkRepository(s.sqlite_path)
        counter = TokenCounter()
        embedder_factory = get_embedder_factory(s)
        request_id = str(uuid.uuid4())
        parsed = parse_pdf(pdf_bytes, pdf_name, "application/pdf", request_id)
        await index_document(parsed, repo=repo, embedder_factory=embedder_factory, counter=counter,
                              settings=s, request_id=request_id)
        search = SearchService(repo, embedder_factory, s)
        max_k = max(args.retrieve_ks)
        all_hits: dict[str, list[int]] = {}
        for g in answerable:
            r = await search.search(query=g.question, mode="hybrid", top_k=max_k, document_id=parsed.document_id,
                                     request_id=str(uuid.uuid4()))
            all_hits[g.id] = [h.page_number for h in r.hits]  # query embedding is cached -> re-slicing k is free

        k_points: list[RetrieveKSweepPoint] = []
        for k in sorted(args.retrieve_ks):
            n = len(answerable)
            ranks_per_q = [compute_page_ranks(all_hits[g.id][:k], g.expected_pages) for g in answerable]
            point = RetrieveKSweepPoint(
                k=k, hit_at_1=sum(hit_at_k(r, 1) for r in ranks_per_q) / n,
                hit_at_3=sum(hit_at_k(r, 3) for r in ranks_per_q) / n,
                hit_at_5=sum(hit_at_k(r, 5) for r in ranks_per_q) / n,
                mrr=sum(reciprocal_rank(r) for r in ranks_per_q) / n,
            )
            k_points.append(point)
            print(f"  k={point.k:>3}  hit@1={point.hit_at_1:.0%}  hit@3={point.hit_at_3:.0%}  MRR={point.mrr:.3f}")

        target = max(p.hit_at_3 for p in k_points)
        best_k = smallest_k_reaching(k_points, target_hit_at_3=target)
        print(f"  -> smallest k reaching the best hit@3 ({target:.0%}): retrieve_k={best_k.k}")

        # ---------------- 3) rerank threshold analysis (real /ask calls, once each) ----------------
        threshold_results = None
        if not args.skip_threshold:
            print(f"\n=== Rerank threshold analysis (one real /ask call per question) ===")
            llm_factory = get_json_llm_factory(s)
            reranker_factory = get_reranker_factory(s, llm_factory)
            generator_factory = get_generator_factory(s, llm_factory)
            deps = RagDeps(search=search, reranker=reranker_factory(), generator=generator_factory(),
                            counter=counter, settings=s)
            scores: list[QuestionScore] = []
            live_cost = 0.0
            for g in golden:
                resp = await answer_question(question=g.question, document_id=parsed.document_id, top_k=5,
                                              use_rerank=True, deps=deps, request_id=str(uuid.uuid4()))
                live_cost += resp.cost.total_cost_usd
                top_score = max((c.rerank_score or 0.0) for c in resp.citations) if resp.citations else 0.0
                was_correct = (not resp.insufficient_evidence) and any(
                    c.page_number in g.expected_pages for c in resp.citations
                ) if not g.is_unanswerable else resp.insufficient_evidence
                scores.append(QuestionScore(question_id=g.id, rerank_score=top_score,
                                             was_correct=was_correct, currently_refused=resp.insufficient_evidence))
                print(f"  {g.id:<5} refused={str(resp.insufficient_evidence):<5} top_rerank_score={top_score:.1f}  correct_now={was_correct}")

            threshold_results = simulate_thresholds(scores, args.thresholds)
            print(f"\n  {'threshold':<10} {'lose':<6} {'block':<6} {'net':<5}")
            for r in threshold_results:
                print(f"  {r.threshold:<10} {len(r.correct_answers_that_would_be_lost):<6} "
                      f"{len(r.bad_answers_that_would_be_blocked):<6} {r.net_score:<5}")
            rec = recommend_threshold(threshold_results, current=s.min_rerank_score)
            print(f"  -> recommended min_rerank_score={rec.threshold} (net={rec.net_score}, current={s.min_rerank_score})")
            print(f"  Cost of this step: ~${live_cost:.6f}")

    report = {
        "pdf": args.pdf, "golden": args.golden, "current_defaults": {
            "chunk_size_tokens": base_settings.chunk_size_tokens, "chunk_overlap_tokens": base_settings.chunk_overlap_tokens,
            "retrieve_k": base_settings.retrieve_k, "min_rerank_score": base_settings.min_rerank_score,
        },
        "chunk_sweep": [vars(p) for p in chunk_points],
        "recommended_chunk_size": best_chunks.chunk_size,
        "retrieve_k_sweep": [vars(p) for p in k_points],
        "recommended_retrieve_k": best_k.k,
        "threshold_results": [vars(r) for r in threshold_results] if threshold_results else None,
        "recommended_min_rerank_score": rec.threshold if threshold_results else None,
    }
    out_dir = Path("evals/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tune-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out_path}")
    print("\nTo apply a recommendation, update the matching value in .env (see .env.example) and re-run your eval.")

    from app.services.openai_client import close_openai_clients
    await close_openai_clients()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
