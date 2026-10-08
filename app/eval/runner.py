"""
Evaluation runner: orchestrates retrieval + answer scoring over a golden
set. Decoupled from FastAPI and from any specific retriever/generator via
two injected async callables -- the same "depend on a Protocol, not a
concrete client" pattern used throughout this project (Embedder, Reranker,
Generator). This is what makes the regression tests in tests/test_eval_*.py
run in milliseconds with zero OpenAI calls, while scripts/run_eval.py wires
the exact same runner to the real SearchService + rag_pipeline.
"""

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from app.eval.answer_metrics import AskLikeResult, check_citation_correct, check_keyword_hit, judge_correctness
from app.eval.models import AnswerResult, EvalReport, GoldenQuestion, QuestionReport, RetrievalResult
from app.eval.retrieval_metrics import compute_page_ranks, hit_at_k, reciprocal_rank, recall_at_k

SearchFn = Callable[[str, str, int], Awaitable[list[int]]]   # (query, document_id, top_k) -> ranked page numbers
AskFn = Callable[[str, str], Awaitable[AskLikeResult]]        # (question, document_id) -> AskLikeResult


class JudgeFn(Protocol):
    async def __call__(self, question: str, answer: str, golden: GoldenQuestion) -> float: ...


@dataclass
class _Row:
    golden: GoldenQuestion
    report: QuestionReport


async def run_eval(
    golden: list[GoldenQuestion],
    document_id: str,
    search_fn: SearchFn,
    ask_fn: AskFn,
    *,
    retrieval_k: int = 5,
    judge_fn: JudgeFn | None = None,
    cost_fn: Callable[[], float] | None = None,
    cost_by_stage_fn: Callable[[], dict[str, float]] | None = None,
) -> EvalReport:
    t0 = time.perf_counter()
    rows: list[_Row] = []

    for g in golden:
        hit_pages = await search_fn(g.question, document_id, retrieval_k)
        page_ranks = compute_page_ranks(hit_pages, g.expected_pages)
        retrieval = RetrievalResult(
            question_id=g.id, hit_pages=hit_pages, page_ranks=page_ranks,
            hit_at_1=hit_at_k(page_ranks, 1), hit_at_3=hit_at_k(page_ranks, 3), hit_at_5=hit_at_k(page_ranks, 5),
            mrr=reciprocal_rank(page_ranks), recall_at_5=recall_at_k(page_ranks, g.expected_pages, 5),
        )

        ask_result = await ask_fn(g.question, document_id)
        correct_citation = check_citation_correct(g, ask_result)
        keyword_hit = check_keyword_hit(g, ask_result)
        is_correct = judge_correctness(g, ask_result, correct_citation)

        judge_score: float | None = None
        if judge_fn is not None and not ask_result.refused:
            judge_score = await judge_fn(g.question, ask_result.answer_text, g)

        answer = AnswerResult(
            question_id=g.id, refused=ask_result.refused, refusal_reason=ask_result.refusal_reason,
            correct_citation=correct_citation, keyword_hit=keyword_hit, judge_score=judge_score,
            is_correct=is_correct,
        )
        rows.append(_Row(golden=g, report=QuestionReport(
            question_id=g.id, question=g.question, retrieval=retrieval, answer=answer,
        )))

    return _aggregate(document_id, rows, time.perf_counter() - t0, cost_fn, cost_by_stage_fn)


def _aggregate(document_id: str, rows: list[_Row], elapsed_s: float,
               cost_fn: Callable[[], float] | None,
               cost_by_stage_fn: Callable[[], dict[str, float]] | None = None) -> EvalReport:
    n = len(rows)
    reports = [r.report for r in rows]

    def rate(pred: Callable[[_Row], bool]) -> float:
        return round(sum(1 for r in rows if pred(r)) / n, 4) if n else 0.0

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    answerable = [r for r in rows if not r.golden.is_unanswerable]
    # False refusal: the question WAS answerable, but the system refused anyway.
    false_refusals = [r for r in answerable if r.report.answer.refused]
    # Hallucinated citation: the system answered (didn't refuse) but cited
    # a page that doesn't actually support the answer.
    hallucinations = [r for r in answerable if not r.report.answer.refused and r.report.answer.correct_citation is False]
    judge_scores = [r.report.answer.judge_score for r in rows if r.report.answer.judge_score is not None]

    return EvalReport(
        document_id=document_id, total_questions=n,
        retrieval_hit_at_1=rate(lambda r: r.report.retrieval.hit_at_1),
        retrieval_hit_at_3=rate(lambda r: r.report.retrieval.hit_at_3),
        retrieval_hit_at_5=rate(lambda r: r.report.retrieval.hit_at_5),
        retrieval_mrr=avg([r.report.retrieval.mrr for r in rows]),
        retrieval_recall_at_5=avg([r.report.retrieval.recall_at_5 for r in answerable]),
        answer_accuracy=rate(lambda r: r.report.answer.is_correct),
        false_refusal_rate=round(len(false_refusals) / len(answerable), 4) if answerable else 0.0,
        hallucinated_citation_rate=round(len(hallucinations) / len(answerable), 4) if answerable else 0.0,
        avg_judge_score=avg(judge_scores) if judge_scores else None,
        cost_usd=cost_fn() if cost_fn else 0.0,
        cost_by_stage=cost_by_stage_fn() if cost_by_stage_fn else {},
        duration_ms=round(elapsed_s * 1000, 2),
        questions=reports,
    )
