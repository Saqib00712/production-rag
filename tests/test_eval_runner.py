"""
Runner integration test using fake search_fn/ask_fn -- proves the
orchestration and aggregation logic is correct, with zero HTTP, zero
OpenAI, deterministic and instant. Live behavior against real retrieval is
what scripts/verify_day5.py's live run checks separately.
"""

import pytest

from app.eval.answer_metrics import AskLikeResult
from app.eval.models import GoldenQuestion
from app.eval.runner import run_eval

GOLDEN = [
    GoldenQuestion(id="q1", question="answerable, found at rank 1", expected_pages=[1]),
    GoldenQuestion(id="q2", question="answerable, found at rank 3", expected_pages=[2]),
    GoldenQuestion(id="q3", question="answerable, never retrieved", expected_pages=[99]),
    GoldenQuestion(id="q4", question="unanswerable, correctly refused"),
    GoldenQuestion(id="q5", question="unanswerable, wrongly answered"),
    GoldenQuestion(id="q6", question="answerable, wrongly refused (false refusal)", expected_pages=[3]),
    GoldenQuestion(id="q7", question="answerable, answered but wrong page cited (hallucination)", expected_pages=[4]),
]

SEARCH_RESULTS = {
    "q1": [1, 2, 3], "q2": [5, 6, 2], "q3": [1, 2, 3],
    "q4": [7, 8], "q5": [1, 2], "q6": [3, 1], "q7": [4, 1],
}
ASK_RESULTS = {
    "q1": AskLikeResult(refused=False, refusal_reason=None, answer_text="ans [S1]", cited_pages=[1]),
    "q2": AskLikeResult(refused=False, refusal_reason=None, answer_text="ans [S1]", cited_pages=[2]),
    "q3": AskLikeResult(refused=True, refusal_reason="no_results", answer_text="", cited_pages=[]),
    "q4": AskLikeResult(refused=True, refusal_reason="low_relevance", answer_text="", cited_pages=[]),
    "q5": AskLikeResult(refused=False, refusal_reason=None, answer_text="a made-up answer [S1]", cited_pages=[1]),
    "q6": AskLikeResult(refused=True, refusal_reason="low_relevance", answer_text="", cited_pages=[]),
    "q7": AskLikeResult(refused=False, refusal_reason=None, answer_text="wrong page [S1]", cited_pages=[1]),
}


async def fake_search(query, document_id, top_k):
    qid = next(g.id for g in GOLDEN if g.question == query)
    return SEARCH_RESULTS[qid][:top_k]


async def fake_ask(question, document_id):
    qid = next(g.id for g in GOLDEN if g.question == question)
    return ASK_RESULTS[qid]


@pytest.mark.asyncio
async def test_runner_aggregates_correctly():
    report = await run_eval(GOLDEN, "doc-1", fake_search, fake_ask, retrieval_k=5)

    assert report.total_questions == 7
    by_id = {q.question_id: q for q in report.questions}

    assert by_id["q1"].retrieval.hit_at_1 is True and by_id["q1"].answer.is_correct is True
    assert by_id["q2"].retrieval.hit_at_3 is True and by_id["q2"].retrieval.hit_at_1 is False
    assert by_id["q3"].retrieval.mrr == 0.0 and by_id["q3"].answer.is_correct is False  # answerable but never found+refused
    assert by_id["q4"].answer.is_correct is True   # unanswerable + refused = correct
    assert by_id["q5"].answer.is_correct is False  # unanswerable + answered = wrong
    assert by_id["q6"].answer.is_correct is False  # answerable + refused = false refusal
    assert by_id["q7"].answer.correct_citation is False and by_id["q7"].answer.is_correct is False

    # answerable = q1,q2,q3,q6,q7 (5); unanswerable = q4,q5 (2)
    assert report.false_refusal_rate == round(2 / 5, 4)          # q3 (never retrieved) and q6
    assert report.hallucinated_citation_rate == round(1 / 5, 4)  # only q7
    assert report.answer_accuracy == round(3 / 7, 4)             # q1, q2, q4 correct


@pytest.mark.asyncio
async def test_runner_with_judge_fn_records_scores():
    async def judge_fn(question, answer, golden):
        return 7.5

    only_answered = [g for g in GOLDEN if g.id in ("q1", "q2")]
    report = await run_eval(only_answered, "doc-1", fake_search, fake_ask, judge_fn=judge_fn)
    assert report.avg_judge_score == 7.5
    assert all(q.answer.judge_score == 7.5 for q in report.questions)


@pytest.mark.asyncio
async def test_judge_fn_never_called_for_refused_questions():
    calls = []

    async def judge_fn(question, answer, golden):
        calls.append(question)
        return 10.0

    refused_only = [g for g in GOLDEN if g.id == "q4"]
    await run_eval(refused_only, "doc-1", fake_search, fake_ask, judge_fn=judge_fn)
    assert calls == []


@pytest.mark.asyncio
async def test_cost_fn_is_captured_in_report():
    report = await run_eval([GOLDEN[0]], "doc-1", fake_search, fake_ask, cost_fn=lambda: 0.0042)
    assert report.cost_usd == 0.0042


@pytest.mark.asyncio
async def test_cost_by_stage_fn_is_captured_in_report():
    """Day 17: an optional per-stage breakdown alongside the scalar total,
    diffed by the caller (scripts/run_eval.py reads it off the shared
    cost_tracker) -- the runner itself just passes through whatever the
    callable returns at aggregation time."""
    report = await run_eval(
        [GOLDEN[0]], "doc-1", fake_search, fake_ask,
        cost_by_stage_fn=lambda: {"embedding": 0.001, "generation": 0.003},
    )
    assert report.cost_by_stage == {"embedding": 0.001, "generation": 0.003}


@pytest.mark.asyncio
async def test_cost_by_stage_defaults_to_empty_dict():
    report = await run_eval([GOLDEN[0]], "doc-1", fake_search, fake_ask)
    assert report.cost_by_stage == {}
