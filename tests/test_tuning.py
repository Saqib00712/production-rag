import pytest

from app.eval.tuning import (
    ChunkSweepPoint, QuestionScore, RetrieveKSweepPoint,
    best_chunk_point, recommend_threshold, simulate_thresholds, smallest_k_reaching,
)


def cp(size, hit1, hit3, mrr, hit5=None):
    return ChunkSweepPoint(chunk_size=size, overlap=int(size * 0.15), hit_at_1=hit1, hit_at_3=hit3,
                           hit_at_5=hit5 if hit5 is not None else hit3, mrr=mrr, recall_at_5=hit3,
                           chunk_count=10, index_tokens=1000, index_cost_usd=0.00002)


def test_best_chunk_point_prefers_higher_hit_at_3():
    points = [cp(200, 0.5, 0.6, 0.5), cp(350, 0.7, 0.9, 0.8), cp(500, 0.6, 0.8, 0.7)]
    assert best_chunk_point(points).chunk_size == 350


def test_best_chunk_point_breaks_ties_with_mrr():
    points = [cp(200, 0.5, 0.8, 0.6), cp(350, 0.5, 0.8, 0.9)]
    assert best_chunk_point(points).chunk_size == 350


def test_best_chunk_point_empty_raises():
    with pytest.raises(ValueError):
        best_chunk_point([])


def rk(k, hit3):
    return RetrieveKSweepPoint(k=k, hit_at_1=hit3, hit_at_3=hit3, hit_at_5=hit3, mrr=hit3)


def test_smallest_k_reaching_target_picks_cheapest_sufficient_k():
    points = [rk(5, 0.7), rk(10, 0.9), rk(15, 0.95), rk(20, 0.95)]
    assert smallest_k_reaching(points, target_hit_at_3=0.9).k == 10


def test_smallest_k_reaching_target_none_when_unreachable():
    points = [rk(5, 0.5), rk(10, 0.6)]
    assert smallest_k_reaching(points, target_hit_at_3=0.9) is None


def qs(qid, score, correct, refused=False):
    return QuestionScore(question_id=qid, rerank_score=score, was_correct=correct, currently_refused=refused)


def test_simulate_thresholds_identifies_lost_and_blocked():
    scores = [
        qs("q1", 8.0, correct=True),   # high score, correct -- never at risk
        qs("q2", 3.0, correct=True),   # low score but WAS correct -- raising threshold LOSES this
        qs("q3", 2.0, correct=False),  # low score, wrong -- raising threshold BLOCKS this (a win)
    ]
    results = simulate_thresholds(scores, thresholds=[0.0, 4.0])
    at_zero = next(r for r in results if r.threshold == 0.0)
    assert at_zero.correct_answers_that_would_be_lost == [] and at_zero.bad_answers_that_would_be_blocked == []
    at_four = next(r for r in results if r.threshold == 4.0)
    assert at_four.correct_answers_that_would_be_lost == ["q2"]
    assert at_four.bad_answers_that_would_be_blocked == ["q3"]
    assert at_four.net_score == 0  # one lost, one blocked


def test_simulate_thresholds_skips_already_refused_questions():
    scores = [qs("q1", 1.0, correct=False, refused=True)]
    results = simulate_thresholds(scores, thresholds=[5.0])
    assert results[0].correct_answers_that_would_be_lost == []
    assert results[0].bad_answers_that_would_be_blocked == []  # already refused, not newly blocked


def test_recommend_threshold_picks_best_net_score():
    results = simulate_thresholds(
        [qs("q1", 3.0, correct=True), qs("q2", 2.0, correct=False), qs("q3", 1.0, correct=False)],
        thresholds=[0.0, 2.5, 4.0],
    )
    rec = recommend_threshold(results, current=4.0)
    assert rec.threshold == 2.5  # blocks q3 only, loses nothing -> net +1, the best


def test_recommend_threshold_ties_prefer_closest_to_current():
    results = simulate_thresholds([qs("q1", 5.0, correct=True)], thresholds=[1.0, 2.0, 9.0])
    # none of these thresholds lose or block anything -> all net_score 0 -> tie
    rec = recommend_threshold(results, current=2.0)
    assert rec.threshold == 2.0


def test_recommend_threshold_empty_raises():
    with pytest.raises(ValueError):
        recommend_threshold([], current=4.0)
