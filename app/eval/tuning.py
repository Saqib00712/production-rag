"""
Pure hyperparameter-tuning logic: no I/O, no OpenAI calls here at all.
scripts/tune.py gathers real numbers (chunking + re-indexing, or replaying
already-paid-for rerank scores); everything in this file just SUMMARIZES
and RECOMMENDS from data already collected, which is what makes it fully
unit-testable without spending anything.

Design principle worth stating explicitly: we tune ONE hyperparameter at a
time (chunk size, then retrieve_k, then rerank threshold), holding the
others fixed, rather than a full grid search. A 3x3x5 grid would need 45
re-indexes and hundreds of paid LLM calls to answer a question that three
small, focused sweeps answer for a fraction of the cost and are far easier
to reason about ("hit@3 rose from 71% to 89% when chunk size went from 200
to 350" is a clear story; a 45-cell grid is not).
"""

from dataclasses import dataclass


@dataclass
class ChunkSweepPoint:
    chunk_size: int
    overlap: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    recall_at_5: float
    chunk_count: int
    index_tokens: int
    index_cost_usd: float


def best_chunk_point(points: list[ChunkSweepPoint]) -> ChunkSweepPoint:
    """hit@3 is the primary signal (matches what /ask actually uses as
    candidate depth before rerank); MRR breaks ties (rewards ranking the
    answer earlier, which matters once rerank is added on top)."""
    if not points:
        raise ValueError("no sweep points to compare")
    return max(points, key=lambda p: (p.hit_at_3, p.mrr))


@dataclass
class RetrieveKSweepPoint:
    k: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float


def smallest_k_reaching(points: list[RetrieveKSweepPoint], target_hit_at_3: float) -> RetrieveKSweepPoint | None:
    """The cheapest retrieve_k that still reaches a target hit@3 -- every
    extra candidate costs rerank tokens downstream, so bigger is not free."""
    candidates = [p for p in points if p.hit_at_3 >= target_hit_at_3]
    return min(candidates, key=lambda p: p.k) if candidates else None


@dataclass
class QuestionScore:
    question_id: str
    rerank_score: float          # the top candidate's score, from one real /ask call
    was_correct: bool            # was the ACTUAL answer (at the CURRENT threshold) correct?
    currently_refused: bool


@dataclass
class ThresholdSimResult:
    threshold: float
    correct_answers_that_would_be_lost: list[str]    # were correct, would newly refuse
    bad_answers_that_would_be_blocked: list[str]      # were wrong, would newly refuse (a win)
    net_score: int                                    # blocked_bad - lost_correct; higher is better


def simulate_thresholds(scores: list[QuestionScore], thresholds: list[float]) -> list[ThresholdSimResult]:
    """Replays ALREADY-COLLECTED rerank scores against candidate thresholds
    with no new LLM calls: raising the threshold only ever turns an
    "answered" question into a "refused" one (it can't un-refuse an
    already-refused question), so we only need to check questions that were
    NOT already refused."""
    results = []
    for t in thresholds:
        lost, blocked = [], []
        for s in scores:
            if s.currently_refused:
                continue
            if s.rerank_score < t:
                (lost if s.was_correct else blocked).append(s.question_id)
        results.append(ThresholdSimResult(
            threshold=t, correct_answers_that_would_be_lost=lost,
            bad_answers_that_would_be_blocked=blocked, net_score=len(blocked) - len(lost),
        ))
    return results


def recommend_threshold(results: list[ThresholdSimResult], current: float) -> ThresholdSimResult:
    """Prefer the threshold with the best net_score; among ties, prefer the
    one closest to the current setting (smaller change = lower risk)."""
    if not results:
        raise ValueError("no threshold results to compare")
    best_score = max(r.net_score for r in results)
    tied = [r for r in results if r.net_score == best_score]
    return min(tied, key=lambda r: abs(r.threshold - current))
