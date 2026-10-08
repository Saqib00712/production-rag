"""Data contracts for the evaluation harness.

Design decision: a golden question can be UNANSWERABLE by design
(`expected_pages = []`). This lets the eval set measure both directions of
error a RAG system can make: missing a real answer, AND confidently
answering a question the documents don't cover. Most eval harnesses only
test the first; the second (false answering) is usually the more dangerous
failure in production.
"""

from pydantic import BaseModel, Field, field_validator


class GoldenQuestion(BaseModel):
    id: str
    question: str
    # Empty list = this question should be REFUSED (no evidence exists).
    expected_pages: list[int] = Field(default_factory=list)
    # Optional lexical sanity check on the answer text. Weak signal (word
    # presence, not meaning) -- a stand-in until Day 6+ adds semantic
    # scoring (RAGAS/DeepEval); flagged here deliberately, not hidden.
    expected_keywords: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def is_unanswerable(self) -> bool:
        return len(self.expected_pages) == 0

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class RetrievalResult(BaseModel):
    question_id: str
    hit_pages: list[int]              # ranked page numbers, best first
    page_ranks: dict[int, int]        # expected page -> earliest rank found
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    mrr: float
    recall_at_5: float


class AnswerResult(BaseModel):
    question_id: str
    refused: bool
    refusal_reason: str | None = None
    correct_citation: bool | None = None   # None only when not applicable (unanswerable + correctly refused)
    keyword_hit: bool | None = None
    judge_score: float | None = None       # 0-10, optional LLM-as-judge score
    is_correct: bool                       # the single pass/fail verdict for this question


class QuestionReport(BaseModel):
    question_id: str
    question: str
    retrieval: RetrievalResult
    answer: AnswerResult


class EvalReport(BaseModel):
    document_id: str
    total_questions: int
    retrieval_hit_at_1: float
    retrieval_hit_at_3: float
    retrieval_hit_at_5: float
    retrieval_mrr: float
    retrieval_recall_at_5: float
    answer_accuracy: float             # fraction of questions with is_correct=True
    false_refusal_rate: float          # answerable questions the system wrongly refused
    hallucinated_citation_rate: float  # answered (not refused) but cited the wrong page
    avg_judge_score: float | None = None
    cost_usd: float = 0.0
    # Day 17: same per-stage breakdown /observability/summary exposes
    # (app.observability.metrics.CostTracker.cost_by_stage), diffed over
    # just this eval run -- see scripts/run_eval.py's cost_by_stage_fn.
    # Empty unless the caller supplies one; cost_usd alone still works.
    cost_by_stage: dict[str, float] = {}
    duration_ms: float = 0.0
    questions: list[QuestionReport] = []
