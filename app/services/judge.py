"""
Optional LLM-as-judge: scores answer quality 0-10 against a reference.

This is a SOFT signal, recorded alongside the deterministic citation-
correctness check in answer_metrics.py, never a substitute for it. It
exists to cover "Answer evaluation" more richly than a keyword check can --
full semantic evaluation frameworks (RAGAS, DeepEval) are the production
upgrade path (see docs/day5.md Production Improvements) once volume
justifies their extra dependencies and setup.
"""

from typing import Protocol

from app.eval.models import GoldenQuestion
from app.services.llm import JsonLLM

SYSTEM = (
    "You grade whether an AI-generated answer correctly addresses a question, given what the correct "
    "answer should mention. Score 0 (wrong or irrelevant) to 10 (fully correct and complete). "
    "Be strict: partial or vague answers score in the middle, not high."
)
SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "reasoning": {"type": "string"}},
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


class Judge(Protocol):
    async def score(self, question: str, answer: str, golden: GoldenQuestion) -> float: ...


class OpenAIJudge:
    def __init__(self, llm: JsonLLM, model: str):
        self._llm, self._model = llm, model

    async def score(self, question: str, answer: str, golden: GoldenQuestion) -> float:
        reference = ", ".join(golden.expected_keywords) or "(no reference keywords provided)"
        data, _usage = await self._llm.complete_json(
            model=self._model, system=SYSTEM,
            user=f"Question: {question}\n\nExpected answer should mention: {reference}\n\nAI answer: {answer}",
            schema_name="judge_score", schema=SCHEMA, max_tokens=150,
        )
        try:
            return float(min(10, max(0, data["score"])))
        except (KeyError, TypeError, ValueError):
            return 0.0

    async def __call__(self, question: str, answer: str, golden: GoldenQuestion) -> float:
        return await self.score(question, answer, golden)
