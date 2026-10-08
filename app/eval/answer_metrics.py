"""
Pure answer-quality scoring against a golden question. No I/O.

The pass/fail logic (`judge_correctness`) is deliberately simple and
deterministic -- it is the "did the system do the RIGHT THING" check, not a
measure of prose quality:
  * unanswerable question + refused           -> correct (this IS success)
  * unanswerable question + answered anyway   -> WRONG (a hallucination)
  * answerable question + refused             -> WRONG (a false refusal;
                                                  usually costs more in
                                                  production than a wrong
                                                  answer, because users lose
                                                  trust in ANY answer)
  * answerable question + correct-page cited  -> correct
  * answerable question + wrong-page cited    -> WRONG
An LLM judge score (0-10, optional) is recorded alongside this but never
substitutes for it -- correctness here is about verifiable citations, not a
model's opinion of its own fluency.
"""

from dataclasses import dataclass

from app.eval.models import GoldenQuestion


@dataclass
class AskLikeResult:
    """Minimal shape the eval runner needs from an /ask call -- decouples
    this module from AskResponse/FastAPI entirely."""

    refused: bool
    refusal_reason: str | None
    answer_text: str
    cited_pages: list[int]


def check_citation_correct(golden: GoldenQuestion, result: AskLikeResult) -> bool | None:
    if golden.is_unanswerable:
        return None  # not applicable: correctness here is about refusal, not citation
    return any(p in golden.expected_pages for p in result.cited_pages)


def check_keyword_hit(golden: GoldenQuestion, result: AskLikeResult) -> bool | None:
    if not golden.expected_keywords or result.refused:
        return None
    text = result.answer_text.lower()
    return any(kw.lower() in text for kw in golden.expected_keywords)


def judge_correctness(golden: GoldenQuestion, result: AskLikeResult, correct_citation: bool | None) -> bool:
    if golden.is_unanswerable:
        return result.refused
    if result.refused:
        return False
    return bool(correct_citation)
