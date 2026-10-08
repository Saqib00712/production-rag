import pytest

from app.models.conversation import Turn
from app.services.contextualizer import OpenAIContextualizer
from app.services.llm import LLMUsage


class FakeJsonLLM:
    """Records every call; returns a scripted rewrite."""

    def __init__(self, rewrite: str):
        self.rewrite = rewrite
        self.calls: list[dict] = []

    async def complete_json(self, *, model, system, user, schema_name, schema, max_tokens):
        self.calls.append({"model": model, "system": system, "user": user})
        return {"standalone_query": self.rewrite}, LLMUsage(50, 10)


def _turn(question: str, answer: str, idx: int = 0) -> Turn:
    return Turn(turn_index=idx, question=question, search_query=question, answer=answer,
                citations=[], refusal_reason=None, created_at=0.0)


@pytest.mark.asyncio
async def test_first_turn_never_calls_the_llm():
    llm = FakeJsonLLM(rewrite="should never be used")
    ctx = OpenAIContextualizer(llm, model="gpt-4o-mini")
    result, usage = await ctx.contextualize("What is the warranty?", history=[])
    assert result == "What is the warranty?"
    assert llm.calls == []
    assert usage == LLMUsage()  # Day 16: the skip path reports zero usage, not whatever a real call would cost


@pytest.mark.asyncio
async def test_followup_with_history_is_rewritten_via_the_llm():
    llm = FakeJsonLLM(rewrite="What is the shipping policy?")
    ctx = OpenAIContextualizer(llm, model="gpt-4o-mini")
    history = [_turn("What is the warranty?", "12 months.")]
    result, usage = await ctx.contextualize("What about the second one?", history=history)
    assert result == "What is the shipping policy?"
    assert len(llm.calls) == 1
    assert "What is the warranty?" in llm.calls[0]["user"]
    assert "What about the second one?" in llm.calls[0]["user"]
    assert usage == LLMUsage(50, 10)  # Day 16: the real call's usage is surfaced, not discarded


@pytest.mark.asyncio
async def test_empty_rewrite_falls_back_to_the_original_question():
    llm = FakeJsonLLM(rewrite="   ")
    ctx = OpenAIContextualizer(llm, model="gpt-4o-mini")
    history = [_turn("What is the warranty?", "12 months.")]
    result, _usage = await ctx.contextualize("and after that?", history=history)
    assert result == "and after that?"


@pytest.mark.asyncio
async def test_only_the_most_recent_max_turns_are_sent():
    llm = FakeJsonLLM(rewrite="irrelevant")
    ctx = OpenAIContextualizer(llm, model="gpt-4o-mini", max_turns=1)
    history = [_turn("first question", "first answer", 0), _turn("second question", "second answer", 1)]
    await ctx.contextualize("a follow-up", history=history)  # tuple return not needed for this assertion
    user_text = llm.calls[0]["user"]
    assert "second question" in user_text
    assert "first question" not in user_text
