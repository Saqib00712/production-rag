import pytest

from app.eval.models import GoldenQuestion
from app.services.judge import OpenAIJudge
from app.services.llm import LLMUsage


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.last_user = reply, None

    async def complete_json(self, *, model, system, user, schema_name, schema, max_tokens):
        self.last_user = user
        return self.reply, LLMUsage(30, 10)


def golden(keywords=("refund", "30 days")):
    return GoldenQuestion(id="q1", question="How long for a refund?", expected_keywords=list(keywords))


@pytest.mark.asyncio
async def test_score_parses_and_clamps():
    judge = OpenAIJudge(FakeLLM({"score": 8, "reasoning": "mentions both"}), "m")
    assert await judge.score("q", "a", golden()) == 8.0

    judge = OpenAIJudge(FakeLLM({"score": 99, "reasoning": "x"}), "m")
    assert await judge.score("q", "a", golden()) == 10.0

    judge = OpenAIJudge(FakeLLM({"score": -3, "reasoning": "x"}), "m")
    assert await judge.score("q", "a", golden()) == 0.0


@pytest.mark.asyncio
async def test_malformed_reply_scores_zero_not_crash():
    judge = OpenAIJudge(FakeLLM({"oops": "no score field"}), "m")
    assert await judge.score("q", "a", golden()) == 0.0


@pytest.mark.asyncio
async def test_reference_keywords_are_included_in_prompt():
    llm = FakeLLM({"score": 5, "reasoning": "x"})
    judge = OpenAIJudge(llm, "m")
    await judge.score("How long?", "30 days", golden(keywords=["thirty", "days"]))
    assert "thirty" in llm.last_user and "days" in llm.last_user


@pytest.mark.asyncio
async def test_callable_interface_matches_judge_fn_protocol():
    judge = OpenAIJudge(FakeLLM({"score": 6, "reasoning": "x"}), "m")
    assert await judge("q", "a", golden()) == 6.0
