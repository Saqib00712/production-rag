import pytest

from app.services.llm import LLMUsage
from app.services.memory_extractor import OpenAIMemoryExtractor


class FakeJsonLLM:
    """Records every call; returns a scripted facts list."""

    def __init__(self, facts: list[str]):
        self.facts = facts
        self.calls: list[dict] = []

    async def complete_json(self, *, model, system, user, schema_name, schema, max_tokens):
        self.calls.append({"model": model, "system": system, "user": user})
        return {"facts": self.facts}, LLMUsage(60, 15)


@pytest.mark.asyncio
async def test_extracts_a_durable_fact_from_the_turn():
    llm = FakeJsonLLM(facts=["Prefers answers in metric units."])
    extractor = OpenAIMemoryExtractor(llm, model="gpt-4o-mini")
    facts, usage = await extractor.extract("Can you convert that to metric?", "Sure, that's 2.5 kg [S1].", existing=[])
    assert facts == ["Prefers answers in metric units."]
    assert "Can you convert that to metric?" in llm.calls[0]["user"]
    assert usage == LLMUsage(60, 15)  # Day 16: the call's usage is surfaced, not discarded


@pytest.mark.asyncio
async def test_most_turns_extract_nothing():
    llm = FakeJsonLLM(facts=[])
    extractor = OpenAIMemoryExtractor(llm, model="gpt-4o-mini")
    facts, _usage = await extractor.extract("What is the warranty?", "12 months [S1].", existing=[])
    assert facts == []


@pytest.mark.asyncio
async def test_blank_facts_are_filtered_out():
    llm = FakeJsonLLM(facts=["  ", "Works in procurement.", ""])
    extractor = OpenAIMemoryExtractor(llm, model="gpt-4o-mini")
    facts, _usage = await extractor.extract("q", "a", existing=[])
    assert facts == ["Works in procurement."]


@pytest.mark.asyncio
async def test_existing_facts_are_passed_to_the_model_for_dedup():
    llm = FakeJsonLLM(facts=[])
    extractor = OpenAIMemoryExtractor(llm, model="gpt-4o-mini")
    await extractor.extract("q", "a", existing=["Works in procurement."])
    assert "Works in procurement." in llm.calls[0]["user"]
