import pytest

from app.services.llm import LLMUsage
from app.services.reranker import OpenAIReranker


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.last_user = None

    async def complete_json(self, *, model, system, user, schema_name, schema, max_tokens):
        self.last_user = user
        return self.reply, LLMUsage(50, 10)


@pytest.mark.asyncio
async def test_maps_labels_back_to_real_chunk_ids():
    llm = FakeLLM({"scores": [{"id": "c1", "score": 9}, {"id": "c2", "score": 2}]})
    result = await OpenAIReranker(llm, "m").rerank("q", [("real-chunk-1", "t1"), ("real-chunk-2", "t2")])
    assert result.scores == {"real-chunk-1": 9.0, "real-chunk-2": 2.0}


@pytest.mark.asyncio
async def test_missing_candidate_defaults_to_zero_fail_closed():
    llm = FakeLLM({"scores": [{"id": "c1", "score": 9}]})  # c2 never scored
    result = await OpenAIReranker(llm, "m").rerank("q", [("a", "t"), ("b", "t")])
    assert result.scores["b"] == 0.0


@pytest.mark.asyncio
async def test_hallucinated_id_is_ignored_not_crashing():
    llm = FakeLLM({"scores": [{"id": "c1", "score": 9}, {"id": "c99", "score": 5}]})
    result = await OpenAIReranker(llm, "m").rerank("q", [("a", "t")])
    assert result.scores == {"a": 9.0}


@pytest.mark.asyncio
async def test_scores_are_clamped_to_0_10():
    llm = FakeLLM({"scores": [{"id": "c1", "score": 999}, {"id": "c2", "score": -5}]})
    result = await OpenAIReranker(llm, "m").rerank("q", [("a", "t"), ("b", "t")])
    assert result.scores["a"] == 10.0 and result.scores["b"] == 0.0


@pytest.mark.asyncio
async def test_real_chunk_ids_never_sent_to_the_model():
    llm = FakeLLM({"scores": []})
    await OpenAIReranker(llm, "m").rerank("q", [("super-secret-real-id-123", "t")])
    assert "super-secret-real-id-123" not in llm.last_user
