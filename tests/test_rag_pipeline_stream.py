"""
Unit tests for answer_question_stream, using fake search/reranker/streaming-
generator deps directly (no HTTP) -- mirrors test_rag_pipeline.py's style.
"""

import pytest

from app.config import Settings
from app.services.chunker import TokenCounter
from app.services.rag_pipeline import RagDeps, answer_question_stream
from tests.fakes import FailingStreamingGenerator, FakeStreamingGenerator, HashEmbedder, ScriptedRerankerFactory


class FakeSearchService:
    """Minimal stand-in for SearchService returning fixed hits."""

    def __init__(self, hits):
        self._hits = hits

    async def search(self, *, query, mode, top_k, document_id, tenant_id, request_id):
        from app.models.search import SearchResponse

        return SearchResponse(query=query, mode=mode, score_type="rrf", top_k=top_k,
                               document_id=document_id, hits=self._hits[:top_k])


def hit(cid="c1", page=1, text="Refunds take 30 days.", score=0.5):
    from app.models.search import SearchHit

    return SearchHit(rank=1, chunk_id=cid, document_id="d1", page_number=page, text=text, score=score)


def make_deps(hits, streaming_generator, reranker=None):
    return RagDeps(
        search=FakeSearchService(hits), reranker=reranker, generator=None,
        streaming_generator=streaming_generator, counter=TokenCounter(use_tiktoken=False),
        settings=Settings(openai_api_key="test", _env_file=None),
    )


async def collect(gen):
    return [e async for e in gen]


@pytest.mark.asyncio
async def test_streams_deltas_then_a_done_event_with_citations():
    sg = FakeStreamingGenerator(["The answer ", "is 30 days [S1]."])
    deps = make_deps([hit()], sg, reranker=ScriptedRerankerFactory()())
    events = await collect(answer_question_stream(
        question="q", document_id="d1", tenant_id="default", top_k=5, use_rerank=True, deps=deps, request_id="r1",
    ))
    deltas = [e["text"] for e in events if e["event"] == "delta"]
    assert deltas == ["The answer ", "is 30 days [S1]."]
    done = events[-1]
    assert done["event"] == "done"
    assert done["data"]["insufficient_evidence"] is False
    assert len(done["data"]["citations"]) == 1
    assert done["data"]["citations"][0]["page_number"] == 1


@pytest.mark.asyncio
async def test_no_search_hits_refuses_immediately_with_no_deltas():
    sg = FakeStreamingGenerator(["should never be used"])
    deps = make_deps([], sg)
    events = await collect(answer_question_stream(
        question="q", document_id="d1", tenant_id="default", top_k=5, use_rerank=False, deps=deps, request_id="r1",
    ))
    assert len(events) == 1 and events[0]["event"] == "done"
    assert events[0]["data"]["refusal_reason"] == "no_results"
    assert sg.calls == 0


@pytest.mark.asyncio
async def test_ungrounded_streamed_answer_gets_a_correction_event():
    sg = FakeStreamingGenerator(["This looks confident but cites nothing at all."])
    deps = make_deps([hit()], sg, reranker=ScriptedRerankerFactory()())
    events = await collect(answer_question_stream(
        question="q", document_id="d1", tenant_id="default", top_k=5, use_rerank=True, deps=deps, request_id="r1",
    ))
    kinds = [e["event"] for e in events]
    assert kinds == ["delta", "correction", "done"]
    assert events[-1]["data"]["refusal_reason"] == "ungrounded_answer"
    assert events[-1]["data"]["citations"] == []


@pytest.mark.asyncio
async def test_stream_failure_mid_generation_yields_correction_and_done():
    deps = make_deps([hit()], FailingStreamingGenerator(), reranker=ScriptedRerankerFactory()())
    events = await collect(answer_question_stream(
        question="q", document_id="d1", tenant_id="default", top_k=5, use_rerank=True, deps=deps, request_id="r1",
    ))
    kinds = [e["event"] for e in events]
    assert kinds == ["delta", "correction", "done"]
    assert events[-1]["data"]["insufficient_evidence"] is True


@pytest.mark.asyncio
async def test_no_streaming_generator_configured_refuses_cleanly():
    deps = make_deps([hit()], None, reranker=ScriptedRerankerFactory()())
    events = await collect(answer_question_stream(
        question="q", document_id="d1", tenant_id="default", top_k=5, use_rerank=True, deps=deps, request_id="r1",
    ))
    assert len(events) == 1 and events[0]["event"] == "done"
    assert events[0]["data"]["insufficient_evidence"] is True


@pytest.mark.asyncio
async def test_hallucinated_citation_is_stripped_and_flagged_as_a_warning():
    sg = FakeStreamingGenerator(["Real fact [S1] and a fake one [S99]."])
    deps = make_deps([hit()], sg, reranker=ScriptedRerankerFactory()())
    events = await collect(answer_question_stream(
        question="q", document_id="d1", tenant_id="default", top_k=5, use_rerank=True, deps=deps, request_id="r1",
    ))
    done = events[-1]["data"]
    assert done["citations"] == [c for c in done["citations"] if c["source_id"] == "S1"]
    assert any("S99" in w for w in done["warnings"])
    assert any("lightly edited" in w for w in done["warnings"])  # streamed text != final validated text


@pytest.mark.asyncio
async def test_usage_chunk_feeds_the_cost_breakdown():
    sg = FakeStreamingGenerator(["Answer [S1]."], prompt_tokens=321, completion_tokens=42)
    deps = make_deps([hit()], sg, reranker=ScriptedRerankerFactory()())
    events = await collect(answer_question_stream(
        question="q", document_id="d1", tenant_id="default", top_k=5, use_rerank=True, deps=deps, request_id="r1",
    ))
    cost = events[-1]["data"]["cost"]
    assert cost["generation_prompt_tokens"] == 321 and cost["generation_completion_tokens"] == 42
    assert cost["generation_cost_usd"] > 0
