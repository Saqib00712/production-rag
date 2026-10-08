"""Unit tests for OpenAIStreamingLLM's chunk parsing, using a fake async
iterator shaped like the OpenAI SDK's streaming chunks -- no network."""

import pytest

from app.services.llm import LLMError, OpenAIStreamingLLM


class FakeDelta:
    def __init__(self, content=None):
        self.content = content


class FakeChoice:
    def __init__(self, content=None):
        self.delta = FakeDelta(content)


class FakeSDKChunk:
    def __init__(self, content=None, usage=None):
        self.choices = [FakeChoice(content)] if content is not None else []
        self.usage = usage


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class FakeClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        return FakeAsyncStream(self._chunks)


@pytest.mark.asyncio
async def test_yields_text_deltas_in_order():
    chunks = [FakeSDKChunk(content="Hello"), FakeSDKChunk(content=" world")]
    llm = OpenAIStreamingLLM(FakeClient(chunks))
    out = [c async for c in llm.stream_completion(model="m", system="s", user="u", max_tokens=100)]
    assert [c.text for c in out if c.text] == ["Hello", " world"]


@pytest.mark.asyncio
async def test_yields_usage_as_separate_final_chunk():
    chunks = [FakeSDKChunk(content="Hi"), FakeSDKChunk(usage=FakeUsage(50, 10))]
    llm = OpenAIStreamingLLM(FakeClient(chunks))
    out = [c async for c in llm.stream_completion(model="m", system="s", user="u", max_tokens=100)]
    usage_chunks = [c for c in out if c.usage]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.prompt_tokens == 50 and usage_chunks[0].usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_empty_delta_content_is_not_yielded_as_text():
    chunks = [FakeSDKChunk(content=None), FakeSDKChunk(content="real")]
    llm = OpenAIStreamingLLM(FakeClient(chunks))
    out = [c async for c in llm.stream_completion(model="m", system="s", user="u", max_tokens=100)]
    assert [c.text for c in out if c.text] == ["real"]
