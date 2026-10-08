"""Thin JSON-mode wrapper over OpenAI chat completions (structured outputs).

Reranker and generator depend on the JsonLLM Protocol, not on the SDK, so
tests inject fakes and the parsing/prompt logic is verifiable offline.
"""

import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from app.services.openai_client import OpenAIConfigError  # re-exported

logger = logging.getLogger(__name__)
LLMConfigError = OpenAIConfigError


class LLMError(Exception):
    """LLM call failed after retries or returned unusable output."""


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class JsonLLM(Protocol):
    async def complete_json(
        self, *, model: str, system: str, user: str, schema_name: str, schema: dict, max_tokens: int
    ) -> tuple[dict, LLMUsage]: ...


@dataclass
class StreamChunk:
    """A tagged unit from a streaming completion: either a text delta OR
    the final usage report, never both. Modeled as one type with two
    optional fields (rather than two separate yielded types) so callers can
    iterate a single async stream uniformly, checking `.text is not None`."""

    text: str | None = None
    usage: LLMUsage | None = None


class StreamingLLM(Protocol):
    def stream_completion(
        self, *, model: str, system: str, user: str, max_tokens: int
    ) -> "AsyncIterator[StreamChunk]": ...


class OpenAIJsonLLM:
    def __init__(self, client):
        self._client = client

    async def complete_json(self, *, model, system, user, schema_name, schema, max_tokens):
        from openai import OpenAIError

        try:
            resp = await self._client.chat.completions.create(
                model=model,
                temperature=0,  # deterministic-ish; remove if you switch to a reasoning model
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            )
        except OpenAIError as exc:
            logger.error("llm_failed", extra={"error_type": type(exc).__name__})
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise LLMError("The model refused the request.")
        try:
            data = json.loads(msg.content or "")
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise LLMError("The model returned an unexpected JSON shape.")
        usage = LLMUsage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else LLMUsage()
        return data, usage


class OpenAIStreamingLLM:
    """
    Plain-TEXT streaming (no response_format=json_schema).

    Why not stream structured JSON output: the non-streaming /ask endpoint
    uses json_schema mode specifically so citations/insufficient_evidence
    arrive as clean, typed fields (Day 4). But streaming raw JSON tokens to
    a user would show them fragments like `{"ans` mid-render -- unusable
    for a chat UI. Getting BOTH real token-by-token prose streaming AND
    structured fields requires incremental JSON parsing (extracting just
    the "answer" string as it fills in), which is real production
    complexity most teams add only once they need it. Day 8 takes the
    simpler, honest path instead: plain-text streaming with INLINE [S#]
    citations (same regex-based validate_citations() from Day 4 parses
    them from the finished text), accepting a real, documented capability
    trade-off -- see docs/day8.md for what's lost by not doing incremental
    JSON parsing.
    """

    def __init__(self, client):
        self._client = client

    async def stream_completion(self, *, model: str, system: str, user: str, max_tokens: int) -> AsyncIterator[StreamChunk]:
        from openai import OpenAIError

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                temperature=0,
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                stream=True,
                stream_options={"include_usage": True},
            )
        except OpenAIError as exc:
            logger.error("llm_stream_start_failed", extra={"error_type": type(exc).__name__})
            raise LLMError(f"LLM stream failed to start: {type(exc).__name__}") from exc

        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield StreamChunk(text=chunk.choices[0].delta.content)
                if getattr(chunk, "usage", None):
                    yield StreamChunk(usage=LLMUsage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens))
        except OpenAIError as exc:
            logger.error("llm_stream_interrupted", extra={"error_type": type(exc).__name__})
            raise LLMError(f"LLM stream interrupted: {type(exc).__name__}") from exc
