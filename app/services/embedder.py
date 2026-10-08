"""
Embedding service.

Embedder is a Protocol: the pipeline depends on the interface, so tests inject
a fake (no network, $0) and later days can add a caching/routing embedder
without touching the pipeline.

Reliability: the OpenAI SDK retries 429/5xx/connection errors with
exponential backoff (max_retries) and enforces a timeout. Batches are sent
sequentially to stay friendly to rate limits. Any failure is wrapped in
EmbeddingError so callers never depend on SDK exception types.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Embedding provider failed after retries."""


class EmbeddingConfigError(Exception):
    """Embedding provider is not configured (e.g. missing API key)."""


@dataclass
class EmbeddingBatchResult:
    vectors: list[list[float]]
    total_tokens: int


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> EmbeddingBatchResult: ...


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        batch_size: int = 96,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        from openai import AsyncOpenAI

        self.model = model
        self._batch_size = batch_size
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    async def embed(self, texts: list[str]) -> EmbeddingBatchResult:
        from openai import OpenAIError

        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                resp = await self._client.embeddings.create(model=self.model, input=batch)
            except OpenAIError as exc:
                # log the type only: never log the key or full request body
                logger.error("embedding_failed", extra={"error_type": type(exc).__name__})
                raise EmbeddingError(f"Embedding request failed: {type(exc).__name__}") from exc
            data = sorted(resp.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in data)
            total_tokens += resp.usage.total_tokens
        return EmbeddingBatchResult(vectors=vectors, total_tokens=total_tokens)
