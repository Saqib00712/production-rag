"""Dependency providers (FastAPI Depends). Tests override these."""

from functools import lru_cache
from typing import Callable

from fastapi import Depends

from app.config import Settings, get_settings
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.tenant_repository import TenantRepository
from app.services.chunker import TokenCounter
from app.services.contextualizer import Contextualizer, OpenAIContextualizer
from app.services.embedder import Embedder, EmbeddingConfigError, OpenAIEmbedder
from app.services.generator import Generator, OpenAIGenerator, OpenAIStreamingGenerator, StreamingGenerator
from app.services.llm import JsonLLM, LLMConfigError, OpenAIJsonLLM, OpenAIStreamingLLM, StreamingLLM
from app.services.memory_extractor import MemoryExtractor, OpenAIMemoryExtractor
from app.services.ocr import Ocr, OpenAIOcr
from app.services.openai_client import get_openai_client
from app.services.reranker import OpenAIReranker, Reranker
from app.services.vector_index_cache import TenantIndexCache

EmbedderFactory = Callable[[], Embedder]
JsonLLMFactory = Callable[[], JsonLLM]
StreamingLLMFactory = Callable[[], StreamingLLM]
StreamingGeneratorFactory = Callable[[], StreamingGenerator | None]
RerankerFactory = Callable[[], Reranker | None]
GeneratorFactory = Callable[[], Generator | None]
ContextualizerFactory = Callable[[], Contextualizer | None]
MemoryExtractorFactory = Callable[[], MemoryExtractor | None]
OcrFactory = Callable[[], Ocr | None]


@lru_cache
def get_token_counter() -> TokenCounter:
    return TokenCounter()


def get_repository(settings: Settings = Depends(get_settings)) -> ChunkRepository:
    return ChunkRepository(settings.sqlite_path)


@lru_cache
def _tenant_repo_singleton(path: str) -> TenantRepository:
    # Cached per db path (not per Settings instance) so the rate limiter's
    # sibling concept here -- app.state -- isn't needed: tests that build a
    # fresh Settings with a fresh tmp path naturally get a fresh repository,
    # while repeated calls with the SAME path (normal app operation) reuse
    # one instance instead of reopening the SQLite file every request.
    return TenantRepository(path)


def get_tenant_repository(settings: Settings = Depends(get_settings)) -> TenantRepository:
    return _tenant_repo_singleton(settings.tenant_db_path)


@lru_cache
def _conversation_repo_singleton(path: str) -> ConversationRepository:
    # Same rationale as _tenant_repo_singleton above: cached per db path,
    # so a fresh Settings (tests, a different tmp path) naturally gets a
    # fresh repository instead of reopening the SQLite file every request.
    return ConversationRepository(path)


def get_conversation_repository(settings: Settings = Depends(get_settings)) -> ConversationRepository:
    return _conversation_repo_singleton(settings.conversation_db_path)


@lru_cache
def _memory_repo_singleton(path: str) -> MemoryRepository:
    # Same rationale as _conversation_repo_singleton above.
    return MemoryRepository(path)


def get_memory_repository(settings: Settings = Depends(get_settings)) -> MemoryRepository:
    return _memory_repo_singleton(settings.memory_db_path)


def get_embedder_factory(settings: Settings = Depends(get_settings)) -> EmbedderFactory:
    """A factory (not an instance) so dry runs and fully-cached re-indexes
    never need an API key or a network client."""

    def factory() -> Embedder:
        key = settings.openai_api_key
        if not key or key.startswith("sk-your"):
            raise EmbeddingConfigError("OPENAI_API_KEY is not configured.")
        return OpenAIEmbedder(
            api_key=key,
            model=settings.openai_embedding_model,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )

    return factory


def get_json_llm_factory(settings: Settings = Depends(get_settings)) -> JsonLLMFactory:
    def factory() -> JsonLLM:
        return OpenAIJsonLLM(get_openai_client(settings))
    return factory


def get_reranker_factory(
    settings: Settings = Depends(get_settings), llm_factory: JsonLLMFactory = Depends(get_json_llm_factory)
) -> RerankerFactory:
    def factory() -> Reranker | None:
        try:
            return OpenAIReranker(llm_factory(), settings.openai_rerank_model)
        except LLMConfigError:
            return None  # ask endpoint runs rerank-less rather than failing
    return factory


def get_generator_factory(
    settings: Settings = Depends(get_settings), llm_factory: JsonLLMFactory = Depends(get_json_llm_factory)
) -> GeneratorFactory:
    def factory() -> Generator | None:
        try:
            return OpenAIGenerator(llm_factory(), settings.openai_chat_model, settings.max_answer_tokens)
        except LLMConfigError:
            return None
    return factory


def get_streaming_llm_factory(settings: Settings = Depends(get_settings)) -> StreamingLLMFactory:
    def factory() -> StreamingLLM:
        return OpenAIStreamingLLM(get_openai_client(settings))
    return factory


def get_streaming_generator_factory(
    settings: Settings = Depends(get_settings), llm_factory: StreamingLLMFactory = Depends(get_streaming_llm_factory)
) -> StreamingGeneratorFactory:
    def factory() -> StreamingGenerator | None:
        try:
            return OpenAIStreamingGenerator(llm_factory(), settings.openai_chat_model, settings.max_answer_tokens)
        except LLMConfigError:
            return None
    return factory


def get_contextualizer_factory(
    settings: Settings = Depends(get_settings), llm_factory: JsonLLMFactory = Depends(get_json_llm_factory)
) -> ContextualizerFactory:
    def factory() -> Contextualizer | None:
        try:
            return OpenAIContextualizer(llm_factory(), settings.openai_contextualizer_model, settings.contextualizer_max_turns)
        except LLMConfigError:
            return None  # ask endpoint falls back to the raw question rather than failing
    return factory


def get_memory_extractor_factory(
    settings: Settings = Depends(get_settings), llm_factory: JsonLLMFactory = Depends(get_json_llm_factory)
) -> MemoryExtractorFactory:
    def factory() -> MemoryExtractor | None:
        try:
            return OpenAIMemoryExtractor(llm_factory(), settings.openai_memory_extractor_model)
        except LLMConfigError:
            return None  # ask endpoint just skips extraction rather than failing
    return factory


def get_ocr_factory(settings: Settings = Depends(get_settings)) -> OcrFactory:
    """Returns None (not an Ocr instance) when OCR is disabled or
    misconfigured, mirroring get_reranker_factory/get_generator_factory's
    fail-open pattern -- callers skip OCR rather than failing the upload."""

    def factory() -> Ocr | None:
        if not settings.ocr_enabled:
            return None
        try:
            return OpenAIOcr(get_openai_client(settings), settings.openai_ocr_model)
        except LLMConfigError:
            return None
    return factory


@lru_cache
def _tenant_index_cache_singleton(backend: str, ef_construction: int, m: int, ef_search: int) -> TenantIndexCache:
    return TenantIndexCache(backend=backend, ef_construction=ef_construction, m=m, ef_search=ef_search)


def get_tenant_index_cache(settings: Settings = Depends(get_settings)) -> TenantIndexCache:
    return _tenant_index_cache_singleton(
        settings.vector_index_backend, settings.vector_index_ef_construction,
        settings.vector_index_m, settings.vector_index_ef_search,
    )
