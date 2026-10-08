import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from app.auth import TenantContext, get_current_tenant
from app.config import Settings, get_settings
from app.dependencies import (
    get_contextualizer_factory, get_embedder_factory, get_generator_factory, get_memory_extractor_factory,
    get_reranker_factory,
)
from app.main import app
from tests.fakes import (
    HashEmbedder, ScriptedContextualizerFactory, ScriptedGeneratorFactory, ScriptedMemoryExtractorFactory,
    ScriptedRerankerFactory,
)

_TEST_TENANT = TenantContext(tenant_id="default", key_label="test-fixture")


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Both rate limiters (Day 7's per-IP, Day 10's per-tenant) live on
    app.state as singletons for the whole process -- without this, hits
    from one test would carry over and spuriously rate-limit a LATER,
    unrelated test. A high default limit here means tests don't hit them
    unless they deliberately lower it (see tests/test_middleware.py)."""
    app.state.rate_limiter.limit = 10_000
    app.state.rate_limiter.reset()
    app.state.tenant_rate_limiter.limit = 10_000
    app.state.tenant_rate_limiter.reset()
    yield


@pytest.fixture(autouse=True)
def _reset_observability():
    """cost_tracker and latency_recorder (app/observability/metrics.py) are
    module-level singletons for the whole process, same rationale as the
    rate limiter above: reset before every test so one test's numbers never
    bleed into another's assertions."""
    from app.observability.metrics import cost_tracker, latency_recorder
    from app.observability.tracing import reset as reset_tracing

    latency_recorder.reset()
    cost_tracker.reset()
    reset_tracing()
    yield


@pytest.fixture(autouse=True)
def _reset_tenant_index_cache():
    """The per-tenant vector index cache (app/services/vector_index_cache.py)
    is memoized by (backend, ef_construction, m, ef_search) via lru_cache in
    dependencies.py -- since most tests share default settings, they'd
    otherwise share ONE cache instance keyed by tenant_id (commonly
    "default" across tests), leaking a stale index built from a PREVIOUS
    test's temp database into a later test. Clearing the lru_cache forces a
    fresh TenantIndexCache per test."""
    from app.dependencies import _tenant_index_cache_singleton

    _tenant_index_cache_singleton.cache_clear()
    yield


@pytest.fixture(autouse=True)
def _default_authenticated_tenant():
    """Day 10 requires an API key on every /documents, /search, /ask call.
    Almost every existing test predates auth and has no business testing it
    (that's tests/test_auth.py's job) -- so by default, EVERY test gets a
    fixed, already-authenticated tenant via a dependency override, exactly
    like get_settings/get_embedder_factory are already overridden. Tests
    that specifically want to exercise real auth (missing/invalid key,
    cross-tenant isolation) remove this override themselves (see
    tests/test_auth.py) rather than fighting it."""
    app.dependency_overrides[get_current_tenant] = lambda: _TEST_TENANT
    yield
    app.dependency_overrides.pop(get_current_tenant, None)


def _build_pdf(pages_text: list[str]) -> bytes:
    pdf = FPDF()
    for text in pages_text:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        if text:
            pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


@pytest.fixture
def two_page_pdf_bytes() -> bytes:
    return _build_pdf([
        "This is page one. It discusses the refund policy in detail.",
        "This is page two. It discusses the warranty terms in detail.",
    ])


@pytest.fixture
def pdf_with_blank_page_bytes() -> bytes:
    return _build_pdf(["Only the first page has real text content here.", ""])


@pytest.fixture
def settings(tmp_path) -> Settings:
    # isolated storage per test; _env_file=None ignores the developer's real .env
    return Settings(
        raw_storage_dir=str(tmp_path / "raw"),
        sqlite_path=str(tmp_path / "rag.db"),
        tenant_db_path=str(tmp_path / "tenants.db"),
        conversation_db_path=str(tmp_path / "conversations.db"),
        memory_db_path=str(tmp_path / "memories.db"),
        openai_api_key="test-key",
        _env_file=None,
    )


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def client(settings, embedder):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: embedder)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def reranker_factory():
    return ScriptedRerankerFactory()


@pytest.fixture
def generator_factory():
    return ScriptedGeneratorFactory()


@pytest.fixture
def contextualizer_factory():
    return ScriptedContextualizerFactory()


@pytest.fixture
def memory_extractor_factory():
    return ScriptedMemoryExtractorFactory()


@pytest.fixture
def ask_client(settings, embedder, reranker_factory, generator_factory, contextualizer_factory, memory_extractor_factory):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: embedder)
    app.dependency_overrides[get_reranker_factory] = lambda: reranker_factory
    app.dependency_overrides[get_generator_factory] = lambda: generator_factory
    app.dependency_overrides[get_contextualizer_factory] = lambda: contextualizer_factory
    app.dependency_overrides[get_memory_extractor_factory] = lambda: memory_extractor_factory
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_key(settings):
    """Real embedder factory + empty key: exercises the 'not configured' path."""
    settings.openai_api_key = ""
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()
