"""
Integration tests for the rate-limit and audit-log middleware, wired into
the real app via TestClient (in-process, no real network).

Rate limiting is set up ONCE at app import time (see app/main.py), not
per-request via FastAPI's dependency_overrides -- middleware wraps the
whole ASGI app, outside the dependency-injection graph. So tests reach into
`app.state.rate_limiter` directly to adjust the limit and reset counts
between tests, rather than overriding a Settings dependency (which
middleware never reads at request time).
"""

from app.config import get_settings
from app.dependencies import get_embedder_factory
from app.main import app
from fastapi.testclient import TestClient
from tests.fakes import HashEmbedder


def make_client(settings, rate_limit_per_minute):
    app.state.rate_limiter.limit = rate_limit_per_minute
    app.state.rate_limiter.reset()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: HashEmbedder())
    return TestClient(app)


def test_health_is_never_rate_limited(settings):
    client = make_client(settings, rate_limit_per_minute=1)
    for _ in range(10):
        assert client.get("/health").status_code == 200
    app.dependency_overrides.clear()


def test_ask_endpoint_is_rate_limited_after_the_configured_count(settings):
    client = make_client(settings, rate_limit_per_minute=2)
    for _ in range(2):
        r = client.post("/ask", json={"question": "anything"})
        assert r.status_code != 429  # not rate-limited yet
    limited = client.post("/ask", json={"question": "anything"})
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    app.dependency_overrides.clear()


def test_rate_limit_key_changes_with_forwarded_for_header(settings):
    client = make_client(settings, rate_limit_per_minute=1)
    first = client.post("/ask", json={"question": "q1"}, headers={"X-Forwarded-For": "1.1.1.1"})
    assert first.status_code != 429
    blocked = client.post("/ask", json={"question": "q2"}, headers={"X-Forwarded-For": "1.1.1.1"})
    assert blocked.status_code == 429
    different_client = client.post("/ask", json={"question": "q3"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert different_client.status_code != 429
    app.dependency_overrides.clear()


def test_documents_upload_is_also_rate_limited(settings):
    client = make_client(settings, rate_limit_per_minute=1)
    files = {"file": ("t.pdf", b"%PDF-1.4 not really valid but nonzero", "application/pdf")}
    client.post("/documents/upload", files=files)
    second = client.post("/documents/upload", files=files)
    assert second.status_code == 429
    app.dependency_overrides.clear()


def test_search_endpoint_is_not_rate_limited_by_the_ask_documents_prefixes(settings):
    # /search is intentionally NOT in the protected prefix list in this
    # design (it's the cheap, no-generation endpoint) -- documenting that
    # choice as a test, so a future change to protected_prefixes is a
    # deliberate decision, not an accident.
    client = make_client(settings, rate_limit_per_minute=1)
    for _ in range(5):
        r = client.post("/search", json={"query": "test", "mode": "keyword"})
        assert r.status_code != 429
    app.dependency_overrides.clear()


def test_every_response_gets_a_request_id_header(settings):
    client = make_client(settings, rate_limit_per_minute=100)
    r = client.get("/health")
    assert "X-Request-ID" in r.headers
    app.dependency_overrides.clear()
