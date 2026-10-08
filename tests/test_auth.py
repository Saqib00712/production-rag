"""
Real authentication + multi-tenancy tests. These deliberately REMOVE the
autouse `get_current_tenant` override from conftest.py (which every other
test relies on to skip auth entirely) so the actual auth dependency runs.
"""

import pytest
from fpdf import FPDF

from app.auth import TenantContext, generate_api_key, get_current_tenant
from app.config import get_settings
from app.dependencies import get_embedder_factory, get_tenant_repository
from app.main import app
from app.repositories.tenant_repository import TenantRepository
from tests.fakes import HashEmbedder


def _pdf(text="Refund policy. Customers may request a refund within thirty days."):
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


@pytest.fixture
def real_auth_client(settings):
    """Like the `client` fixture, but WITHOUT the default tenant override --
    real API keys must be presented."""
    from fastapi.testclient import TestClient

    app.dependency_overrides.pop(get_current_tenant, None)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: HashEmbedder())
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def tenant_repo(settings) -> TenantRepository:
    return get_tenant_repository(settings)


def _make_key(tenant_repo, tenant_id="tenant-a", label="test key") -> str:
    key = generate_api_key()
    tenant_repo.create_key(key, tenant_id=tenant_id, label=label)
    return key


def test_missing_api_key_is_401(real_auth_client):
    r = real_auth_client.post("/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")})
    assert r.status_code == 401


def test_invalid_api_key_is_401(real_auth_client):
    r = real_auth_client.post(
        "/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")},
        headers={"Authorization": "Bearer sk-tenant-not-a-real-key"},
    )
    assert r.status_code == 401


def test_valid_bearer_key_is_accepted(real_auth_client, tenant_repo):
    key = _make_key(tenant_repo)
    r = real_auth_client.post(
        "/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 201


def test_valid_x_api_key_header_is_also_accepted(real_auth_client, tenant_repo):
    key = _make_key(tenant_repo)
    r = real_auth_client.post(
        "/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 201


def test_revoked_key_is_rejected(real_auth_client, tenant_repo):
    key = _make_key(tenant_repo)
    record = tenant_repo.lookup(key)
    tenant_repo.revoke(record.key_hash)
    r = real_auth_client.post(
        "/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 401


def test_malformed_authorization_header_is_401(real_auth_client):
    r = real_auth_client.get(
        "/documents/00000000-0000-4000-8000-000000000000/chunks",
        headers={"Authorization": "NotBearer something"},
    )
    assert r.status_code == 401


def test_one_tenant_cannot_see_another_tenants_document(real_auth_client, tenant_repo):
    key_a = _make_key(tenant_repo, "tenant-a")
    key_b = _make_key(tenant_repo, "tenant-b")

    up = real_auth_client.post(
        "/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")},
        headers={"Authorization": f"Bearer {key_a}"},
    )
    doc_id = up.json()["document_id"]

    # Tenant B tries to index tenant A's document by guessing/reusing the id.
    r = real_auth_client.post(f"/documents/{doc_id}/index", headers={"Authorization": f"Bearer {key_b}"})
    assert r.status_code == 404  # NOT 403 -- existence is not confirmed either


def test_one_tenant_cannot_read_another_tenants_chunks(real_auth_client, tenant_repo):
    key_a = _make_key(tenant_repo, "tenant-a")
    key_b = _make_key(tenant_repo, "tenant-b")

    up = real_auth_client.post(
        "/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")},
        headers={"Authorization": f"Bearer {key_a}"},
    )
    doc_id = up.json()["document_id"]
    real_auth_client.post(f"/documents/{doc_id}/index", headers={"Authorization": f"Bearer {key_a}"})

    r = real_auth_client.get(f"/documents/{doc_id}/chunks", headers={"Authorization": f"Bearer {key_b}"})
    assert r.status_code == 404


def test_search_never_crosses_tenants_even_with_no_document_filter(real_auth_client, tenant_repo):
    key_a = _make_key(tenant_repo, "tenant-a")
    key_b = _make_key(tenant_repo, "tenant-b")

    up_a = real_auth_client.post(
        "/documents/upload",
        files={"file": ("a.pdf", _pdf("Alpha corp refund policy is thirty days."), "application/pdf")},
        headers={"Authorization": f"Bearer {key_a}"},
    )
    real_auth_client.post(f"/documents/{up_a.json()['document_id']}/index", headers={"Authorization": f"Bearer {key_a}"})

    # Tenant B has indexed NOTHING. A document-less search must see nothing of tenant A's.
    r = real_auth_client.post(
        "/search", json={"query": "refund policy", "mode": "keyword"},
        headers={"Authorization": f"Bearer {key_b}"},
    )
    assert r.status_code == 200 and r.json()["hits"] == []


def test_per_tenant_rate_limit_returns_429(real_auth_client, tenant_repo):
    key = _make_key(tenant_repo)
    app.state.tenant_rate_limiter.limit = 1
    app.state.tenant_rate_limiter.reset()
    try:
        files = {"file": ("a.pdf", _pdf(), "application/pdf")}
        first = real_auth_client.post("/documents/upload", files=files, headers={"Authorization": f"Bearer {key}"})
        assert first.status_code != 429
        second = real_auth_client.post("/documents/upload", files=files, headers={"Authorization": f"Bearer {key}"})
        assert second.status_code == 429
    finally:
        app.state.tenant_rate_limiter.limit = 10_000
        app.state.tenant_rate_limiter.reset()


def test_daily_budget_exceeded_returns_402(real_auth_client, tenant_repo, settings):
    key = _make_key(tenant_repo)
    import datetime

    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    tenant_repo.add_usage("tenant-a", today, settings.tenant_daily_budget_usd)  # already at/over budget

    r = real_auth_client.post(
        "/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 402


def test_budget_check_is_per_tenant_not_global(real_auth_client, tenant_repo, settings):
    key_a = _make_key(tenant_repo, "tenant-a")
    key_b = _make_key(tenant_repo, "tenant-b")
    import datetime

    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    tenant_repo.add_usage("tenant-a", today, settings.tenant_daily_budget_usd)

    files = {"file": ("a.pdf", _pdf(), "application/pdf")}
    blocked = real_auth_client.post("/documents/upload", files=files, headers={"Authorization": f"Bearer {key_a}"})
    allowed = real_auth_client.post("/documents/upload", files=files, headers={"Authorization": f"Bearer {key_b}"})
    assert blocked.status_code == 402
    assert allowed.status_code == 201
