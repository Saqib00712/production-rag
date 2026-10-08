"""
HTTP-level integration tests proving the per-tenant index cache (Day 12) is
correctly wired into real /search and /ask calls -- not just correct in
isolation (test_vector_index.py, test_vector_index_cache.py).
"""

import pytest
from fpdf import FPDF

from app.config import get_settings
from app.dependencies import get_embedder_factory
from app.main import app
from fastapi.testclient import TestClient
from tests.fakes import BowEmbedder


def _pdf(text):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


@pytest.fixture(params=["brute_force", "hnsw"])
def backend_client(request, settings):
    if request.param == "hnsw":
        pytest.importorskip("hnswlib")  # optional dep, Day 20 -- see requirements-hnsw.txt
    settings.vector_index_backend = request.param
    embedder = BowEmbedder()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: embedder)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_whole_tenant_vector_search_works_on_both_backends(backend_client):
    """A document-less (whole-tenant) vector search must find the right
    content regardless of which index backend is configured."""
    doc = backend_client.post("/documents/upload", files={"file": ("a.pdf", _pdf(
        "Refund policy. Customers may request a refund within thirty days."), "application/pdf")}).json()["document_id"]
    assert backend_client.post(f"/documents/{doc}/index").status_code == 200

    r = backend_client.post("/search", json={"query": "money back please", "mode": "vector"})
    body = r.json()
    assert r.status_code == 200
    assert any(h["page_number"] == 1 for h in body["hits"])  # BowEmbedder: "money" clusters with "refund"


def test_newly_indexed_document_is_findable_without_restarting_the_process(backend_client):
    """Proves cache invalidation actually works: index doc1, search (builds
    and caches the index), THEN index doc2 for the SAME tenant, and confirm
    doc2's content is found -- which is only possible if indexing doc2
    correctly invalidated the cached index built before doc2 existed."""
    doc1 = backend_client.post("/documents/upload", files={"file": ("a.pdf", _pdf(
        "Warranty terms. Two year warranty on manufacturing defects."), "application/pdf")}).json()["document_id"]
    backend_client.post(f"/documents/{doc1}/index")
    backend_client.post("/search", json={"query": "warranty defects", "mode": "vector"})  # warms the cache

    doc2 = backend_client.post("/documents/upload", files={"file": ("b.pdf", _pdf(
        "Shipping information. Delivery arrives within five business days."), "application/pdf")}).json()["document_id"]
    backend_client.post(f"/documents/{doc2}/index")

    r = backend_client.post("/search", json={"query": "delivery arrive", "mode": "vector"})
    body = r.json()
    assert any(h["document_id"] == doc2 for h in body["hits"]), (
        "doc2 was not found -- the whole-tenant index cache was not invalidated after indexing it"
    )


def test_ask_endpoint_also_benefits_from_the_cached_index(backend_client, reranker_factory=None):
    from app.dependencies import get_reranker_factory, get_generator_factory
    from tests.fakes import ScriptedGeneratorFactory, ScriptedRerankerFactory

    rf, gf = ScriptedRerankerFactory(), ScriptedGeneratorFactory()
    app.dependency_overrides[get_reranker_factory] = lambda: rf
    app.dependency_overrides[get_generator_factory] = lambda: gf

    doc = backend_client.post("/documents/upload", files={"file": ("a.pdf", _pdf(
        "Privacy notice. We never sell personal data to third parties."), "application/pdf")}).json()["document_id"]
    backend_client.post(f"/documents/{doc}/index")

    gf.reply = {"answer": "We never sell personal data [S1].", "citations": ["S1"], "insufficient_evidence": False}
    r = backend_client.post("/ask", json={"question": "do you sell my data"})
    assert r.status_code == 200 and r.json()["insufficient_evidence"] is False
    app.dependency_overrides.pop(get_reranker_factory, None)
    app.dependency_overrides.pop(get_generator_factory, None)
