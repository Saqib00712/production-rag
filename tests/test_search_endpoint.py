import pytest
from fpdf import FPDF
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_embedder_factory
from app.main import app
from tests.fakes import BowEmbedder, FailingEmbedder


def _pdf(pages):
    pdf = FPDF()
    for t in pages:
        pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, t)
    return bytes(pdf.output())


PAGES = [
    "Refund policy. Customers can request a refund within thirty days of purchase.",
    "Warranty terms. The device has a two year warranty for manufacturing defects.",
    "Shipping information. Standard shipping takes five business days.",
    "Privacy notice. We never sell personal data to anyone.",
]


@pytest.fixture
def emb():
    return BowEmbedder()


@pytest.fixture
def sc(settings, emb):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: emb)
    yield TestClient(app)
    app.dependency_overrides.clear()


def _index(client, pages=PAGES):
    r = client.post("/documents/upload", files={"file": ("d.pdf", _pdf(pages), "application/pdf")})
    doc = r.json()["document_id"]
    assert client.post(f"/documents/{doc}/index").status_code == 200
    return doc


def _search(client, q, mode="hybrid", **kw):
    return client.post("/search", json={"query": q, "mode": mode, **kw})


def test_keyword_search_finds_page_and_never_calls_embedder(sc, emb):
    _index(sc)
    calls_after_index = len(emb.calls)
    body = _search(sc, "warranty defects", "keyword").json()
    assert body["hits"][0]["page_number"] == 2 and body["score_type"] == "bm25"
    assert body["query_tokens"] == 0 and len(emb.calls) == calls_after_index


def test_vector_search_finds_meaning_without_shared_words(sc):
    _index(sc)
    kw = _search(sc, "money back please", "keyword").json()
    vec = _search(sc, "money back please", "vector").json()
    assert kw["hits"] == []                                   # no lexical overlap
    assert vec["hits"][0]["page_number"] == 1 and vec["score_type"] == "cosine"
    assert vec["hits"][0]["vector_score"] > 0.9


def test_hybrid_fuses_both_retrievers(sc):
    _index(sc)
    body = _search(sc, "warranty defects", "hybrid").json()
    top = body["hits"][0]
    assert top["page_number"] == 2 and body["score_type"] == "rrf"
    assert top["keyword_rank"] == 1 and top["vector_rank"] == 1
    assert abs(top["score"] - 2 / 61) < 1e-9


def test_hybrid_rescues_query_keyword_search_misses(sc):
    _index(sc)
    body = _search(sc, "money back please", "hybrid").json()
    assert body["hits"][0]["page_number"] == 1 and body["hits"][0]["keyword_rank"] is None


def test_hybrid_degrades_to_keyword_when_provider_down(sc):
    _index(sc)
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: FailingEmbedder())
    r = _search(sc, "shipping business days", "hybrid")
    body = r.json()
    assert r.status_code == 200 and body["degraded"] is True and body["warnings"]
    assert body["hits"][0]["page_number"] == 3 and body["score_type"] == "bm25"
    assert _search(sc, "shipping", "vector").status_code == 502   # no silent fallback


def test_query_embedding_is_cached(sc, emb):
    _index(sc)
    before = len(emb.calls)
    first = _search(sc, "money back", "vector").json()
    second = _search(sc, "money back", "vector").json()
    assert len(emb.calls) == before + 1
    assert first["query_tokens"] > 0 and second["query_tokens"] == 0 and second["query_cost_usd"] == 0


def test_document_filter_and_top_k(sc):
    d1 = _index(sc)
    _index(sc, ["Refund rules for another product. Refund within ten days.", "Second page about warranty."])
    all_hits = _search(sc, "refund", "keyword", top_k=10).json()["hits"]
    assert len({h["document_id"] for h in all_hits}) == 2
    only = _search(sc, "refund", "keyword", document_id=d1).json()["hits"]
    assert {h["document_id"] for h in only} == {d1}
    assert len(_search(sc, "refund", "keyword", top_k=1).json()["hits"]) == 1


def test_validation(sc):
    assert _search(sc, "", "keyword").status_code == 422
    assert _search(sc, "   ", "keyword").status_code == 422
    assert _search(sc, "x", "keyword", top_k=0).status_code == 422
    assert _search(sc, "x", "keyword", top_k=21).status_code == 422
    assert _search(sc, "x", "semantic").status_code == 422
    assert _search(sc, "x" * 501, "keyword").status_code == 422
    assert _search(sc, "x", "keyword", document_id="nope").status_code == 422


def test_hostile_query_does_not_break_fts(sc):
    _index(sc)
    for q in ['"; DROP TABLE chunks; --', "NEAR( a b", "a:b OR", "* ) (", "refund AND NOT"]:
        r = _search(sc, q, "keyword")
        assert r.status_code == 200, q
    assert _search(sc, "refund", "keyword").json()["hits"]     # data intact


def test_stopword_only_query_returns_empty_with_warning(sc):
    _index(sc)
    body = _search(sc, "what is it", "keyword").json()
    assert body["hits"] == [] and body["warnings"]


def test_search_with_nothing_indexed_is_empty_not_error(sc):
    assert _search(sc, "refund", "hybrid").json()["hits"] == []


def test_vector_mode_without_key_is_503(client_no_key):
    assert _search(client_no_key, "refund", "vector").status_code == 503
