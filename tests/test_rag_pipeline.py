"""Full /ask integration tests using scripted reranker/generator (no live LLM calls)."""

from fpdf import FPDF


def _pdf(pages):
    pdf = FPDF()
    for t in pages:
        pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, t)
    return bytes(pdf.output())


PAGES = [
    "Refund policy. Customers may request a refund within thirty days of purchase.",
    "Warranty terms. The device has a two year warranty for manufacturing defects.",
]


def _index(client, pages=PAGES):
    r = client.post("/documents/upload", files={"file": ("d.pdf", _pdf(pages), "application/pdf")})
    doc = r.json()["document_id"]
    assert client.post(f"/documents/{doc}/index").status_code == 200
    return doc


def test_grounded_answer_with_valid_citation_is_returned(ask_client, generator_factory):
    _index(ask_client)
    generator_factory.reply = {"answer": "Refunds within 30 days [S1].", "citations": ["S1"], "insufficient_evidence": False}
    r = ask_client.post("/ask", json={"question": "refund policy"})
    body = r.json()
    assert r.status_code == 200
    assert body["insufficient_evidence"] is False and body["refusal_reason"] is None
    assert body["answer"] == "Refunds within 30 days [S1]."
    assert len(body["citations"]) == 1 and body["citations"][0]["page_number"] in (1, 2)
    assert body["cost"]["total_cost_usd"] > 0


def test_no_search_results_refuses_without_calling_generator(ask_client, generator_factory):
    r = ask_client.post("/ask", json={"question": "anything at all"})
    body = r.json()
    assert body["insufficient_evidence"] is True and body["refusal_reason"] == "no_results"
    assert generator_factory.calls == 0


def test_low_relevance_after_rerank_refuses_without_generating(ask_client, reranker_factory, generator_factory):
    doc = _index(ask_client)
    chunks = ask_client.get(f"/documents/{doc}/chunks").json()["chunks"]
    reranker_factory.score_map = {c["chunk_id"]: 1.0 for c in chunks}  # below min_rerank_score (4.0)
    r = ask_client.post("/ask", json={"question": "refund policy"})
    body = r.json()
    assert r.status_code == 200
    assert body["insufficient_evidence"] is True and body["refusal_reason"] == "low_relevance"
    assert generator_factory.calls == 0  # never pay for generation on a bad match


def test_high_relevance_after_rerank_proceeds_to_generate(ask_client, reranker_factory, generator_factory):
    doc = _index(ask_client)
    chunks = ask_client.get(f"/documents/{doc}/chunks").json()["chunks"]
    reranker_factory.score_map = {c["chunk_id"]: 9.0 for c in chunks}
    r = ask_client.post("/ask", json={"question": "refund policy"})
    assert r.status_code == 200 and generator_factory.calls == 1


def test_model_says_insufficient_evidence_is_a_refusal_not_an_error(ask_client, generator_factory):
    _index(ask_client)
    generator_factory.reply = {"answer": "", "citations": [], "insufficient_evidence": True}
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["insufficient_evidence"] is True and body["refusal_reason"] == "model_insufficient_evidence"


def test_answer_with_no_valid_citation_is_rejected_as_ungrounded(ask_client, generator_factory):
    _index(ask_client)
    generator_factory.reply = {"answer": "Some claim with no support.", "citations": [], "insufficient_evidence": False}
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["insufficient_evidence"] is True and body["refusal_reason"] == "ungrounded_answer"


def test_hallucinated_citation_id_is_stripped_and_warned(ask_client, generator_factory):
    _index(ask_client)
    generator_factory.reply = {"answer": "A fact [S1, S99].", "citations": ["S1", "S99"], "insufficient_evidence": False}
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["answer"] == "A fact [S1]."
    assert any("S99" in w for w in body["warnings"])


def test_reranker_failure_degrades_to_hybrid_ranking_not_a_500(ask_client, reranker_factory, generator_factory):
    _index(ask_client)
    reranker_factory.raise_error = True
    r = ask_client.post("/ask", json={"question": "refund policy"})
    assert r.status_code == 200
    assert any("Reranker unavailable" in w for w in r.json()["warnings"])
    assert generator_factory.calls == 1  # pipeline still proceeded to generation


def test_generator_failure_returns_502_not_a_silent_answer(ask_client, generator_factory):
    _index(ask_client)
    generator_factory.raise_error = True
    r = ask_client.post("/ask", json={"question": "refund policy"})
    assert r.status_code == 502


def test_no_rerank_flag_skips_reranker_entirely(ask_client, reranker_factory, generator_factory):
    _index(ask_client)
    ask_client.post("/ask", json={"question": "refund policy", "rerank": False})
    assert reranker_factory.calls == 0 and generator_factory.calls == 1


def test_missing_api_key_ask_still_returns_a_clean_refusal(client_no_key):
    r = client_no_key.post("/ask", json={"question": "refund policy"})
    body = r.json()
    assert r.status_code == 200
    assert body["insufficient_evidence"] is True


def test_ask_validation_errors(ask_client):
    assert ask_client.post("/ask", json={"question": ""}).status_code == 422
    assert ask_client.post("/ask", json={"question": "  "}).status_code == 422
    assert ask_client.post("/ask", json={"question": "x" * 1001}).status_code == 422
    assert ask_client.post("/ask", json={"question": "x", "top_k": 0}).status_code == 422
    assert ask_client.post("/ask", json={"question": "x", "top_k": 11}).status_code == 422


def test_answer_scoped_to_one_document(ask_client, generator_factory):
    d1 = _index(ask_client)
    _index(ask_client, ["Unrelated content about refund rules for a different product."])
    generator_factory.reply = {"answer": "Scoped answer [S1].", "citations": ["S1"], "insufficient_evidence": False}
    body = ask_client.post("/ask", json={"question": "refund policy", "document_id": d1}).json()
    assert all(c["document_id"] == d1 for c in body["citations"])
