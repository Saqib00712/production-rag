from tests.fakes import FailingEmbedder
from app.dependencies import get_embedder_factory
from app.main import app


def _upload(client, pdf):
    r = client.post("/documents/upload", files={"file": ("a.pdf", pdf, "application/pdf")})
    assert r.status_code == 201
    return r.json()["document_id"]


def test_dry_run_costs_nothing_and_writes_nothing(client, embedder, two_page_pdf_bytes):
    doc = _upload(client, two_page_pdf_bytes)
    r = client.post(f"/documents/{doc}/index?dry_run=true")
    body = r.json()
    assert r.status_code == 200 and body["dry_run"] is True
    assert body["chunk_count"] == 2 and body["tokens"] > 0
    assert embedder.calls == []
    assert client.get(f"/documents/{doc}/chunks").status_code == 404


def test_index_stores_chunks_with_page_numbers(client, embedder, two_page_pdf_bytes):
    doc = _upload(client, two_page_pdf_bytes)
    r = client.post(f"/documents/{doc}/index")
    assert r.status_code == 200
    assert r.json()["chunks_needing_embedding"] == 2 and len(embedder.calls) == 1
    chunks = client.get(f"/documents/{doc}/chunks").json()["chunks"]
    by_page = {c["page_number"]: c for c in chunks}
    assert "refund" in by_page[1]["text"] and "warranty" in by_page[2]["text"]
    assert all(c["has_embedding"] and c["embedding_dim"] == 8 for c in chunks)


def test_reindex_is_free_thanks_to_embedding_cache(client, embedder, two_page_pdf_bytes):
    doc = _upload(client, two_page_pdf_bytes)
    client.post(f"/documents/{doc}/index")
    r = client.post(f"/documents/{doc}/index").json()
    assert r["chunks_from_cache"] == 2 and r["tokens"] == 0 and r["cost_usd"] == 0
    assert len(embedder.calls) == 1  # no second API call


def test_same_content_in_new_upload_reuses_embeddings(client, embedder, two_page_pdf_bytes):
    client.post(f"/documents/{_upload(client, two_page_pdf_bytes)}/index")
    r = client.post(f"/documents/{_upload(client, two_page_pdf_bytes)}/index").json()
    assert r["chunks_from_cache"] == 2 and len(embedder.calls) == 1


def test_blank_page_is_skipped_with_warning(client, pdf_with_blank_page_bytes):
    doc = _upload(client, pdf_with_blank_page_bytes)
    body = client.post(f"/documents/{doc}/index").json()
    assert body["skipped_pages"] == [2] and body["warnings"]


def test_provider_failure_returns_502_and_saves_nothing(client, two_page_pdf_bytes):
    doc = _upload(client, two_page_pdf_bytes)
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: FailingEmbedder())
    assert client.post(f"/documents/{doc}/index").status_code == 502
    assert client.get(f"/documents/{doc}/chunks").status_code == 404


def test_token_budget_guard_returns_413(client, settings, two_page_pdf_bytes):
    doc = _upload(client, two_page_pdf_bytes)
    settings.max_index_tokens = 1
    assert client.post(f"/documents/{doc}/index").status_code == 413


def test_invalid_and_unknown_ids(client):
    assert client.post("/documents/not-a-uuid/index").status_code == 422
    assert client.post("/documents/6f1c2b1e-0000-4000-8000-000000000000/index").status_code == 404


def test_missing_api_key_gives_503_but_dry_run_still_works(client_no_key, two_page_pdf_bytes):
    doc = _upload(client_no_key, two_page_pdf_bytes)
    assert client_no_key.post(f"/documents/{doc}/index").status_code == 503
    assert client_no_key.post(f"/documents/{doc}/index?dry_run=true").status_code == 200


def test_index_flags_pii_without_blocking(client, two_page_pdf_bytes):
    # two_page_pdf_bytes has no email/phone; use a fresh doc with an email instead.
    import io
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "Contact Jane Doe at jane.doe@example.com for more information.")
    doc = client.post("/documents/upload", files={"file": ("c.pdf", bytes(pdf.output()), "application/pdf")}).json()["document_id"]
    body = client.post(f"/documents/{doc}/index").json()
    assert body["pii_detected"].get("email") == 1
    assert any("PII" in w for w in body["warnings"])
    assert body["chunk_count"] > 0  # indexing was NOT blocked


def test_index_reports_no_pii_for_clean_content(client, two_page_pdf_bytes):
    doc = client.post("/documents/upload", files={"file": ("a.pdf", two_page_pdf_bytes, "application/pdf")}).json()["document_id"]
    body = client.post(f"/documents/{doc}/index").json()
    assert body["pii_detected"] == {}


# --- Day 21: PII redaction mode ---

def test_pii_mode_off_skips_detection_entirely(client, settings):
    import io
    from fpdf import FPDF

    settings.pii_mode = "off"
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "Contact Jane Doe at jane.doe@example.com for more information.")
    doc = client.post("/documents/upload", files={"file": ("c.pdf", bytes(pdf.output()), "application/pdf")}).json()["document_id"]
    body = client.post(f"/documents/{doc}/index").json()
    assert body["pii_detected"] == {} and body["pii_mode"] == "off"
    assert not any("PII" in w for w in body["warnings"])


def test_pii_mode_redact_removes_the_value_before_it_is_ever_chunked_or_stored(client, settings):
    from fpdf import FPDF

    settings.pii_mode = "redact"
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "Contact Jane Doe at jane.doe@example.com for more information.")
    doc = client.post("/documents/upload", files={"file": ("c.pdf", bytes(pdf.output()), "application/pdf")}).json()["document_id"]
    body = client.post(f"/documents/{doc}/index").json()
    assert body["pii_detected"].get("email") == 1 and body["pii_mode"] == "redact"
    assert any("REDACTED" in w for w in body["warnings"])

    chunks = client.get(f"/documents/{doc}/chunks").json()["chunks"]
    all_text = " ".join(c["text"] for c in chunks)
    assert "jane.doe@example.com" not in all_text        # the raw value never reached storage
    assert "[REDACTED_EMAIL]" in all_text                 # the redacted marker did
