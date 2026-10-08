"""
Integration tests for POST /documents/upload.

These use FastAPI's TestClient, which runs the app in-process (no real
network socket, no server process to manage) -- fast, and still exercises
the full request path: routing, validation, parsing, response shaping.
"""


def test_upload_valid_pdf_returns_201_with_pages(client, two_page_pdf_bytes):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", two_page_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["metadata"]["page_count"] == 2
    assert body["metadata"]["filename"] == "test.pdf"
    assert len(body["pages"]) == 2
    assert "document_id" in body


def test_upload_rejects_non_pdf_content_type(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", b"just some text", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_rejects_corrupt_pdf(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("fake.pdf", b"not a real pdf", "application/pdf")},
    )

    assert response.status_code == 422


def test_upload_rejects_oversized_file(client, settings):
    settings.max_upload_size_mb = 0  # any non-empty body now exceeds the limit
    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", b"%PDF-1.4 fake but nonzero", "application/pdf")},
    )
    assert response.status_code == 413


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Day 21: OCR wiring through the real /documents/upload endpoint ---

def test_upload_ocrs_a_scanned_page_when_ocr_is_enabled(client, settings, pdf_with_blank_page_bytes):
    import pytest
    pytest.importorskip("pymupdf")  # real page rendering; only the vision call is faked
    from app.dependencies import get_ocr_factory
    from app.main import app
    from tests.fakes import ScriptedOcrFactory

    fake = ScriptedOcrFactory(text="Transcribed: terms and conditions apply.")
    app.dependency_overrides[get_ocr_factory] = lambda: fake
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("scanned.pdf", pdf_with_blank_page_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["pages"][1]["status"] == "ocr"
        assert "Transcribed" in body["pages"][1]["text"]
        assert body["metadata"]["ocr_pages_used"] == 1
        assert body["metadata"]["ocr_cost_usd"] > 0
        assert len(fake.calls) == 1
    finally:
        app.dependency_overrides.pop(get_ocr_factory, None)


def test_upload_without_ocr_override_leaves_scanned_page_empty(client, pdf_with_blank_page_bytes):
    """Default wiring: get_ocr_factory's REAL dependency runs (not a fake),
    and since settings.ocr_enabled defaults to False, it returns None --
    the scanned page comes back exactly as it always has, Day 1-20."""
    response = client.post(
        "/documents/upload",
        files={"file": ("scanned.pdf", pdf_with_blank_page_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["pages"][1]["status"] == "empty"
    assert body["metadata"]["ocr_pages_used"] == 0
