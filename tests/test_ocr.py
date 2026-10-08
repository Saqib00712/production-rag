"""
Unit tests for Day 21's OCR pass (app/services/pdf_parser.py's
_run_ocr_pass + app/services/ocr.py), using a scripted fake Ocr (no real
vision call, no network) over a REAL rendered page (PyMuPDF IS exercised
here -- only the OpenAI call itself is faked), plus the fail-open paths
that don't need PyMuPDF installed at all.
"""

import pytest

from app.models.document import PageStatus
from app.services.pdf_parser import parse_pdf
from tests.fakes import ScriptedOcrFactory


async def test_no_ocr_factory_leaves_empty_pages_untouched(pdf_with_blank_page_bytes):
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes, filename="t.pdf", content_type="application/pdf",
        request_id="r1",
    )
    assert parsed.pages[1].status == PageStatus.EMPTY
    assert parsed.metadata.ocr_pages_used == 0


async def test_ocr_factory_returning_none_leaves_empty_pages_untouched(pdf_with_blank_page_bytes):
    """Mirrors get_ocr_factory's real behavior when ocr_enabled=False or no
    API key is configured -- the factory itself returns None."""
    fake = ScriptedOcrFactory(none_ocr=True)
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes, filename="t.pdf", content_type="application/pdf",
        request_id="r2", ocr_factory=fake, ocr_max_pages=5,
    )
    assert parsed.pages[1].status == PageStatus.EMPTY
    assert parsed.metadata.ocr_pages_used == 0
    assert fake.calls == []


async def test_successful_ocr_fills_in_the_empty_page(pdf_with_blank_page_bytes):
    pytest.importorskip("pymupdf")  # real page rendering; only the vision call is faked
    fake = ScriptedOcrFactory(text="Scanned warranty terms: two years, serial required.")
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes, filename="t.pdf", content_type="application/pdf",
        request_id="r3", ocr_factory=fake, ocr_max_pages=5,
        ocr_price_input_per_million=0.15, ocr_price_output_per_million=0.60,
    )
    assert parsed.pages[1].status == PageStatus.OCR
    assert "warranty" in parsed.pages[1].text.lower()
    assert parsed.pages[1].char_count == len(parsed.pages[1].text)
    assert parsed.metadata.ocr_pages_used == 1
    assert parsed.metadata.ocr_tokens == 140  # 120 prompt + 20 completion, from the fake
    assert parsed.metadata.ocr_cost_usd > 0
    assert len(fake.calls) == 1  # exactly one vision call, for the one empty page


async def test_ocr_respects_the_max_pages_cap():
    """Three blank pages, cap of 1 -> only the FIRST is OCR'd; the rest stay
    EMPTY. This is the cost/abuse guard, not a quality knob."""
    pytest.importorskip("pymupdf")  # real page rendering; only the vision call is faked
    from tests.conftest import _build_pdf

    pdf_bytes = _build_pdf(["", "", ""])
    fake = ScriptedOcrFactory(text="some text")
    parsed = await parse_pdf(
        file_bytes=pdf_bytes, filename="t.pdf", content_type="application/pdf",
        request_id="r4", ocr_factory=fake, ocr_max_pages=1,
    )
    statuses = [p.status for p in parsed.pages]
    assert statuses.count(PageStatus.OCR) == 1
    assert statuses.count(PageStatus.EMPTY) == 2
    assert len(fake.calls) == 1


async def test_ocr_zero_max_pages_skips_the_pass_entirely(pdf_with_blank_page_bytes):
    fake = ScriptedOcrFactory()
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes, filename="t.pdf", content_type="application/pdf",
        request_id="r5", ocr_factory=fake, ocr_max_pages=0,
    )
    assert parsed.pages[1].status == PageStatus.EMPTY
    assert fake.calls == []


async def test_ocr_transcribe_failure_leaves_the_page_empty_not_crashed(pdf_with_blank_page_bytes):
    fake = ScriptedOcrFactory(raise_error=True)
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes, filename="t.pdf", content_type="application/pdf",
        request_id="r6", ocr_factory=fake, ocr_max_pages=5,
    )
    assert parsed.pages[1].status == PageStatus.EMPTY
    assert parsed.metadata.ocr_pages_used == 0  # the failed page was never counted as "used"


async def test_ocr_fails_open_when_pymupdf_is_not_installed(pdf_with_blank_page_bytes, monkeypatch):
    """Simulates a machine without `pip install -r requirements-ocr.txt` --
    the whole OCR pass must fail open (page stays EMPTY), never crash the
    upload, exactly like hnswlib (Day 20) fails open for the default
    backend."""
    import app.services.pdf_parser as pdf_parser_module

    def _raise_import_error(file_bytes, page_index):
        raise ImportError("simulated: PyMuPDF not installed")

    monkeypatch.setattr(pdf_parser_module, "render_page_to_png", _raise_import_error)
    fake = ScriptedOcrFactory()
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes, filename="t.pdf", content_type="application/pdf",
        request_id="r7", ocr_factory=fake, ocr_max_pages=5,
    )
    assert parsed.pages[1].status == PageStatus.EMPTY
    assert parsed.metadata.ocr_pages_used == 0
    assert fake.calls == []  # never even reached the (fake) vision call
