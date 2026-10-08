"""
Unit tests for services/pdf_parser.py.

These tests call the parser directly -- no HTTP, no FastAPI, no OpenAI.
They run in well under a second and cost $0. This is exactly the kind of
test suite that should run on every commit.

Day 21: parse_pdf is now async (it may need to await an OCR vision call),
so every test here awaits it -- with no ocr_factory passed, behavior is
byte-for-byte what it was Day 1-20 (see tests/test_ocr.py for the OCR path
itself).
"""

import pytest

from app.models.document import PageStatus
from app.services.pdf_parser import PdfParsingError, parse_pdf


async def test_parses_two_pages_with_correct_page_numbers(two_page_pdf_bytes):
    parsed = await parse_pdf(
        file_bytes=two_page_pdf_bytes,
        filename="test.pdf",
        content_type="application/pdf",
        request_id="test-req-1",
    )

    assert parsed.metadata.page_count == 2
    assert len(parsed.pages) == 2
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[1].page_number == 2


async def test_page_text_content_is_preserved(two_page_pdf_bytes):
    parsed = await parse_pdf(
        file_bytes=two_page_pdf_bytes,
        filename="test.pdf",
        content_type="application/pdf",
        request_id="test-req-2",
    )

    assert "refund policy" in parsed.pages[0].text
    assert "warranty terms" in parsed.pages[1].text
    # Page 1's content must never leak into page 2 -- this is the core
    # guarantee citations depend on.
    assert "warranty" not in parsed.pages[0].text
    assert "refund" not in parsed.pages[1].text


async def test_blank_page_is_flagged_not_silently_dropped(pdf_with_blank_page_bytes):
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes,
        filename="test.pdf",
        content_type="application/pdf",
        request_id="test-req-3",
    )

    assert parsed.pages[0].status == PageStatus.OK
    assert parsed.pages[1].status == PageStatus.EMPTY
    assert parsed.empty_page_numbers == [2]


async def test_corrupt_pdf_raises_pdf_parsing_error():
    garbage_bytes = b"this is not a pdf file at all"

    with pytest.raises(PdfParsingError):
        await parse_pdf(
            file_bytes=garbage_bytes,
            filename="fake.pdf",
            content_type="application/pdf",
            request_id="test-req-4",
        )


async def test_char_count_matches_text_length(two_page_pdf_bytes):
    parsed = await parse_pdf(
        file_bytes=two_page_pdf_bytes,
        filename="test.pdf",
        content_type="application/pdf",
        request_id="test-req-5",
    )

    for page in parsed.pages:
        assert page.char_count == len(page.text)


async def test_blank_page_stays_empty_when_no_ocr_factory_given(pdf_with_blank_page_bytes):
    """Day 21 regression guard: omitting ocr_factory entirely (its default)
    must behave EXACTLY like OCR doesn't exist -- no import attempted, no
    behavior change from Day 1-20."""
    parsed = await parse_pdf(
        file_bytes=pdf_with_blank_page_bytes,
        filename="test.pdf",
        content_type="application/pdf",
        request_id="test-req-6",
    )
    assert parsed.pages[1].status == PageStatus.EMPTY
    assert parsed.metadata.ocr_pages_used == 0
    assert parsed.metadata.ocr_cost_usd == 0.0
