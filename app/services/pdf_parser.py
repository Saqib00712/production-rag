"""
PDF text extraction service.

Design decision: this module has exactly one responsibility -- turn PDF
bytes into a ParsedDocument with page-level fidelity. It does not know
about HTTP, FastAPI, or storage. That separation means we can unit test it
with zero web server involved, and reuse it later from a background worker
(Day 15: Human-in-the-Loop / async processing) without changes.

Library choice: pypdf.
  - Pure Python, no external system dependencies (unlike e.g. poppler-based
    tools), which keeps Docker images small and builds reproducible.
  - Actively maintained fork of the old PyPDF2 project.
  - Good enough for text-based PDFs. It does NOT do OCR itself -- a page
    with no embedded text layer comes back EMPTY. Day 21 adds an OPTIONAL
    OCR pass (app/services/ocr.py) for exactly those pages, via a vision-
    capable OpenAI model; when it's off (the default) or unavailable, a
    scanned page is detected and flagged exactly as before, never hidden.
"""

import logging
import uuid
from io import BytesIO
from typing import Callable

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models.document import (
    DocumentMetadata,
    PageContent,
    PageStatus,
    ParsedDocument,
)
from app.services.ocr import Ocr, OcrError, render_page_to_png

logger = logging.getLogger(__name__)


class PdfParsingError(Exception):
    """Raised when a PDF cannot be opened at all (corrupt, not a real PDF)."""


async def parse_pdf(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    request_id: str,
    *,
    ocr_factory: Callable[[], "Ocr | None"] | None = None,
    ocr_max_pages: int = 0,
    ocr_price_input_per_million: float = 0.0,
    ocr_price_output_per_million: float = 0.0,
) -> ParsedDocument:
    """
    Parse PDF bytes into a ParsedDocument, preserving page numbers.

    Day 21's OCR pass is entirely optional and fails open: `ocr_factory`
    defaults to None (no OCR at all, byte-for-byte Day 1-20 behavior).
    When given, it's called ONLY for pages with no extractable text, up to
    `ocr_max_pages` per document (a hard cost/abuse cap -- a 300-page
    scanned PDF does NOT silently trigger 300 vision calls). Any failure
    along this path (no vision-capable LLM configured, PyMuPDF not
    installed, the vision call itself erroring) leaves that page EMPTY,
    exactly as it would have been without OCR -- it never fails the upload.

    Raises:
        PdfParsingError: if the file cannot be opened as a PDF at all.
    """
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except PdfReadError as exc:
        logger.error(
            "pdf_open_failed",
            extra={"request_id": request_id, "error_type": "PdfReadError"},
        )
        raise PdfParsingError(f"Could not open '{filename}' as a PDF") from exc

    if reader.is_encrypted:
        # We do not attempt password guessing. An encrypted PDF with no
        # extractable text is a security/UX decision to surface to the
        # user, not silently skip.
        logger.warning(
            "pdf_encrypted", extra={"request_id": request_id}
        )
        raise PdfParsingError(
            f"'{filename}' is password-protected; cannot extract text"
        )

    pages: list[PageContent] = []
    for index, page in enumerate(reader.pages):
        page_number = index + 1  # 1-indexed for human-readable citations
        try:
            text = page.extract_text() or ""
        except Exception:
            # A single malformed page should not fail the whole document.
            # We log it, mark it, and continue -- the rest of a 200-page
            # PDF is still usable.
            logger.warning(
                "page_extraction_error",
                extra={"request_id": request_id, "error_type": "PageExtractError"},
            )
            pages.append(
                PageContent(page_number=page_number, text="", status=PageStatus.EXTRACTION_ERROR)
            )
            continue

        status = PageStatus.OK if text.strip() else PageStatus.EMPTY
        pages.append(PageContent(page_number=page_number, text=text, status=status))

    ocr_pages_used, ocr_tokens, ocr_cost_usd = 0, 0, 0.0
    if ocr_factory is not None and ocr_max_pages > 0:
        ocr_pages_used, ocr_tokens, ocr_cost_usd = await _run_ocr_pass(
            pages, file_bytes, ocr_factory, ocr_max_pages,
            ocr_price_input_per_million, ocr_price_output_per_million, request_id,
        )

    document_id = str(uuid.uuid4())
    metadata = DocumentMetadata(
        filename=filename,
        content_type=content_type,
        size_bytes=len(file_bytes),
        page_count=len(pages),
        ocr_pages_used=ocr_pages_used,
        ocr_tokens=ocr_tokens,
        ocr_cost_usd=ocr_cost_usd,
    )

    parsed = ParsedDocument(document_id=document_id, metadata=metadata, pages=pages)

    if parsed.empty_page_numbers:
        logger.warning(
            "document_has_empty_pages",
            extra={
                "request_id": request_id,
                "document_id": document_id,
            },
        )

    logger.info(
        "pdf_parsed",
        extra={
            "request_id": request_id,
            "document_id": document_id,
        },
    )
    return parsed


async def _run_ocr_pass(
    pages: list[PageContent],
    file_bytes: bytes,
    ocr_factory: Callable[[], "Ocr | None"],
    max_pages: int,
    price_input_per_million: float,
    price_output_per_million: float,
    request_id: str,
) -> tuple[int, int, float]:
    """Mutates `pages` IN PLACE, replacing each EMPTY page's text with its
    OCR transcription where that succeeds (status becomes PageStatus.OCR).
    Returns (pages_actually_ocr_d, total_cost_usd). Every failure mode here
    -- no OCR configured, PyMuPDF missing, the vision call itself failing --
    leaves the page exactly as it was (EMPTY), logs a warning, and moves on;
    OCR is a strict improvement attempt, never a new way for an upload to
    fail."""
    from app.services.cost import estimate_cost_usd

    ocr = ocr_factory()
    if ocr is None:
        return 0, 0, 0.0

    candidates = [p for p in pages if p.status == PageStatus.EMPTY][:max_pages]
    if not candidates:
        return 0, 0, 0.0

    used, tokens, cost = 0, 0, 0.0
    for page in candidates:
        try:
            image_bytes = render_page_to_png(file_bytes, page.page_number - 1)
        except ImportError:
            logger.warning("ocr_skipped_pymupdf_missing", extra={"request_id": request_id})
            return used, tokens, cost  # fail open for the WHOLE pass -- it'll fail for every page identically
        except Exception:
            logger.warning("ocr_render_failed", extra={"request_id": request_id, "page_number": page.page_number})
            continue

        try:
            result = await ocr.transcribe(image_bytes)
        except OcrError:
            logger.warning("ocr_transcribe_failed", extra={"request_id": request_id, "page_number": page.page_number})
            continue

        page_cost = (
            estimate_cost_usd(result.prompt_tokens, price_input_per_million)
            + estimate_cost_usd(result.completion_tokens, price_output_per_million)
        )
        cost += page_cost
        tokens += result.prompt_tokens + result.completion_tokens
        used += 1
        if result.text.strip():
            page.text = result.text
            page.status = PageStatus.OCR
            object.__setattr__(page, "char_count", len(page.text))

    if used:
        logger.info("ocr_pass_completed", extra={"request_id": request_id, "pages_ocrd": used, "cost_usd": round(cost, 6)})
    return used, tokens, round(cost, 8)
