"""
Vision-based OCR for scanned PDF pages (Day 21).

Why OpenAI vision instead of a traditional OCR library (Tesseract etc.):
this curriculum is deliberately OpenAI-API-only throughout (see
docs/PROJECT_GUIDE.md) -- no local models, no separate inference stack to
install and maintain. A vision-capable chat model can transcribe a page
image through the exact same client/auth/retry/cost-accounting path every
other LLM call in this project already uses, instead of adding a second,
unrelated dependency (a system Tesseract binary, language data files) that
behaves completely differently from everything else here.

Why this is a Protocol + fake, same as every other LLM-backed service
(Embedder, JsonLLM, Reranker, Generator): callers (pdf_parser.py) depend on
`Ocr`, never on the OpenAI SDK directly, so tests exercise the real
page-selection and cost-accounting logic without a network call.

Cost reality check: a vision call is the single most expensive operation in
this pipeline, by a wide margin, because an image is encoded as hundreds to
over a thousand "tokens" depending on resolution -- nowhere near the cost of
embedding a paragraph of text. This is why OCR is opt-in
(`settings.ocr_enabled`, default False) AND capped per document
(`settings.ocr_max_pages_per_document`), unlike every other guard in this
project, which exists to catch abuse rather than normal, expected cost.
"""

import base64
import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

OCR_PROMPT = (
    "Transcribe ALL text visible in this image exactly as it appears, preserving reading order "
    "(left-to-right, top-to-bottom, column by column for multi-column layouts). Output ONLY the "
    "transcribed text -- no commentary, no markdown, no description of the image. If the image "
    "contains no readable text, output an empty string."
)


@dataclass
class OcrResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class OcrError(Exception):
    """A vision call failed after retries, or returned something unusable."""


class Ocr(Protocol):
    async def transcribe(self, image_bytes: bytes) -> OcrResult: ...


class OpenAIOcr:
    """Sends a rendered page image to a vision-capable chat model and
    returns the transcribed text. Uses the SAME AsyncOpenAI client every
    other OpenAI-backed service shares (app.services.openai_client), so
    connection pooling and clean shutdown are already handled."""

    def __init__(self, client, model: str):
        self._client, self._model = client, model

    async def transcribe(self, image_bytes: bytes) -> OcrResult:
        from openai import OpenAIError

        b64 = base64.b64encode(image_bytes).decode("ascii")
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
            )
        except OpenAIError as exc:
            logger.error("ocr_failed", extra={"error_type": type(exc).__name__})
            raise OcrError(f"OCR request failed: {type(exc).__name__}") from exc

        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return OcrResult(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


def render_page_to_png(file_bytes: bytes, page_index: int) -> bytes:
    """
    Renders ONE page of a PDF (0-indexed) to PNG bytes for OCR.

    Library choice: PyMuPDF (imported as `fitz`), lazily imported here --
    mirrors app/services/vector_index.py's HnswIndex (Day 20): a ships-with-
    prebuilt-wheels, no-system-dependency rendering library kept OUT of the
    hard requirements files (see requirements-ocr.txt) since the default
    (ocr_enabled=False) never needs it, exactly like hnswlib and the default
    brute_force vector backend.

    Raises ImportError if PyMuPDF isn't installed -- callers (pdf_parser.py)
    catch this and fail open (page stays EMPTY, OCR silently skipped for
    that document) rather than failing the whole upload over an optional
    dependency.
    """
    import pymupdf as fitz  # PyMuPDF -- `import fitz` is the deprecated spelling as of pymupdf 1.24+

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        page = doc[page_index]
        # 2x zoom: a plain 72dpi render is often too low-resolution for a
        # vision model to read small print reliably; this is a reasonable
        # default, not a tuned constant.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return pix.tobytes("png")
