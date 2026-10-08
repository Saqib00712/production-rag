"""
Data contracts for parsed documents.

Design decision: page-level structure is a first-class citizen, not an
afterthought. Every later pipeline stage (chunking on Day 2, embedding on
Day 3, retrieval + citation later) consumes PageContent objects, so page
numbers are never lost -- they are carried as metadata through the entire
pipeline rather than reconstructed at the end.
"""

from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class PageStatus(str, Enum):
    """
    Why this exists: not all pages extract successfully. A page might be
    blank, might be a scanned image with no embedded text layer, or might
    error during parsing. We need to know WHICH case we're in -- silently
    treating "needs OCR" the same as "genuinely blank" is a data-quality bug
    that surfaces weeks later as "why can't the bot find anything from
    Appendix C?"
    """

    OK = "ok"
    EMPTY = "empty"                # page has no extractable text (likely scanned)
    EXTRACTION_ERROR = "error"     # parser raised an exception on this page
    OCR = "ocr"                    # Day 21: no embedded text layer, but a vision call transcribed it


class PageContent(BaseModel):
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(default="", description="Raw extracted text for this page")
    char_count: int = Field(default=0, ge=0)
    status: PageStatus = PageStatus.OK

    def model_post_init(self, __context) -> None:
        # Keep char_count consistent with text automatically -- avoids a
        # class of bugs where callers forget to update char_count by hand.
        object.__setattr__(self, "char_count", len(self.text))


class DocumentMetadata(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    page_count: int
    tenant_id: str = "default"  # backward-compat default for documents uploaded before Day 10
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ocr_pages_used: int = 0     # Day 21: pages that had no text layer and were transcribed via vision OCR
    ocr_tokens: int = 0
    ocr_cost_usd: float = 0.0


class ParsedDocument(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    pages: list[PageContent]

    @property
    def empty_page_numbers(self) -> list[int]:
        """Pages that likely need OCR -- surfaced explicitly, never hidden."""
        return [p.page_number for p in self.pages if p.status == PageStatus.EMPTY]

    @property
    def total_char_count(self) -> int:
        return sum(p.char_count for p in self.pages)
