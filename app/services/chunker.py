"""
Page-aware, sentence-boundary chunker.

Design decisions (each is an interview talking point):
  * Chunks NEVER cross page boundaries -> every chunk has exactly one
    page_number, so citations are exact. Cost: a sentence spanning two pages
    is split. Accepted trade-off for citation correctness.
  * Size is measured in TOKENS (what the embedding model bills/limits), not
    characters.
  * Chunks are packed from whole sentences/lines, not cut mid-word, so each
    chunk is readable and embeds a coherent idea.
  * Overlap = trailing sentences carried into the next chunk, so an answer
    sitting on a boundary appears whole in at least one chunk.
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TokenCounter:
    """cl100k_base (used by text-embedding-3-*) via tiktoken; falls back to
    ~4 chars/token if the encoding file can't be loaded (e.g. offline)."""

    def __init__(self, encoding_name: str = "cl100k_base", use_tiktoken: bool = True):
        self._enc = None
        if use_tiktoken:
            try:
                import tiktoken

                self._enc = tiktoken.get_encoding(encoding_name)
            except Exception:
                logger.warning("tiktoken_unavailable_using_approximation")

    @property
    def is_exact(self) -> bool:
        return self._enc is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            # disallowed_special=() -> document text like "<|endoftext|>" is
            # treated as plain text instead of raising.
            return len(self._enc.encode(text, disallowed_special=()))
        return max(1, math.ceil(len(text) / 4))


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    page_number: int
    text: str
    token_count: int


@dataclass(frozen=True)
class _Unit:
    sep: str  # separator placed BEFORE this unit when joined: " " or "\n"
    text: str
    tokens: int


class Chunker:
    def __init__(self, counter: TokenCounter, chunk_size: int, overlap: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self._counter = counter
        self._size = chunk_size
        self._overlap = overlap

    # ---- public API ----
    def chunk_pages(self, pages: Iterable[tuple[int, str]]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for page_number, text in pages:
            drafts.extend(self.chunk_page(page_number, text, start_index=len(drafts)))
        return drafts

    def chunk_page(self, page_number: int, text: str, start_index: int = 0) -> list[ChunkDraft]:
        units = self._build_units(text)
        out: list[ChunkDraft] = []
        i = 0
        while i < len(units):
            j, total = i, 0
            while j < len(units) and total + units[j].tokens <= self._size:
                total += units[j].tokens
                j += 1
            if j == i:  # pathological unit larger than size: still make progress
                j = i + 1
            chunk_text = self._join(units[i:j])
            out.append(
                ChunkDraft(
                    chunk_index=start_index + len(out),
                    page_number=page_number,
                    text=chunk_text,
                    token_count=self._counter.count(chunk_text),
                )
            )
            if j >= len(units):
                break
            # overlap: step back over trailing units up to `overlap` tokens,
            # but always advance at least one unit (guarantees termination)
            k, carried = j, 0
            while k > i + 1 and carried + units[k - 1].tokens <= self._overlap:
                k -= 1
                carried += units[k].tokens
            i = k
        return out

    # ---- internals ----
    @staticmethod
    def _join(units: list[_Unit]) -> str:
        return "".join((u.sep if n else "") + u.text for n, u in enumerate(units))

    def _build_units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip()]
            for s_no, sentence in enumerate(sentences):
                base_sep = "\n" if (s_no == 0 and units) else " "
                for k, piece in enumerate(self._fit(sentence)):
                    units.append(
                        _Unit(base_sep if k == 0 else " ", piece, self._counter.count(piece))
                    )
        return units

    def _fit(self, sentence: str) -> list[str]:
        """Split a sentence larger than chunk_size on word boundaries."""
        if self._counter.count(sentence) <= self._size:
            return [sentence]
        pieces: list[str] = []
        current: list[str] = []
        cur_tokens = 0
        for word in sentence.split():
            # very long "words" (base64, URLs): slice by characters
            parts = [word] if len(word) <= self._size * 3 else [
                word[n : n + self._size * 3] for n in range(0, len(word), self._size * 3)
            ]
            for part in parts:
                wt = self._counter.count(part) + 1
                if current and cur_tokens + wt > self._size:
                    pieces.append(" ".join(current))
                    current, cur_tokens = [], 0
                current.append(part)
                cur_tokens += wt
        if current:
            pieces.append(" ".join(current))
        return pieces
