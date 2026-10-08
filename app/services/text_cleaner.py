"""
Text cleaning between extraction and chunking.

Why: PDF extraction returns text with layout artifacts (mid-sentence line
breaks, hyphenated line wraps, control chars, odd unicode). Those add noise
to embeddings and make chunk boundaries ugly. Cleaning is deterministic and
free (no LLM). Page boundaries are NOT touched: we clean page by page.
"""

import re
import unicodedata

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_HYPHEN_WRAP = re.compile(r"(\w)-\n([a-z])")        # "infor-\nmation"
_SOFT_BREAK = re.compile(r"(?<![.!?:;\n])\n(?=[a-z(])")  # "latency\nmonitoring"
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ligatures (ﬁ -> fi), nbsp, etc.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _SOFT_BREAK.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
