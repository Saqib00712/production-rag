"""Turn ranked hits into numbered, token-budgeted sources [S1], [S2]...

The [S#] ids are what the model cites; we map them back to page numbers
ourselves, so page numbers never pass through the model.
"""

from dataclasses import dataclass

from app.services.chunker import TokenCounter


@dataclass(frozen=True)
class Source:
    source_id: str
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    rerank_score: float | None = None


def build_sources(hits, counter: TokenCounter, max_sources: int, max_tokens: int,
                  scores: dict[str, float] | None = None) -> list[Source]:
    sources: list[Source] = []
    used = 0
    seen: set[str] = set()
    for h in hits:
        if len(sources) >= max_sources:
            break
        text = h.text.strip()
        if not text or text in seen:
            continue
        tokens = counter.count(text)
        if used + tokens > max_tokens:
            if sources:
                continue  # a later, smaller chunk might still fit
            text, tokens = text[: max_tokens * 4], max_tokens  # never return zero context
        seen.add(text)
        used += tokens
        sources.append(Source(
            source_id=f"S{len(sources) + 1}", chunk_id=h.chunk_id, document_id=h.document_id,
            page_number=h.page_number, text=text, rerank_score=(scores or {}).get(h.chunk_id),
        ))
    return sources
