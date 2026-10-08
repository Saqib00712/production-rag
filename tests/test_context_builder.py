from types import SimpleNamespace

from app.services.chunker import TokenCounter
from app.services.context_builder import build_sources

COUNTER = TokenCounter(use_tiktoken=False)


def hit(cid, text, page=1, doc="d1"):
    return SimpleNamespace(chunk_id=cid, document_id=doc, page_number=page, text=text)


def test_assigns_sequential_source_ids_and_keeps_page_numbers():
    hits = [hit("a", "alpha text", 1), hit("b", "beta text", 2)]
    sources = build_sources(hits, COUNTER, max_sources=5, max_tokens=1000)
    assert [s.source_id for s in sources] == ["S1", "S2"]
    assert sources[0].page_number == 1 and sources[1].page_number == 2


def test_respects_max_sources():
    hits = [hit(str(i), f"text {i}") for i in range(10)]
    assert len(build_sources(hits, COUNTER, max_sources=3, max_tokens=10_000)) == 3


def test_stops_at_token_budget_but_keeps_at_least_one_source():
    hits = [hit("a", "word " * 20), hit("b", "word " * 20)]
    sources = build_sources(hits, COUNTER, max_sources=5, max_tokens=6)
    assert len(sources) == 1


def test_skips_empty_and_duplicate_text():
    hits = [hit("a", "same text"), hit("b", "  "), hit("c", "same text")]
    sources = build_sources(hits, COUNTER, max_sources=5, max_tokens=1000)
    assert len(sources) == 1 and sources[0].chunk_id == "a"


def test_attaches_rerank_scores_when_provided():
    hits = [hit("a", "text a"), hit("b", "text b")]
    sources = build_sources(hits, COUNTER, max_sources=5, max_tokens=1000, scores={"a": 8.5})
    by_id = {s.chunk_id: s for s in sources}
    assert by_id["a"].rerank_score == 8.5 and by_id["b"].rerank_score is None
