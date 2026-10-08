import pytest

from app.services.chunker import Chunker, TokenCounter

COUNTER = TokenCounter(use_tiktoken=False)  # deterministic (~4 chars/token)


def make(size=50, overlap=15):
    return Chunker(COUNTER, size, overlap)


def sentences(n):
    return " ".join(f"Sentence number {i} is right here." for i in range(n))


def test_empty_text_gives_no_chunks():
    assert make().chunk_page(1, "") == []


def test_chunks_keep_page_number_and_never_cross_pages():
    drafts = make().chunk_pages([(1, sentences(20)), (2, "Warranty text on page two.")])
    assert {d.page_number for d in drafts} == {1, 2}
    assert all("Warranty" not in d.text for d in drafts if d.page_number == 1)
    assert [d.chunk_index for d in drafts] == list(range(len(drafts)))


def test_chunks_respect_size_limit():
    for d in make().chunk_page(1, sentences(60)):
        assert d.token_count <= 50 + 5


def test_overlap_repeats_trailing_sentence():
    drafts = make().chunk_page(1, sentences(60))
    assert len(drafts) > 2
    for a, b in zip(drafts, drafts[1:]):
        last_sentence = a.text.split(". ")[-1].rstrip(".")
        assert last_sentence in b.text or a.text.split("\n")[-1][-15:] in b.text


def test_zero_overlap_has_no_repeated_text():
    drafts = Chunker(COUNTER, 50, 0).chunk_page(1, sentences(40))
    joined = " ".join(d.text for d in drafts)
    assert joined.count("Sentence number 7 ") == 1


def test_oversized_sentence_is_hard_split_without_losing_words():
    words = [f"w{i}" for i in range(2000)]
    drafts = make().chunk_page(1, " ".join(words))
    assert len(drafts) > 1
    rebuilt = " ".join(d.text for d in drafts).split()
    assert set(rebuilt) == set(words)


def test_huge_single_word_terminates():
    drafts = make().chunk_page(1, "x" * 10_000)
    assert len(drafts) >= 1


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        Chunker(COUNTER, 50, 50)
    with pytest.raises(ValueError):
        Chunker(COUNTER, 0, 0)
