from app.services.text_cleaner import clean_text


def test_joins_mid_sentence_line_breaks():
    assert clean_text("added Redis caching for latency\nmonitoring.") == "added Redis caching for latency monitoring."


def test_keeps_breaks_after_sentence_end_and_before_capitals():
    assert clean_text("First line.\nSecond line") == "First line.\nSecond line"
    assert clean_text("EDUCATION\nUniversity") == "EDUCATION\nUniversity"


def test_dehyphenates_wrapped_words():
    assert clean_text("infor-\nmation retrieval") == "information retrieval"


def test_removes_control_chars_and_normalizes_unicode():
    assert clean_text("a\x00b \ufb01le") == "ab file"


def test_collapses_blank_lines_and_spaces():
    assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_empty_input():
    assert clean_text("  \n ") == ""
