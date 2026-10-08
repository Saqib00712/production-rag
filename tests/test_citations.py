from app.services.citations import validate_citations


def test_valid_single_and_group_citation_passthrough():
    r = validate_citations("Refunds take 5 days [S1].", [], {"S1", "S2"})
    assert r.text == "Refunds take 5 days [S1]." and r.cited_ids == ["S1"] and r.invalid_ids == []
    r = validate_citations("Covered by two sources [S1, S2].", [], {"S1", "S2"})
    assert r.cited_ids == ["S1", "S2"]


def test_hallucinated_id_is_stripped_not_trusted():
    r = validate_citations("Made up fact [S9].", [], {"S1"})
    assert r.invalid_ids == ["S9"] and "S9" not in r.cited_ids
    assert "[S9]" not in r.text  # empty bracket group removed, not left dangling


def test_mixed_valid_and_invalid_in_one_group():
    r = validate_citations("Mixed claim [S1, S9].", [], {"S1"})
    assert r.text == "Mixed claim [S1]." and r.cited_ids == ["S1"] and r.invalid_ids == ["S9"]


def test_no_citation_at_all_yields_empty_cited_ids():
    r = validate_citations("An answer with no citation.", [], {"S1"})
    assert r.cited_ids == [] and r.invalid_ids == []


def test_citations_listed_separately_but_not_inline_are_still_counted():
    r = validate_citations("General statement.", ["S1"], {"S1", "S2"})
    assert r.cited_ids == ["S1"]


def test_dedupes_repeated_citation_and_preserves_first_seen_order():
    r = validate_citations("A [S2]. B [S1]. C [S2].", [], {"S1", "S2"})
    assert r.cited_ids == ["S2", "S1"]


def test_whitespace_cleanup_around_punctuation():
    r = validate_citations("Fact one [S9] . Fact two [S1].", [], {"S1"})
    assert " ." not in r.text
