from app.eval.answer_metrics import AskLikeResult, check_citation_correct, check_keyword_hit, judge_correctness
from app.eval.models import GoldenQuestion


def answerable(pages=(1,), keywords=()):
    return GoldenQuestion(id="q1", question="q?", expected_pages=list(pages), expected_keywords=list(keywords))


def unanswerable():
    return GoldenQuestion(id="q1", question="q?")


def result(refused=False, reason=None, text="answer", pages=()):
    return AskLikeResult(refused=refused, refusal_reason=reason, answer_text=text, cited_pages=list(pages))


def test_citation_correct_when_page_matches():
    assert check_citation_correct(answerable(pages=[2]), result(pages=[2])) is True


def test_citation_incorrect_when_page_does_not_match():
    assert check_citation_correct(answerable(pages=[2]), result(pages=[5])) is False


def test_citation_check_is_none_for_unanswerable_question():
    assert check_citation_correct(unanswerable(), result(refused=True)) is None


def test_keyword_hit_case_insensitive():
    g = answerable(keywords=["Refund"])
    assert check_keyword_hit(g, result(text="Your REFUND was processed.")) is True
    assert check_keyword_hit(g, result(text="unrelated text")) is False


def test_keyword_hit_none_when_no_keywords_or_refused():
    g = answerable(keywords=["x"])
    assert check_keyword_hit(answerable(), result()) is None
    assert check_keyword_hit(g, result(refused=True)) is None


def test_correctness_unanswerable_and_refused_is_correct():
    assert judge_correctness(unanswerable(), result(refused=True), correct_citation=None) is True


def test_correctness_unanswerable_but_answered_is_wrong():
    assert judge_correctness(unanswerable(), result(refused=False), correct_citation=None) is False


def test_correctness_answerable_but_refused_is_wrong():
    assert judge_correctness(answerable(), result(refused=True), correct_citation=None) is False


def test_correctness_answerable_and_correctly_cited_is_correct():
    assert judge_correctness(answerable(), result(pages=[1]), correct_citation=True) is True


def test_correctness_answerable_but_wrong_citation_is_wrong():
    assert judge_correctness(answerable(), result(pages=[9]), correct_citation=False) is False
