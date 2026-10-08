import json

import pytest

from app.eval.loader import GoldenSetError, load_golden_set


def write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_loads_valid_set(tmp_path):
    p = write(tmp_path, "g.json", json.dumps([
        {"id": "q1", "question": "What is X?", "expected_pages": [1]},
        {"id": "q2", "question": "Unanswerable?"},
    ]))
    qs = load_golden_set(p)
    assert len(qs) == 2 and qs[1].is_unanswerable


def test_missing_file_raises():
    with pytest.raises(GoldenSetError):
        load_golden_set("does/not/exist.json")


def test_invalid_json_raises(tmp_path):
    p = write(tmp_path, "g.json", "{not valid json")
    with pytest.raises(GoldenSetError):
        load_golden_set(p)


def test_not_a_list_raises(tmp_path):
    p = write(tmp_path, "g.json", json.dumps({"id": "q1", "question": "x"}))
    with pytest.raises(GoldenSetError):
        load_golden_set(p)


def test_empty_list_raises(tmp_path):
    p = write(tmp_path, "g.json", "[]")
    with pytest.raises(GoldenSetError):
        load_golden_set(p)


def test_duplicate_ids_raise(tmp_path):
    p = write(tmp_path, "g.json", json.dumps([
        {"id": "q1", "question": "a?", "expected_pages": [1]},
        {"id": "q1", "question": "b?", "expected_pages": [2]},
    ]))
    with pytest.raises(GoldenSetError):
        load_golden_set(p)


def test_blank_question_rejected(tmp_path):
    p = write(tmp_path, "g.json", json.dumps([{"id": "q1", "question": "   "}]))
    with pytest.raises(Exception):
        load_golden_set(p)
