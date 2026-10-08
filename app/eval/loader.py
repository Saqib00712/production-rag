"""Load a golden question set from a JSON file.

JSON, not YAML: one fewer dependency, and the schema is simple enough that
YAML's readability edge doesn't pay for itself here. Swap in PyYAML later
if your golden sets grow large enough to want comments.
"""

import json
from pathlib import Path

from app.eval.models import GoldenQuestion


class GoldenSetError(Exception):
    pass


def load_golden_set(path: str | Path) -> list[GoldenQuestion]:
    p = Path(path)
    if not p.exists():
        raise GoldenSetError(f"Golden set not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise GoldenSetError(f"{p} must contain a JSON array of questions")
    questions = [GoldenQuestion.model_validate(item) for item in raw]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise GoldenSetError(f"Duplicate question id(s) in {p}: {dupes}")
    if not questions:
        raise GoldenSetError(f"{p} contains no questions")
    return questions
