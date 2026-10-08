"""Citation validation: the model may only cite sources we actually gave it."""

import re
from dataclasses import dataclass

_GROUP = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")
_ID = re.compile(r"S\d+")


@dataclass
class ValidatedAnswer:
    text: str
    cited_ids: list[str]     # valid, in order of first appearance
    invalid_ids: list[str]   # hallucinated ids that were removed


def validate_citations(answer: str, listed_ids: list[str], valid_ids: set[str]) -> ValidatedAnswer:
    cited: list[str] = []
    invalid: list[str] = []

    def fix(match: re.Match) -> str:
        ids = _ID.findall(match.group(1))
        good = [i for i in ids if i in valid_ids]
        invalid.extend(i for i in ids if i not in valid_ids)
        for i in good:
            if i not in cited:
                cited.append(i)
        return "[" + ", ".join(good) + "]" if good else ""

    text = _GROUP.sub(fix, answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    for i in listed_ids:  # ids the model listed but did not place inline
        if i in valid_ids and i not in cited:
            cited.append(i)
        elif i not in valid_ids and i not in invalid:
            invalid.append(i)
    return ValidatedAnswer(text=text, cited_ids=cited, invalid_ids=sorted(set(invalid)))
