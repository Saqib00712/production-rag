"""
Lightweight, regex-based PII detection.

Deliberately informational, not blocking: this is a document Q&A system
over documents like resumes, where an email or phone number is often the
POINT of the content (the user WANTS to ask "what's this candidate's
email?"). Blocking or redacting on sight would break the product. What this
guardrail DOES do is make what's flowing through the pipeline visible and
logged, so a real decision (redact before sending to OpenAI? warn the
uploader? nothing?) can be made deliberately per deployment, via
`settings.warn_on_pii`, rather than silently.

Security detail: we report COUNTS per category, never the matched VALUES.
A PII detector that logs the actual email/phone/SSN it found would just
become a second place those secrets are now sitting in your log stream.
"""

import re

_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def _luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def detect_pii(text: str) -> dict[str, int]:
    """Returns {category: match_count} for categories actually found.
    Credit-card-shaped numbers are Luhn-checked to cut false positives
    (e.g. a long invoice or phone number is NOT a valid card number)."""
    if not text:
        return {}
    counts: dict[str, int] = {}
    for category, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        if category == "credit_card":
            matches = [m for m in matches if _luhn_valid(re.sub(r"[ -]", "", m))]
        if matches:
            counts[category] = len(matches)
    return counts


def merge_counts(all_counts: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in all_counts:
        for k, v in counts.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def redact_pii(text: str) -> str:
    """
    Day 21: replaces each detected PII span with a `[REDACTED_<CATEGORY>]`
    marker, in place, so the ORIGINAL value never reaches chunking,
    embedding, or OpenAI at all -- a strictly stronger guarantee than
    `detect_pii`'s "count and warn" (Day 7), which deliberately never
    touches the content. Opt-in via `settings.pii_mode = "redact"`
    (app/services/ingestion.py) because it is a lossy, irreversible
    transformation: a resume's email address, which a user may genuinely
    want to ask the bot about, is destroyed just like a leaked SSN would be.

    Credit cards are only redacted when Luhn-valid, for the same
    false-positive reason `detect_pii` applies the same check -- a random
    16-digit invoice number is not a card number and redacting it would
    just make the document CITATIONS inaccurate for no security benefit.

    Order matters here in a way it doesn't for `detect_pii` (which only
    COUNTS matches against the untouched original text): the phone pattern
    is broad enough to also match pieces of a credit-card-shaped number, so
    redacting phone-shaped spans FIRST would chew up a card number into
    several `[REDACTED_PHONE]` fragments before the credit-card pattern
    ever got a chance to see the whole thing. Processing credit_card (the
    most specific pattern) before the broader phone pattern avoids that.
    """
    if not text:
        return text
    for category in ("credit_card", "ssn", "email", "phone"):
        pattern = _PATTERNS[category]
        if category == "credit_card":
            text = pattern.sub(
                lambda m: f"[REDACTED_{category.upper()}]" if _luhn_valid(re.sub(r"[ -]", "", m.group(0)))
                else m.group(0),
                text,
            )
        else:
            text = pattern.sub(f"[REDACTED_{category.upper()}]", text)
    return text
