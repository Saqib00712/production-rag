from app.services.pii import detect_pii, merge_counts, redact_pii


def test_detects_email():
    assert detect_pii("Contact me at jane.doe@example.com please")["email"] == 1


def test_detects_multiple_emails():
    assert detect_pii("a@x.com and b@y.com")["email"] == 2


def test_detects_ssn_format():
    assert detect_pii("SSN: 123-45-6789 on file")["ssn"] == 1


def test_detects_phone_like_number():
    result = detect_pii("Call us at 555-123-4567 anytime")
    assert result.get("phone", 0) >= 1


def test_credit_card_requires_luhn_validity():
    # A real (test) Luhn-valid Visa number vs. a random 16-digit non-valid one
    valid = detect_pii("Card: 4111 1111 1111 1111")
    invalid = detect_pii("Order number: 1234 5678 9012 3456")
    assert valid.get("credit_card", 0) == 1
    assert invalid.get("credit_card", 0) == 0


def test_no_pii_in_clean_text():
    assert detect_pii("This document discusses general shipping policy.") == {}


def test_empty_text():
    assert detect_pii("") == {}


def test_merge_counts_sums_across_pages():
    merged = merge_counts([{"email": 1}, {"email": 2, "ssn": 1}, {}])
    assert merged == {"email": 3, "ssn": 1}


def test_detector_never_returns_the_matched_value_itself():
    result = detect_pii("email: secret.person@company.com")
    assert "secret.person@company.com" not in str(result)  # counts only, never the value


# --- Day 21: redaction ---

def test_redact_pii_masks_email():
    out = redact_pii("Contact jane.doe@example.com for details")
    assert "jane.doe@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_pii_masks_ssn():
    out = redact_pii("SSN: 123-45-6789 on file")
    assert "123-45-6789" not in out
    assert "[REDACTED_SSN]" in out


def test_redact_pii_only_masks_luhn_valid_credit_cards():
    valid = redact_pii("Card: 4111 1111 1111 1111")
    invalid = redact_pii("Order number: 1234 5678 9012 3456")
    assert "4111 1111 1111 1111" not in valid and "[REDACTED_CREDIT_CARD]" in valid
    # Not Luhn-valid, so never tagged CREDIT_CARD -- this matches detect_pii's own existing
    # Luhn gate (test_credit_card_requires_luhn_validity above). The broad phone pattern may
    # still flag digit runs like this as a false-positive phone number, same as it already does
    # for detect_pii -- that pre-existing, documented loose-matching behavior is unchanged here.
    assert "[REDACTED_CREDIT_CARD]" not in invalid


def test_redact_pii_handles_multiple_categories_in_one_pass():
    out = redact_pii("Email a@b.com or call 555-123-4567, SSN 123-45-6789")
    assert "a@b.com" not in out and "123-45-6789" not in out
    assert "[REDACTED_EMAIL]" in out and "[REDACTED_SSN]" in out


def test_redact_pii_leaves_clean_text_untouched():
    text = "This document discusses general shipping policy."
    assert redact_pii(text) == text


def test_redact_pii_empty_text():
    assert redact_pii("") == ""
