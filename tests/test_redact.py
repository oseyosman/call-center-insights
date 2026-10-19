import pytest
from callinsight.redact import (
    luhn_checksum_valid,
    redact_pii,
    redact_segments,
)


def test_luhn_checksum_valid():
    # Valid credit card numbers (test numbers)
    assert luhn_checksum_valid("4532015112830366") is True
    assert luhn_checksum_valid("49927398716") is False  # 11 digits, invalid length
    assert luhn_checksum_valid("499273987160000") is True  # 15 digits Amex test with valid luhn
    # Altered last digit
    assert luhn_checksum_valid("4532015112830367") is False


def test_redact_emails():
    text = "Please reach out at support@company.org or admin.user+tag@domain.co.uk."
    redacted = redact_pii(text)
    assert "support@company.org" not in redacted
    assert "admin.user+tag@domain.co.uk" not in redacted
    assert redacted.count("[REDACTED_EMAIL]") == 2


def test_redact_credit_cards():
    # Valid Visa test number with spaces
    text = "Charged to card 4532 0151 1283 0366 yesterday."
    redacted = redact_pii(text)
    assert "4532 0151 1283 0366" not in redacted
    assert "[REDACTED_CARD]" in redacted


def test_redact_invalid_credit_card_not_marked_card():
    # 16 digits but fails Luhn check
    text = "Reference number is 4532015112830367."
    redacted = redact_pii(text)
    # Should not be marked as card, but will be marked as long digit string
    assert "[REDACTED_CARD]" not in redacted
    assert "[REDACTED_DIGITS]" in redacted


def test_redact_ssn():
    text = "My SSN is 000-12-3456 and my spouse's is 999 88 7777."
    redacted = redact_pii(text)
    assert "000-12-3456" not in redacted
    assert "999 88 7777" not in redacted
    assert redacted.count("[REDACTED_SSN]") == 2


def test_redact_phones():
    text = (
        "Call me at (555) 019-2834, or my direct line +1-800-555-0199, "
        "or cell 555-234-5678."
    )
    redacted = redact_pii(text)
    assert "(555) 019-2834" not in redacted
    assert "+1-800-555-0199" not in redacted
    assert "555-234-5678" not in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_redact_long_digit_strings():
    text = "Order confirmation ID is 987654321 and pin 12345."
    redacted = redact_pii(text)
    assert "987654321" not in redacted
    assert "[REDACTED_DIGITS]" in redacted
    # 5 digits is not long digit string
    assert "12345" in redacted


def test_redact_segments():
    segments = [
        {"start": 0.0, "end": 2.5, "speaker": "customer", "text": "Hi, my phone is 555-123-4567."},
        {"start": 2.6, "end": 4.0, "speaker": "agent", "text": "Thank you, how can I help?"},
    ]
    cleaned = redact_segments(segments)
    assert len(cleaned) == 2
    assert "555-123-4567" not in cleaned[0]["text"]
    assert "[REDACTED_PHONE]" in cleaned[0]["text"]
    assert cleaned[0]["start"] == 0.0
    assert cleaned[1]["text"] == "Thank you, how can I help?"
