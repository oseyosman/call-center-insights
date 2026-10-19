"""PII Redaction module for call transcripts.

Redacts:
- Emails: [REDACTED_EMAIL]
- Credit/Debit Card Numbers: verified with Luhn check -> [REDACTED_CARD]
- SSN-like patterns (e.g., XXX-XX-XXXX) -> [REDACTED_SSN]
- Phone numbers: US / international variations -> [REDACTED_PHONE]
- Long digit strings (6+ consecutive digits) -> [REDACTED_DIGITS]

LIMITATION NOTICE:
Customer and agent personal names are NOT reliably removed by regex patterns,
as doing so requires heavy contextual Named Entity Recognition (NER) models which
frequently underperform on conversational transcripts and false-positive common nouns.
Do not assume personal names are stripped.
"""

import re
from typing import List, Tuple


def luhn_checksum_valid(number_str: str) -> bool:
    """Verify digits string using the Luhn algorithm (mod 10).

    Valid card lengths are typically between 13 and 19 digits.
    """
    digits = [int(c) for c in number_str if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False

    total = 0
    reverse_digits = digits[::-1]
    for idx, digit in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = digit * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += digit
    return total % 10 == 0


# Email regex: standard RFC 5322 compatible regex
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# SSN regex: standard 3-2-4 pattern with hyphens or spaces
SSN_REGEX = re.compile(
    r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"
)

# Potential Card pattern: 13-19 digits, possibly separated by dashes or spaces
# We look for candidate digit blocks before verifying with Luhn
CARD_CANDIDATE_REGEX = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b"
)

# Phone number regex: 10 or 11 digits formatted with optional country code, parens, hyphens, dots
# Ensures there is a boundary so it does not match inside a longer digit sequence
PHONE_REGEX = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]?\d{4}\b"
    r"|(?<!\d)\d{10}\b"
)

# Standalone long digit strings (6 or more consecutive digits, e.g. account numbers, pins, tokens)
LONG_DIGIT_REGEX = re.compile(
    r"\b\d{6,}\b"
)


def redact_pii(text: str) -> str:
    """Redact PII from a transcript text string.

    Order of operations:
    1. Email addresses -> [REDACTED_EMAIL]
    2. Credit cards (Luhn valid) -> [REDACTED_CARD]
    3. SSN patterns -> [REDACTED_SSN]
    4. Phone numbers -> [REDACTED_PHONE]
    5. Long digit strings (6+ digits) -> [REDACTED_DIGITS]

    Returns redacted text.
    """
    if not text:
        return text

    # 1. Redact Emails
    redacted = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)

    # 2. Redact Valid Credit Cards (Luhn verified)
    def _card_replacer(match: re.Match) -> str:
        candidate = match.group(0)
        digits_only = "".join(c for c in candidate if c.isdigit())
        if 13 <= len(digits_only) <= 19 and luhn_checksum_valid(digits_only):
            return "[REDACTED_CARD]"
        return candidate

    redacted = CARD_CANDIDATE_REGEX.sub(_card_replacer, redacted)

    # 3. Redact SSN-like patterns
    redacted = SSN_REGEX.sub("[REDACTED_SSN]", redacted)

    # 4. Redact Phone numbers
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)

    # 5. Redact remaining long digit strings (>= 6 digits)
    redacted = LONG_DIGIT_REGEX.sub("[REDACTED_DIGITS]", redacted)

    return redacted


def redact_segments(segments: List[dict]) -> List[dict]:
    """Redact PII from a list of transcript segments.

    Each segment is a dict expected to have at least 'text', 'start', 'end', and optional 'speaker'.
    Returns a new list of segments with 'text' redacted.
    """
    cleaned_segments = []
    for seg in segments:
        new_seg = dict(seg)
        if "text" in new_seg and isinstance(new_seg["text"], str):
            new_seg["text"] = redact_pii(new_seg["text"])
        cleaned_segments.append(new_seg)
    return cleaned_segments
