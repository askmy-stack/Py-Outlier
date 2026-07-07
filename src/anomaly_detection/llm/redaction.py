"""PII redaction utilities before sending context to an LLM."""

from __future__ import annotations

import re

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_pii(text: str) -> str:
    """Replace common PII patterns with redaction placeholders."""
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    redacted = _PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    return _SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
