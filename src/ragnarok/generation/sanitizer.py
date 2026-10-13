"""
Minimal Output Sanitization and Secret Redaction Subsystem.

Deterministic output protection adhering to dev/arc/guardrails.md:
1. Regex scanner detecting exposed API keys, database connection strings, credentials, and PII.
2. Replaces sensitive matches with standardized tags: [REDACTED <TYPE>].
3. Strips ANSI escape codes and terminal artifacts for clean CLI rendering.
"""

from __future__ import annotations

import re

# Regex patterns for deterministic secret and PII detection
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_OPENAI_KEY_PATTERN = re.compile(r"sk-[a-zA-Z0-9_-]{20,}")
_ANTHROPIC_KEY_PATTERN = re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}")
_GROQ_KEY_PATTERN = re.compile(r"gsk_[a-zA-Z0-9_-]{20,}")
_GENERIC_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?"
)
_DATABASE_URI_PATTERN = re.compile(
    r"(?i)(?:postgresql|postgres|mysql|mongodb|redis):\/\/[^\s\"'<>]+"
)
_PASSWORD_PATTERN = re.compile(
    r"(?i)(password|passwd|secret)\s*[:=]\s*['\"]?([^\s'\"\n]{6,})['\"]?"
)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def sanitize_generated_text(text: str) -> str:
    """
    Pure procedural function that scans text and redacts credentials, connection strings,
    and sensitive tokens with explicit redaction tags.

    Parameters:
        text: Raw synthesized text stream or completed response.

    Returns:
        Sanitized text safe for terminal rendering and evaluation auditing.
    """
    if not text:
        return ""

    sanitized = _ANSI_ESCAPE_PATTERN.sub("", text)

    # Specific API key formats
    sanitized = _OPENAI_KEY_PATTERN.sub("[REDACTED OPENAI_KEY]", sanitized)
    sanitized = _ANTHROPIC_KEY_PATTERN.sub("[REDACTED ANTHROPIC_KEY]", sanitized)
    sanitized = _GROQ_KEY_PATTERN.sub("[REDACTED GROQ_KEY]", sanitized)

    # Database URIs
    sanitized = _DATABASE_URI_PATTERN.sub("[REDACTED DATABASE_URL]", sanitized)

    # Generic key / token key-value pairs
    sanitized = _GENERIC_KEY_PATTERN.sub(r"\1: [REDACTED API_KEY]", sanitized)

    # Password key-value pairs
    sanitized = _PASSWORD_PATTERN.sub(r"\1: [REDACTED PASSWORD]", sanitized)

    # PII: Email addresses
    sanitized = _EMAIL_PATTERN.sub("[REDACTED EMAIL]", sanitized)

    # Network identifiers: IPv4 addresses
    sanitized = _IPV4_PATTERN.sub("[REDACTED IP_ADDRESS]", sanitized)

    return sanitized
