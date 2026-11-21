"""
Minimal Output Sanitization and Secret Redaction Subsystem.

Deterministic output protection adhering to dev/arc/guardrails.md:
1. Regex scanner detecting exposed API keys, database connection strings, credentials, and PII.
2. Replaces sensitive matches with standardized tags: [REDACTED <TYPE>].
3. Strips ANSI escape codes and terminal artifacts for clean CLI rendering.
"""

from __future__ import annotations

from ragnarok.guardrails.output_guard import SpeculativeStreamSanitizer, scan_and_redact_secrets

__all__ = ["sanitize_generated_text", "SpeculativeStreamSanitizer"]


def sanitize_generated_text(text: str) -> str:
    """
    Pure procedural function that scans text and redacts credentials, connection strings,
    and sensitive tokens with explicit redaction tags.

    Parameters:
        text: Raw synthesized text stream or completed response.

    Returns:
        Sanitized text safe for terminal rendering and evaluation auditing.
    """
    sanitized, _ = scan_and_redact_secrets(text)
    return sanitized
