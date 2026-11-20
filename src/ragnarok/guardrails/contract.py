"""
Native Guardrails Data Contracts and Schemas.

Defines Pydantic v2 schemas and standardized user-facing refusal notices
adhering to dev/arc/guardrails.md and dev/workspace/guardrail.md.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Standardized User-Facing Refusal Messages
LENGTH_VIOLATION_MESSAGE = (
    "Your query exceeds the maximum allowed length (1,024 tokens). "
    "Please submit a more concise question."
)
INJECTION_VIOLATION_MESSAGE = (
    "Your query could not be processed due to safety policy restrictions."
)
OFF_TOPIC_VIOLATION_MESSAGE = (
    "I am specialized only in answering questions related to the indexed document corpus."
)
UNGROUNDED_VIOLATION_MESSAGE = (
    "I could not find sufficient factual support in the provided documents to answer this question."
)


class InputGuardDecision(BaseModel):
    """
    Standardized validation decision for input queries before retrieval or embedding.
    """

    is_safe: bool = Field(
        description="Whether the query passed security, length, and injection checks."
    )
    is_topical: bool = Field(
        default=True,
        description="Whether the query is on-topic and aligned with the document corpus.",
    )
    violation_type: str | None = Field(
        default=None,
        description="Type of violation detected: 'length', 'control_char', 'injection', 'off_topic', or None.",
    )
    fallback_message: str | None = Field(
        default=None,
        description="Standardized user-facing refusal message if validation failed.",
    )
    tier: str = Field(
        default="tier1",
        description="The guardrail tier that made the final decision: 'tier1', 'tier2', or 'bypassed'.",
    )


class DocumentSecurityError(Exception):
    """Raised when an ingested document contains forbidden injection directives in reject mode."""


class DocumentGuardDecision(BaseModel):
    """
    Standardized validation decision for ingested documents before chunking and persistence.
    """

    is_safe: bool = Field(
        description="Whether the document content is free from indirect prompt injection directives."
    )
    defanged_content: str = Field(
        description="Document content with any detected injection directives defanged or sanitized."
    )
    injections_detected: list[str] = Field(
        default_factory=list,
        description="List of detected injection pattern tags (e.g., 'SYSTEM_OVERRIDE', 'IGNORE_INSTRUCTIONS').",
    )
    quarantined: bool = Field(
        default=False,
        description="Whether the document was quarantined and isolated to the quarantine directory.",
    )
    quarantine_path: str | None = Field(
        default=None,
        description="Filesystem path where the quarantined document and metadata are saved.",
    )
    threat_excerpts: list[str] = Field(
        default_factory=list,
        description="Excerpts of detected injection directives for forensic review.",
    )


class OutputGuardDecision(BaseModel):
    """
    Standardized validation decision for synthesized text before terminal presentation.
    """

    is_grounded: bool = Field(
        default=True,
        description="Whether synthesized factual claims are grounded in retrieved context.",
    )
    redacted_text: str = Field(
        description="Sanitized text with exposed secrets, credentials, and PII masked.",
    )
    secrets_detected: list[str] = Field(
        default_factory=list,
        description="List of secret categories detected and redacted (e.g., 'API_KEY', 'DATABASE_URL').",
    )
    fallback_message: str | None = Field(
        default=None,
        description="Standardized refusal message if output grounding verification failed.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List of specific claims flagged as unsupported during grounding check.",
    )
