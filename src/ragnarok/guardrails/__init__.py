"""
Native Guardrails Subsystem for RAGnarok.

Pure Python, framework-free security architecture:
- Tier 1: Deterministic fast-path regex and length validation.
- Tier 2: Semantic prompt injection and domain topicality validation.
- Output Layer 1: Deterministic secret and PII redaction.
- Output Layer 2: Context grounding and faithfulness verification.
"""

from __future__ import annotations

from ragnarok.guardrails.context_guard import (
    ContextScreeningResult,
    ContextSecurityGuard,
)
from ragnarok.guardrails.contract import (
    INJECTION_VIOLATION_MESSAGE,
    LENGTH_VIOLATION_MESSAGE,
    OFF_TOPIC_VIOLATION_MESSAGE,
    UNGROUNDED_VIOLATION_MESSAGE,
    DocumentGuardDecision,
    DocumentSecurityError,
    InputGuardDecision,
    OutputGuardDecision,
    format_guardrail_refusal,
)
from ragnarok.guardrails.deberta_classifiers import (
    DebertaNLIGroundingClassifier,
    DebertaPromptInjectionClassifier,
    NLIEntailmentClassifierProtocol,
    PromptInjectionClassifierProtocol,
)
from ragnarok.guardrails.document_guard import (
    DocumentSecurityGuard,
    scan_and_defang_document,
)
from ragnarok.guardrails.input_guard import (
    InputGuard,
    check_tier1_fastpath,
    extract_normalized_variants,
)
from ragnarok.guardrails.output_guard import (
    OutputGuard,
    SpeculativeStreamSanitizer,
    scan_and_redact_secrets,
)
from ragnarok.guardrails.service import GuardrailsService

__all__ = [
    "ContextScreeningResult",
    "ContextSecurityGuard",
    "GuardrailsService",
    "InputGuard",
    "OutputGuard",
    "DebertaPromptInjectionClassifier",
    "DebertaNLIGroundingClassifier",
    "PromptInjectionClassifierProtocol",
    "NLIEntailmentClassifierProtocol",
    "SpeculativeStreamSanitizer",
    "DocumentSecurityGuard",
    "InputGuardDecision",
    "OutputGuardDecision",
    "DocumentGuardDecision",
    "DocumentSecurityError",
    "format_guardrail_refusal",
    "check_tier1_fastpath",
    "extract_normalized_variants",
    "scan_and_redact_secrets",
    "scan_and_defang_document",
    "LENGTH_VIOLATION_MESSAGE",
    "INJECTION_VIOLATION_MESSAGE",
    "OFF_TOPIC_VIOLATION_MESSAGE",
    "UNGROUNDED_VIOLATION_MESSAGE",
]
