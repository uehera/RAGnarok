"""
Native Output Guardrail Subsystem (Two-Layer Architecture).

Adheres to dev/arc/guardrails.md and dev/workspace/guardrail.md:
- Layer 1: Deterministic secret and PII scanner redacting sensitive tokens
          (API keys, DB connection strings, passwords, emails, IPs) with [REDACTED <TYPE>].
- Layer 2: Grounding and faithfulness verification verifying assertions against retrieved
          parent context blocks.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from ragnarok.core.config import GuardrailsConfig
from ragnarok.guardrails.contract import (
    UNGROUNDED_VIOLATION_MESSAGE,
    OutputGuardDecision,
)

logger = structlog.get_logger(__name__)

# Output Secret & PII Detection Patterns
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_ANTHROPIC_KEY_PATTERN = re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}")
_OPENAI_KEY_PATTERN = re.compile(r"sk-(?!ant-)[a-zA-Z0-9_-]{20,}")
_GROQ_KEY_PATTERN = re.compile(r"gsk_[a-zA-Z0-9_-]{20,}")
_GENERIC_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token)\s*[:=]\s*(?:(api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token)\s*[:=]\s*)?['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?"
)
_DATABASE_URI_PATTERN = re.compile(
    r"(?i)(?:postgresql|postgres|mysql|mongodb|redis|sqlite|duckdb)(?:\+[a-z]+)?:\/\/[^\s\"'<>]+"
)
_PASSWORD_PATTERN = re.compile(
    r"(?i)(password|passwd|secret)\s*[:=]\s*(?:(?:password|passwd|secret)\s*[:=]\s*)?['\"]?([^\s'\"\n]{6,})['\"]?"
)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Common Standardized Refusal Phrases
_STANDARD_REFUSAL_MARKERS = [
    "i cannot find sufficient information",
    "i could not find sufficient factual support",
    "not found in the provided documents",
    "insufficient information in the provided context",
]


def scan_and_redact_secrets(text: str) -> tuple[str, list[str]]:
    """
    Deterministic procedural scanner replacing exposed credentials and PII with redaction tags.

    Returns:
        tuple[str, list[str]]: (Sanitized text, list of detected secret types)
    """
    if not text:
        return "", []

    detected_types: list[str] = []
    sanitized = _ANSI_ESCAPE_PATTERN.sub("", text)

    if _ANTHROPIC_KEY_PATTERN.search(sanitized):
        detected_types.append("ANTHROPIC_KEY")
        sanitized = _ANTHROPIC_KEY_PATTERN.sub("[REDACTED ANTHROPIC_KEY]", sanitized)

    if _OPENAI_KEY_PATTERN.search(sanitized):
        detected_types.append("OPENAI_KEY")
        sanitized = _OPENAI_KEY_PATTERN.sub("[REDACTED OPENAI_KEY]", sanitized)

    if _GROQ_KEY_PATTERN.search(sanitized):
        detected_types.append("GROQ_KEY")
        sanitized = _GROQ_KEY_PATTERN.sub("[REDACTED GROQ_KEY]", sanitized)

    if _DATABASE_URI_PATTERN.search(sanitized):
        detected_types.append("DATABASE_URL")
        sanitized = _DATABASE_URI_PATTERN.sub("[REDACTED DATABASE_URL]", sanitized)

    if _GENERIC_KEY_PATTERN.search(sanitized):
        detected_types.append("API_KEY")
        sanitized = _GENERIC_KEY_PATTERN.sub(r"\1: [REDACTED API_KEY]", sanitized)

    if _PASSWORD_PATTERN.search(sanitized):
        detected_types.append("PASSWORD")
        sanitized = _PASSWORD_PATTERN.sub(r"\1: [REDACTED PASSWORD]", sanitized)

    if _EMAIL_PATTERN.search(sanitized):
        detected_types.append("EMAIL")
        sanitized = _EMAIL_PATTERN.sub("[REDACTED EMAIL]", sanitized)

    if _IPV4_PATTERN.search(sanitized):
        detected_types.append("IP_ADDRESS")
        sanitized = _IPV4_PATTERN.sub("[REDACTED IP_ADDRESS]", sanitized)

    return sanitized, detected_types


class SpeculativeStreamSanitizer:
    """
    Sliding window stream sanitizer buffering tokens during live streaming.
    Prevents credentials, secrets, and PII from leaking to the terminal across token boundaries.
    """

    def __init__(self, window_size: int = 40) -> None:
        self.window_size = window_size
        self._buffer: str = ""
        self.detected_secrets: list[str] = []

    def feed(self, token: str) -> str:
        """
        Ingests a streamed token, runs sliding window secret redaction,
        and returns safe printable text (leaving a trailing window).
        """
        self._buffer += token

        # If buffer is shorter than window, hold it back
        if len(self._buffer) <= self.window_size:
            return ""

        # Check if the buffer contains any completed secrets
        sanitized_chunk, detected = scan_and_redact_secrets(self._buffer)
        for s in detected:
            if s not in self.detected_secrets:
                self.detected_secrets.append(s)

        # Retain trailing window in buffer
        safe_prefix_length = max(0, len(sanitized_chunk) - self.window_size)
        to_emit = sanitized_chunk[:safe_prefix_length]
        self._buffer = sanitized_chunk[safe_prefix_length:]
        return to_emit

    def flush(self) -> str:
        """
        Flushes and redacts all remaining tokens in the buffer upon stream completion.
        """
        sanitized, detected = scan_and_redact_secrets(self._buffer)
        for s in detected:
            if s not in self.detected_secrets:
                self.detected_secrets.append(s)
        self._buffer = ""
        return sanitized


class OutputGuard:
    """
    Two-Layer Output Guardrail Service coordinating deterministic secret redaction
    and context grounding verification (LiteLLM prompt or local DeBERTa NLI).
    """

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
        llm_client: Any = None,
        nli_classifier: Any = None,
    ) -> None:
        self.config = config or GuardrailsConfig()
        self.llm_client = llm_client
        self.nli_classifier = nli_classifier

    def validate(self, text: str, context: str | None = None) -> OutputGuardDecision:
        """
        Validates synthesized text through Layer 1 secret redaction and Layer 2 grounding checks.
        """
        if not self.config.enabled:
            return OutputGuardDecision(
                is_grounded=True,
                redacted_text=text,
                secrets_detected=[],
                unsupported_claims=[],
            )

        # 1. Layer 1: Secret & PII Scanning
        sanitized_text, detected_secrets = scan_and_redact_secrets(text)
        if detected_secrets:
            logger.info("Output Guard: Secrets/PII redacted from output", types=detected_secrets)

        # 2. Layer 2: Grounding & Faithfulness Verification
        is_grounded = True
        fallback_msg: str | None = None
        unsupported_claims: list[str] = []

        if self.config.enable_grounding_check and context:
            is_grounded, unsupported_claims = self._verify_grounding(sanitized_text, context)
            if not is_grounded:
                logger.warning(
                    "Output Guard: Ungrounded claims detected in synthesized response",
                    claims_count=len(unsupported_claims),
                )
                fallback_msg = UNGROUNDED_VIOLATION_MESSAGE
                sanitized_text = UNGROUNDED_VIOLATION_MESSAGE

        return OutputGuardDecision(
            is_grounded=is_grounded,
            redacted_text=sanitized_text,
            secrets_detected=detected_secrets,
            fallback_message=fallback_msg,
            unsupported_claims=unsupported_claims,
        )

    def _verify_grounding(self, text: str, context: str) -> tuple[bool, list[str]]:
        """
        Verifies whether the text is grounded in the provided context blocks.
        Returns a tuple of (is_grounded, list_of_unsupported_claims).
        """
        # Refusal statements are grounded by definition
        text_lower = text.lower()
        if any(marker in text_lower for marker in _STANDARD_REFUSAL_MARKERS):
            return True, []

        if not context or not context.strip():
            return False, ["No context documents provided to verify claims."]

        if self.config.classifier_backend == "deberta":
            return self._verify_grounding_deberta(text, context)

        if self.llm_client is not None and hasattr(self.llm_client, "verify_grounding"):
            res = self.llm_client.verify_grounding(text, context)
            if isinstance(res, tuple):
                return bool(res[0]), list(res[1])
            return bool(res), [] if bool(res) else ["Ungrounded claims identified by custom evaluator."]

        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.set_verbose = False

            prompt = (
                "You are an evaluator checking whether an answer is fully grounded in the provided context.\n"
                f"Context:\n{context}\n\n"
                f"Answer:\n{text}\n\n"
                "Are all factual claims in the answer supported by the context? "
                "Respond ONLY with 'YES' or 'NO'."
            )
            response = litellm.completion(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
                timeout=self.config.timeout_seconds,
            )
            verdict = (response.choices[0].message.content or "").strip().upper()
            is_valid = "YES" in verdict
            unsupported = (
                []
                if is_valid
                else ["Synthesized statements contradict or extrapolate beyond provided context sources."]
            )
            return is_valid, unsupported
        except Exception as exc:
            logger.warning("Output Guard: Grounding evaluation call failed; failing open", error=str(exc))
            return True, []

    def _verify_grounding_deberta(self, text: str, context: str) -> tuple[bool, list[str]]:
        """Verifies grounding using local cross-encoder/nli-deberta-v3-base."""
        from ragnarok.guardrails.deberta_classifiers import DebertaNLIGroundingClassifier

        classifier = self.nli_classifier
        if classifier is None:
            classifier = DebertaNLIGroundingClassifier(
                model_name=self.config.nli_grounding_model
            )
            self.nli_classifier = classifier

        scores = classifier.predict_grounding_score(premise=context, hypothesis=text)
        contradiction = scores.get("contradiction", 0.0)
        is_valid = contradiction <= self.config.contradiction_threshold
        unsupported = (
            []
            if is_valid
            else [f"NLI Contradiction score ({contradiction:.2f}) exceeded threshold ({self.config.contradiction_threshold:.2f})."]
        )
        return is_valid, unsupported
