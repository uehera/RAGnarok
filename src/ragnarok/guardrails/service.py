"""
Unified Guardrails Service coordinating Input and Output safety checks.
"""

from __future__ import annotations

from typing import Any

from ragnarok.context.contract import ExpandedParentContext
from ragnarok.core.config import AppConfig, GuardrailsConfig, load_config
from ragnarok.guardrails.context_guard import ContextScreeningResult, ContextSecurityGuard
from ragnarok.guardrails.contract import (
    DocumentGuardDecision,
    InputGuardDecision,
    OutputGuardDecision,
)
from ragnarok.guardrails.document_guard import DocumentSecurityGuard
from ragnarok.guardrails.input_guard import InputGuard
from ragnarok.guardrails.output_guard import OutputGuard, scan_and_redact_secrets


class GuardrailsService:
    """
    Facade coordinating Tier 1 / Tier 2 input guards, ingestion document guards,
    in-flight context screening, and Layer 1 / Layer 2 output guards.
    """

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
        llm_client: Any = None,
    ) -> None:
        self.config = config or GuardrailsConfig()
        self.input_guard = InputGuard(config=self.config, llm_client=llm_client)
        self.output_guard = OutputGuard(config=self.config, llm_client=llm_client)
        self.document_guard = DocumentSecurityGuard(config=self.config)
        self.context_guard = ContextSecurityGuard(
            config=self.config,
            action=self.config.context_screening_action,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        llm_client: Any = None,
    ) -> GuardrailsService:
        """
        Factory creating GuardrailsService from application config.
        """
        cfg = config or load_config()
        return cls(config=cfg.guardrails, llm_client=llm_client)

    def validate_input(self, query: str) -> InputGuardDecision:
        """
        Validates an input query before retrieval or embedding.
        """
        return self.input_guard.validate(query)

    async def validate_input_async(self, query: str) -> InputGuardDecision:
        """
        Asynchronously validates an input query before retrieval or embedding.
        """
        return await self.input_guard.validate_async(query)

    def validate_output(
        self,
        text: str,
        context: str | None = None,
    ) -> OutputGuardDecision:
        """
        Validates and sanitizes synthesized text before terminal presentation.
        """
        return self.output_guard.validate(text=text, context=context)

    def validate_document(self, content: str) -> DocumentGuardDecision:
        """
        Validates and defangs document content to prevent indirect prompt injection.
        """
        return self.document_guard.validate(content)

    def screen_context(
        self,
        parents: list[ExpandedParentContext],
    ) -> tuple[list[ExpandedParentContext], ContextScreeningResult]:
        """
        Screens retrieved parent context chunks in-flight before prompt assembly.
        """
        return self.context_guard.screen_parents(parents)

    def redact_secrets(self, text: str) -> str:
        """
        Deterministic secret redaction convenience helper.
        """
        sanitized, _ = scan_and_redact_secrets(text)
        return sanitized
