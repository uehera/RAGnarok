"""
Context Management Orchestration Service.

Integrates parent context expansion, token budgeting, and structured prompt formatting
into a unified pipeline service with explicit dependency injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ragnarok.context.budget import TokenBudgetEngine
from ragnarok.context.contract import AssembledContextPayload
from ragnarok.context.expansion import ParentContextExpander, ParentLookupProtocol
from ragnarok.context.formatter import ContextFormatter
from ragnarok.core.config import ContextManagementConfig, GuardrailsConfig
from ragnarok.ingestion.chunker import TokenCounterProtocol
from ragnarok.retrieval.contract import ChildResultCandidate

if TYPE_CHECKING:
    from ragnarok.guardrails.context_guard import ContextSecurityGuard

logger = structlog.get_logger(__name__)


class ContextManagementService:
    """
    High-level service orchestrating Phase 5: Parent Context Expansion & Context Management.
    """

    def __init__(
        self,
        expander: ParentContextExpander,
        budget_engine: TokenBudgetEngine,
        formatter: ContextFormatter | None = None,
        context_guard: ContextSecurityGuard | None = None,
    ) -> None:
        self.expander = expander
        self.budget_engine = budget_engine
        self.formatter = formatter if formatter is not None else ContextFormatter()
        self.context_guard = context_guard

    @classmethod
    def from_config(
        cls,
        parent_lookup: ParentLookupProtocol,
        config: ContextManagementConfig | None = None,
        token_counter: TokenCounterProtocol | None = None,
        guardrails_config: GuardrailsConfig | None = None,
    ) -> ContextManagementService:
        """Factory method constructing ContextManagementService from configuration."""
        cfg = config if config is not None else ContextManagementConfig()
        expander = ParentContextExpander(parent_lookup=parent_lookup)
        budget_engine = TokenBudgetEngine(
            max_tokens=cfg.max_tokens,
            token_counter=token_counter,
        )
        formatter = ContextFormatter()
        context_guard: ContextSecurityGuard | None = None
        if guardrails_config and guardrails_config.enable_context_screening:
            from ragnarok.guardrails.context_guard import ContextSecurityGuard

            context_guard = ContextSecurityGuard(
                config=guardrails_config,
                action=guardrails_config.context_screening_action,
            )
        return cls(
            expander=expander,
            budget_engine=budget_engine,
            formatter=formatter,
            context_guard=context_guard,
        )

    def build_context(
        self,
        candidates: list[ChildResultCandidate],
        target_doc: str | None = None,
    ) -> AssembledContextPayload:
        """
        Executes the full context management pipeline for top-K retrieval candidates:
        1. Expands and deduplicates parent context blocks.
        2. Applies in-flight context security screening (Surface 2.5).
        3. Applies greedy rank-ordered token budgeting.
        4. Formats blocks into delimited source text and generates provenance mappings.
        """
        logger.info(
            "Starting context management assembly",
            input_candidates=len(candidates),
            target_doc=target_doc,
        )

        expanded_parents = self.expander.expand(candidates, target_doc=target_doc)

        screened_threats: list[str] = []
        dropped_count = 0
        if self.context_guard is not None:
            expanded_parents, screening_result = self.context_guard.screen_parents(expanded_parents)
            screened_threats = screening_result.threats_detected
            dropped_count = screening_result.dropped_candidates

        budget_result = self.budget_engine.apply_budget(expanded_parents)
        payload = self.formatter.format(
            selected_parents=budget_result.selected_parents,
            omitted_parents=budget_result.omitted_parents,
            max_tokens_budget=self.budget_engine.max_tokens,
            total_tokens=budget_result.total_tokens,
            token_counts=budget_result.token_counts,
        )
        payload.screened_threats = screened_threats
        payload.dropped_parents_count = dropped_count

        logger.info(
            "Context management assembly complete",
            selected_parents=payload.selected_parent_count,
            omitted_parents=payload.omitted_parent_count,
            total_tokens=payload.total_tokens,
            dropped_threats=dropped_count,
        )
        return payload
