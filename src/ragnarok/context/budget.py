"""
Token Budgeting Engine for Context Management.

Enforces context token ceiling (context_management.max_tokens) against ordered parent blocks.
Strict Invariants:
1. Whole-Parent Integrity: Never slice, truncate, split, or summarize parent blocks across token boundaries.
2. Relevance-Driven Cutoff: Parents are evaluated in strict best_child_rank order.
3. No Backfilling: When a parent exceeds the remaining budget, selection halts immediately without
   skipping or backfilling smaller lower-ranked parents.
"""

from __future__ import annotations

from typing import NamedTuple

import structlog

from ragnarok.context.contract import ExpandedParentContext
from ragnarok.ingestion.chunker import TokenCounterProtocol, resolve_token_counter

logger = structlog.get_logger(__name__)


class BudgetResult(NamedTuple):
    """Result of token budget evaluation across candidate parent blocks."""

    selected_parents: list[ExpandedParentContext]
    omitted_parents: list[ExpandedParentContext]
    token_counts: list[int]
    total_tokens: int


def budget_parent_contexts(
    parents: list[ExpandedParentContext],
    max_tokens: int,
    token_counter: TokenCounterProtocol,
) -> BudgetResult:
    """
    Pure procedural function that accumulates whole parent blocks in ranked order
    until the token budget ceiling is reached.

    Parameters:
        parents: Ordered list of expanded parent chunks (ascending by best_child_rank).
        max_tokens: Maximum allowed total tokens for the assembled context.
        token_counter: Token counter implementation adhering to TokenCounterProtocol.

    Returns:
        BudgetResult containing selected parents, omitted parents, individual token counts,
        and total accumulated tokens.
    """
    selected: list[ExpandedParentContext] = []
    omitted: list[ExpandedParentContext] = []
    token_counts: list[int] = []
    total_tokens = 0

    halted = False

    for parent in parents:
        if halted:
            omitted.append(parent)
            continue

        cost = token_counter.count_tokens(parent.content)

        if total_tokens + cost <= max_tokens:
            selected.append(parent)
            token_counts.append(cost)
            total_tokens += cost
        else:
            # Budget exceeded: Halt immediately to prevent slicing or cherry-picking
            logger.info(
                "Context token budget reached; halting selection",
                max_tokens=max_tokens,
                current_total=total_tokens,
                parent_cost=cost,
                parent_chunk_id=str(parent.parent_chunk_id),
                best_child_rank=parent.best_child_rank,
            )
            halted = True
            omitted.append(parent)

    logger.info(
        "Token budgeting complete",
        total_parents=len(parents),
        selected_count=len(selected),
        omitted_count=len(omitted),
        total_tokens=total_tokens,
        max_tokens=max_tokens,
    )

    return BudgetResult(
        selected_parents=selected,
        omitted_parents=omitted,
        token_counts=token_counts,
        total_tokens=total_tokens,
    )


class TokenBudgetEngine:
    """
    Stateful service managing context token limits and parent accumulation.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        token_counter: TokenCounterProtocol | None = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.max_tokens = max_tokens
        self.token_counter = (
            token_counter
            if token_counter is not None
            else resolve_token_counter(
                size_counter="tiktoken",  # Fast reliable fallback default
                model_name=model_name,
            )
        )

    def apply_budget(self, parents: list[ExpandedParentContext]) -> BudgetResult:
        """Applies configured max_tokens budget against candidate parents."""
        return budget_parent_contexts(
            parents=parents,
            max_tokens=self.max_tokens,
            token_counter=self.token_counter,
        )
