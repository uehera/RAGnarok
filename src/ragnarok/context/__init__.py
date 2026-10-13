"""
Context Management Subsystem of RAGnarok.

Exports:
- Parent context expansion (ParentContextExpander, expand_parent_contexts, ParentLookupProtocol)
- Token budgeting engine (TokenBudgetEngine, budget_parent_contexts, BudgetResult)
- Context formatting and provenance mapping (ContextFormatter, assemble_context_payload, format_heading_path)
- Orchestration service (ContextManagementService)
- Data contracts (ExpandedParentContext, MatchedChildReference, ContextBlock, AssembledContextPayload)
"""

from __future__ import annotations

from ragnarok.context.budget import (
    BudgetResult,
    TokenBudgetEngine,
    budget_parent_contexts,
)
from ragnarok.context.contract import (
    AssembledContextPayload,
    ContextBlock,
    ExpandedParentContext,
    MatchedChildReference,
)
from ragnarok.context.expansion import (
    ParentContextExpander,
    ParentLookupProtocol,
    expand_parent_contexts,
)
from ragnarok.context.formatter import (
    ContextFormatter,
    assemble_context_payload,
    format_heading_path,
)
from ragnarok.context.service import ContextManagementService

__all__ = [
    # Contracts
    "MatchedChildReference",
    "ExpandedParentContext",
    "ContextBlock",
    "AssembledContextPayload",
    # Expansion
    "ParentLookupProtocol",
    "expand_parent_contexts",
    "ParentContextExpander",
    # Budgeting
    "BudgetResult",
    "budget_parent_contexts",
    "TokenBudgetEngine",
    # Formatting
    "format_heading_path",
    "assemble_context_payload",
    "ContextFormatter",
    # Service
    "ContextManagementService",
]
