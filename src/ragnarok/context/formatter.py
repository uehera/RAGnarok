"""
Structured Context Block Assembly and Provenance Mapping.

Formats budgeted parent sections into delimited source blocks:
    [Source X] (Document: <filename>, Section: <heading_path>)
    <parent_content>

Constructs the internal provenance lookup table mapping [Source X] back to
document IDs, parent chunk IDs, and matched child chunk references.
Adheres strictly to dev/workspace/context_management.md.
"""

from __future__ import annotations

from typing import Any

import structlog

from ragnarok.context.contract import (
    AssembledContextPayload,
    ContextBlock,
    ExpandedParentContext,
)

logger = structlog.get_logger(__name__)


def format_heading_path(heading_path: list[str]) -> str:
    """Formats heading path breadcrumbs into a human-readable section breadcrumb string."""
    if not heading_path:
        return "Root Section"
    return " > ".join(heading_path)


def assemble_context_payload(
    selected_parents: list[ExpandedParentContext],
    omitted_parents: list[ExpandedParentContext],
    max_tokens_budget: int,
    total_tokens: int | None = None,
    token_counts: list[int] | None = None,
    block_delimiter: str = "\n\n---\n\n",
) -> AssembledContextPayload:
    """
    Pure procedural function assembling budgeted parent chunks into formatted prompt text
    and structured provenance metadata.

    Parameters:
        selected_parents: List of parents chosen within token budget.
        omitted_parents: List of parents excluded due to budget limit.
        max_tokens_budget: Configured max token ceiling.
        total_tokens: Precomputed total token count (or calculated if omitted).
        token_counts: Precomputed per-block token counts.
        block_delimiter: Delimiter between formatted [Source X] blocks.

    Returns:
        AssembledContextPayload conforming to standardized data contracts.
    """
    if not selected_parents:
        logger.debug("No parents selected; returning empty assembled payload")
        return AssembledContextPayload(
            total_tokens=0,
            max_tokens_budget=max_tokens_budget,
            selected_parent_count=0,
            omitted_parent_count=len(omitted_parents),
            assembled_context_text="",
            context_blocks=[],
            provenance_map={},
        )

    formatted_text_blocks: list[str] = []
    context_blocks: list[ContextBlock] = []
    provenance_map: dict[str, dict[str, Any]] = {}

    for i, parent in enumerate(selected_parents, start=1):
        source_label = f"[Source {i}]"
        section_str = format_heading_path(parent.heading_path)
        block_tokens = (
            token_counts[i - 1] if token_counts is not None and i - 1 < len(token_counts) else 0
        )

        # Standardized source header and block text
        # Format:
        # [Source X] (Document: <filename>, Section: <heading_path>)
        # <parent_content>
        header_line = f"{source_label} (Document: {parent.filename}, Section: {section_str})"
        formatted_block_str = f"{header_line}\n{parent.content}"
        formatted_text_blocks.append(formatted_block_str)

        # Create structured ContextBlock
        block_obj = ContextBlock(
            source_index=i,
            document_id=parent.document_id,
            filename=parent.filename,
            parent_chunk_id=parent.parent_chunk_id,
            parent_index=parent.parent_index,
            heading_path=parent.heading_path,
            content=parent.content,
            token_count=block_tokens,
            best_child_rank=parent.best_child_rank,
            matched_children=parent.matched_children,
        )
        context_blocks.append(block_obj)

        # Build detailed provenance mapping table
        provenance_map[source_label] = {
            "source_index": i,
            "document_id": str(parent.document_id),
            "filename": parent.filename,
            "parent_chunk_id": str(parent.parent_chunk_id),
            "parent_index": parent.parent_index,
            "heading_path": parent.heading_path,
            "best_child_rank": parent.best_child_rank,
            "matched_children": [
                {
                    "chunk_id": str(c.chunk_id),
                    "rank": c.rank,
                    "score": c.score,
                    "score_type": c.score_type,
                }
                for c in parent.matched_children
            ],
        }

    assembled_text = block_delimiter.join(formatted_text_blocks)
    calculated_total = (
        total_tokens if total_tokens is not None else sum(b.token_count for b in context_blocks)
    )

    return AssembledContextPayload(
        total_tokens=calculated_total,
        max_tokens_budget=max_tokens_budget,
        selected_parent_count=len(context_blocks),
        omitted_parent_count=len(omitted_parents),
        assembled_context_text=assembled_text,
        context_blocks=context_blocks,
        provenance_map=provenance_map,
    )


class ContextFormatter:
    """
    Stateful service coordinating context assembly and formatting.
    """

    def __init__(self, block_delimiter: str = "\n\n---\n\n") -> None:
        self.block_delimiter = block_delimiter

    def format(
        self,
        selected_parents: list[ExpandedParentContext],
        omitted_parents: list[ExpandedParentContext],
        max_tokens_budget: int,
        total_tokens: int | None = None,
        token_counts: list[int] | None = None,
    ) -> AssembledContextPayload:
        """Assembles formatted context blocks and provenance lookup table."""
        return assemble_context_payload(
            selected_parents=selected_parents,
            omitted_parents=omitted_parents,
            max_tokens_budget=max_tokens_budget,
            total_tokens=total_tokens,
            token_counts=token_counts,
            block_delimiter=self.block_delimiter,
        )
