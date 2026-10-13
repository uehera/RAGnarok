"""
Parent Context Expansion Subsystem.

Resolves full parent context sections from PostgreSQL for surviving top-K child matches,
deduplicates co-parent child hits, preserves scoring provenance, and orders parents strictly
by best matching child rank without synthetic score calculations.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

import structlog

from ragnarok.context.contract import ExpandedParentContext, MatchedChildReference
from ragnarok.core.models import Document, ParentChunk
from ragnarok.retrieval.contract import ChildResultCandidate

logger = structlog.get_logger(__name__)


class ParentLookupProtocol(Protocol):
    """Structural protocol for batch-resolving parent chunks and documents from storage."""

    def get_parent_chunks_by_ids(self, parent_chunk_ids: list[UUID]) -> dict[UUID, ParentChunk]:
        """Retrieves parent chunks matching a list of parent chunk UUIDs."""
        ...

    def get_documents_by_ids(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        """Retrieves document records matching a list of document UUIDs."""
        ...


def expand_parent_contexts(
    candidates: list[ChildResultCandidate],
    parent_lookup: ParentLookupProtocol,
) -> list[ExpandedParentContext]:
    """
    Pure procedural function expanding child candidates into deduplicated parent contexts.

    Steps:
    1. Extract and deduplicate parent_chunk_id foreign keys from candidates.
    2. Group child references under each parent and determine best_child_rank.
    3. Batch-fetch parent records from PostgreSQL.
    4. Batch-fetch source document metadata (filenames).
    5. Order unique parents strictly by best_child_rank ascending (highest relevance first).
    """
    if not candidates:
        logger.debug("Empty candidate list received; returning empty parent expansion")
        return []

    # Step 1 & 2: Group children by parent_chunk_id
    parent_to_children: dict[UUID, list[ChildResultCandidate]] = {}
    for candidate in candidates:
        if candidate.parent_chunk_id is None:
            logger.warning(
                "Child candidate has no parent_chunk_id; skipping expansion",
                chunk_id=str(candidate.chunk_id),
            )
            continue
        parent_to_children.setdefault(candidate.parent_chunk_id, []).append(candidate)

    if not parent_to_children:
        logger.warning("No valid parent_chunk_id references found in candidates")
        return []

    unique_parent_ids = list(parent_to_children.keys())

    # Step 3: Batch query PostgreSQL parent_chunks
    parent_records = parent_lookup.get_parent_chunks_by_ids(unique_parent_ids)

    # Step 4: Batch query PostgreSQL documents for filenames
    doc_ids = {record.document_id for record in parent_records.values()}
    doc_records = parent_lookup.get_documents_by_ids(list(doc_ids)) if doc_ids else {}

    expanded_list: list[ExpandedParentContext] = []

    for parent_id, children in parent_to_children.items():
        parent_chunk = parent_records.get(parent_id)
        if parent_chunk is None:
            logger.error(
                "Data integrity alert: parent_chunk_id not found in database",
                parent_chunk_id=str(parent_id),
                referencing_child_count=len(children),
            )
            continue

        doc = doc_records.get(parent_chunk.document_id)
        filename = doc.filename if doc else f"document_{parent_chunk.document_id}"

        # Determine best child rank (lowest integer value = highest relevance)
        best_child_rank = min(c.final_rank for c in children)

        # Assemble matched children references preserving ranks and native scores
        matched_refs = [
            MatchedChildReference(
                chunk_id=c.chunk_id,
                rank=c.final_rank,
                score=c.final_score,
                score_type=c.score_type
                or ("rerank_score" if c.rerank_score is not None else "fusion_score"),
            )
            for c in children
        ]
        # Sort child references by rank ascending
        matched_refs.sort(key=lambda m: m.rank)

        expanded_parent = ExpandedParentContext(
            parent_chunk_id=parent_chunk.parent_chunk_id,
            document_id=parent_chunk.document_id,
            filename=filename,
            parent_index=parent_chunk.parent_index,
            heading_path=parent_chunk.heading_path,
            content=parent_chunk.content,
            best_child_rank=best_child_rank,
            matched_children=matched_refs,
        )
        expanded_list.append(expanded_parent)

    # Step 5: Order unique parents strictly by best_child_rank ascending
    expanded_list.sort(key=lambda p: (p.best_child_rank, p.parent_index))

    logger.info(
        "Expanded parent contexts",
        input_candidates=len(candidates),
        unique_parents=len(expanded_list),
    )
    return expanded_list


class ParentContextExpander:
    """
    Stateful service coordinating parent context resolution with explicit dependency injection.
    """

    def __init__(self, parent_lookup: ParentLookupProtocol) -> None:
        self.parent_lookup = parent_lookup

    def expand(self, candidates: list[ChildResultCandidate]) -> list[ExpandedParentContext]:
        """Expands child chunk retrieval candidates into deduplicated, ordered parent contexts."""
        return expand_parent_contexts(candidates, self.parent_lookup)
