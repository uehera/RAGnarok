"""
Parent Context Contract Subsystem.

Defines strongly-typed Pydantic schemas for expanded parent context blocks,
token-budgeted selections, and assembled prompt payloads with end-to-end provenance.
Adheres strictly to dev/arc/result_contract.md and dev/workspace/context_management.md.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MatchedChildReference(BaseModel):
    """
    Provenance reference for an individual child chunk that triggered parent expansion.
    Preserves child retrieval rank and native scoring metrics.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID = Field(description="Unique child chunk identifier")
    rank: int = Field(ge=1, description="1-based retrieval rank of the matched child")
    score: float | None = Field(default=None, description="Retrieval score of the child")
    score_type: str | None = Field(
        default=None,
        description="Type of retrieval score (e.g., bm25_score, similarity_score, fusion_score, rerank_score)",
    )


class ExpandedParentContext(BaseModel):
    """
    Expanded parent section retrieved from PostgreSQL for surviving top-K children.
    Associates the complete parent text with all matching children and inherits best-child rank.
    """

    model_config = ConfigDict(frozen=False)

    parent_chunk_id: UUID = Field(description="Unique parent container identifier")
    document_id: UUID = Field(description="Source document identifier")
    filename: str = Field(default="", description="Original source document filename")
    parent_index: int = Field(ge=0, description="Sequential index of the parent in the document")
    heading_path: list[str] = Field(
        default_factory=list, description="Structural heading hierarchy breadcrumbs"
    )
    content: str = Field(description="Complete untruncated parent section text")
    best_child_rank: int = Field(
        ge=1, description="Highest rank among all child chunks triggering this parent"
    )
    matched_children: list[MatchedChildReference] = Field(
        default_factory=list, description="All child chunks matching this parent"
    )


class ContextBlock(BaseModel):
    """
    An individual bounded source context block ready for prompt formatting.
    Assigned a 1-based sequential source identifier ([Source X]).
    """

    model_config = ConfigDict(frozen=False)

    source_index: int = Field(ge=1, description="1-based sequential source identifier")
    document_id: UUID = Field(description="Source document identifier")
    filename: str = Field(description="Source document filename")
    parent_chunk_id: UUID = Field(description="Unique parent identifier in PostgreSQL")
    parent_index: int = Field(
        ge=0, description="Sequence order of the parent in the source document"
    )
    heading_path: list[str] = Field(
        default_factory=list, description="Structural heading hierarchy"
    )
    content: str = Field(description="Complete, untruncated parent section text")
    token_count: int = Field(ge=0, description="Token count of this parent block")
    best_child_rank: int = Field(
        ge=1, description="Highest retrieval rank among triggering children"
    )
    matched_children: list[MatchedChildReference] = Field(
        default_factory=list, description="Matched child chunk records"
    )


class AssembledContextPayload(BaseModel):
    """
    Final context payload handed off to the LLM Answer Generation stage.
    Carries the assembled prompt string, token consumption metrics, and complete provenance map.
    """

    model_config = ConfigDict(frozen=False)

    total_tokens: int = Field(ge=0, description="Total tokens in assembled context text")
    max_tokens_budget: int = Field(ge=1, description="Configured token budget limit")
    selected_parent_count: int = Field(
        ge=0, description="Number of complete parent chunks included"
    )
    omitted_parent_count: int = Field(
        ge=0, description="Number of retrieved parents excluded due to budget"
    )
    assembled_context_text: str = Field(
        description="Delimited prompt text containing formatted sources"
    )
    context_blocks: list[ContextBlock] = Field(
        default_factory=list, description="Ordered list of included context blocks"
    )
    provenance_map: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Mapping of [Source X] labels to database and retrieval metadata",
    )
