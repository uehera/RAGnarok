"""
Answer Generation and Citations Data Contracts.

Defines strongly-typed Pydantic schemas for generation responses,
footnote citations, latency diagnostics, and token metrics.
Adheres strictly to dev/workspace/generation_citations.md and dev/arc/generation.md.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CitationFootnote(BaseModel):
    """
    Structured metadata for an individual cited footnote.
    Maps an inline [X] marker back to its source document and parent section.
    """

    model_config = ConfigDict(frozen=True)

    citation_index: int = Field(ge=1, description="Inline marker index (1, 2, 3...) matching [X]")
    document_id: UUID | None = Field(
        default=None, description="Source document identifier in PostgreSQL"
    )
    filename: str = Field(description="Original source document filename")
    parent_chunk_id: UUID | None = Field(
        default=None, description="Source parent chunk identifier in PostgreSQL"
    )
    heading_path: list[str] = Field(
        default_factory=list, description="Structural heading breadcrumb hierarchy"
    )

    def format_footnote_text(self) -> str:
        """Formats the citation entry into standardized CLI text."""
        section_str = " > ".join(self.heading_path) if self.heading_path else "Root Section"
        parent_id_str = str(self.parent_chunk_id) if self.parent_chunk_id else "N/A"
        return (
            f"[{self.citation_index}] {self.filename}\n"
            f"    Section: {section_str}\n"
            f"    Parent Chunk: {parent_id_str}"
        )


class LatencyMetrics(BaseModel):
    """Execution latency metrics for the generation request."""

    model_config = ConfigDict(frozen=True)

    ttft_ms: float = Field(ge=0.0, description="Time-to-first-token in milliseconds")
    total_duration_ms: float = Field(
        ge=0.0, description="Total generation duration in milliseconds"
    )


class TokenUsage(BaseModel):
    """Token consumption metrics for prompt and generation completion."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class GenerationResponse(BaseModel):
    """
    Standardized final response payload produced by the Answer Generation stage.
    Carries the synthesized, sanitized text, cited footnotes, and diagnostics.
    """

    model_config = ConfigDict(frozen=False)

    query: str = Field(description="User input query")
    answer: str = Field(description="Synthesized, finalized, and guardrail-screened answer text")
    model_used: str = Field(description="Active model identifier that generated the response")
    fallback_triggered: bool = Field(
        default=False, description="True if a fallback model was invoked"
    )
    citations: list[CitationFootnote] = Field(
        default_factory=list, description="Structured metadata for each valid cited footnote"
    )
    ghost_citations: list[int] = Field(
        default_factory=list,
        description="Extracted citation indices not found in context provenance",
    )
    latency_metrics: LatencyMetrics = Field(description="Latency metrics including TTFT")
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Token usage statistics"
    )

    def to_provenance_dict(self) -> dict[str, Any]:
        """Serializes generation metadata for diagnostics, logging, and evaluation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "model_used": self.model_used,
            "fallback_triggered": self.fallback_triggered,
            "citations_count": len(self.citations),
            "citations": [c.model_dump() for c in self.citations],
            "ghost_citations": self.ghost_citations,
            "ttft_ms": self.latency_metrics.ttft_ms,
            "total_duration_ms": self.latency_metrics.total_duration_ms,
        }
