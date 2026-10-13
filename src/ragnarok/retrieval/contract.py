"""
Retrieval Result Contract Subsystem.

Defines the normalized Child Result Contract schema, retaining raw retrieval scores,
ranks across pipeline stages (BM25, dense, RRF fusion, reranking), and provenance metadata.
Enforces score direction locking ('higher_is_better') without raw score conflation.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChildResultCandidate(BaseModel):
    """
    Normalized child chunk retrieval candidate schema.
    Carries core chunk identifiers, native scores, fusion scores, and rerank metrics.
    """

    model_config = ConfigDict(frozen=False)

    chunk_id: UUID = Field(description="Unique child chunk identifier")
    parent_chunk_id: UUID | None = Field(
        default=None, description="Identifier of the parent chunk containing this child"
    )
    document_id: UUID | None = Field(default=None, description="Identifier of the source document")
    content: str | None = Field(
        default=None, description="Text content of the child chunk for reranking and inspection"
    )

    # Retrieval stage ranks (1-based) and native scores
    bm25_rank: int | None = Field(
        default=None, ge=1, description="Rank position in BM25s output, if present"
    )
    bm25_score: float | None = Field(default=None, description="Native BM25 score, if present")
    dense_rank: int | None = Field(
        default=None, ge=1, description="Rank position in TurboVec output, if present"
    )
    dense_score: float | None = Field(
        default=None, description="Native cosine similarity score, if present"
    )

    # Fusion stage metrics
    fusion_rank: int | None = Field(
        default=None, ge=1, description="Rank position in RRF output, if hybrid search active"
    )
    fusion_score: float | None = Field(
        default=None, description="Reciprocal Rank Fusion score, if hybrid search active"
    )

    # Reranking stage metrics
    rerank_rank: int | None = Field(
        default=None, ge=1, description="Rank position after Cross-Encoder reranking, if enabled"
    )
    rerank_score: float | None = Field(
        default=None, description="Cross-Encoder relevance score, if enabled"
    )

    # Metadata & Directionality
    score_type: str | None = Field(
        default=None,
        description="Type identifier of primary score (e.g., bm25_score, similarity_score)",
    )
    score_direction: str = Field(
        default="higher_is_better", description="Scoring direction convention"
    )

    @property
    def final_rank(self) -> int:
        """Returns the final active rank according to pipeline progression."""
        if self.rerank_rank is not None:
            return self.rerank_rank
        if self.fusion_rank is not None:
            return self.fusion_rank
        if self.bm25_rank is not None:
            return self.bm25_rank
        if self.dense_rank is not None:
            return self.dense_rank
        return 1

    @property
    def final_score(self) -> float | None:
        """Returns the final active score according to pipeline progression."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fusion_score is not None:
            return self.fusion_score
        if self.dense_score is not None:
            return self.dense_score
        if self.bm25_score is not None:
            return self.bm25_score
        return None

    def to_provenance_dict(self) -> dict[str, Any]:
        """Serializes candidate into a structured dictionary for logging and provenance."""
        return {
            "chunk_id": str(self.chunk_id),
            "parent_chunk_id": str(self.parent_chunk_id) if self.parent_chunk_id else None,
            "document_id": str(self.document_id) if self.document_id else None,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "fusion_rank": self.fusion_rank,
            "fusion_score": self.fusion_score,
            "rerank_rank": self.rerank_rank,
            "rerank_score": self.rerank_score,
            "final_rank": self.final_rank,
            "final_score": self.final_score,
            "score_direction": self.score_direction,
        }
