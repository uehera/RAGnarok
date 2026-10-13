"""
Dense Search Execution Subsystem.

Generates query embeddings via sentence-transformers, searches the TurboVec IdMapIndex
using cosine similarity, applies optional metadata allowlist filters, resolves 64-bit
numeric IDs back to PostgreSQL child chunk UUIDs, and packages results into the Child
Result Contract schema.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

import structlog

from ragnarok.core.config import DenseRetrievalConfig
from ragnarok.core.models import ChildChunk
from ragnarok.indexing.dense import DenseIndex
from ragnarok.retrieval.contract import ChildResultCandidate

logger = structlog.get_logger(__name__)


class ChunkLookupProtocol(Protocol):
    """Protocol for fetching child chunk metadata records by UUIDs."""

    def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> dict[UUID, ChildChunk]:
        """Retrieves child chunks matching a list of chunk UUIDs."""
        ...


def execute_dense_search(
    index: DenseIndex,
    query: str,
    top_k: int = 50,
    allowlist: list[UUID] | None = None,
    repository: ChunkLookupProtocol | None = None,
) -> list[ChildResultCandidate]:
    """
    Pure procedural function executing dense vector similarity retrieval against TurboVec.

    Parameters:
        index: Initialized or loaded DenseIndex instance.
        query: Raw search query text.
        top_k: Maximum number of dense candidates to retrieve.
        allowlist: Optional list of allowed child chunk UUIDs from metadata filtering.
        repository: Optional relational repository to enrich parent_chunk_id and content.

    Returns:
        list[ChildResultCandidate]: Ranked child chunk candidates ordered by descending similarity.
    """
    cleaned_query = query.strip()
    if not cleaned_query or index.is_empty:
        return []

    # If an explicit allowlist was supplied but is empty (0 matching metadata records),
    # return empty candidate set immediately without vector search.
    if allowlist is not None and len(allowlist) == 0:
        logger.info("Dense search skipped: empty metadata allowlist")
        return []

    # 1. Search TurboVec dense index (resolves uint64 handles to UUIDs)
    raw_results = index.search(
        query=cleaned_query,
        top_k=top_k,
        allowlist=allowlist,
    )
    if not raw_results:
        return []

    chunk_ids = [uid for uid, _ in raw_results]

    # 2. Enrich with relational metadata if repository is provided
    chunks_map: dict[UUID, ChildChunk] = {}
    if repository is not None:
        try:
            chunks_map = repository.get_chunks_by_ids(chunk_ids)
        except Exception as exc:
            logger.warning(
                "Failed to enrich dense chunks from repository",
                error=str(exc),
            )

    # 3. Construct ChildResultCandidate objects with 1-based ranks
    candidates: list[ChildResultCandidate] = []
    for rank_idx, (chunk_id, score) in enumerate(raw_results, start=1):
        chunk_entity = chunks_map.get(chunk_id)
        candidate = ChildResultCandidate(
            chunk_id=chunk_id,
            parent_chunk_id=chunk_entity.parent_chunk_id if chunk_entity else None,
            document_id=chunk_entity.document_id if chunk_entity else None,
            content=chunk_entity.content if chunk_entity else None,
            dense_rank=rank_idx,
            dense_score=score,
            score_type="similarity_score",
            score_direction="higher_is_better",
        )
        candidates.append(candidate)

    logger.info(
        "Dense vector search executed successfully",
        query=cleaned_query,
        candidates_found=len(candidates),
        top_score=candidates[0].dense_score if candidates else None,
    )
    return candidates


class DenseSearcher:
    """
    Stateful service executing dense vector search queries.
    Encapsulates DenseIndex, optional relational repository, and configuration.
    """

    def __init__(
        self,
        index: DenseIndex,
        repository: ChunkLookupProtocol | None = None,
        config: DenseRetrievalConfig | None = None,
    ) -> None:
        self.index = index
        self.repository = repository
        self.config = config if config is not None else DenseRetrievalConfig()

    def search(
        self,
        query: str,
        top_k: int = 50,
        allowlist: list[UUID] | None = None,
    ) -> list[ChildResultCandidate]:
        """
        Executes dense vector search and formats candidates into the Child Result Contract.
        """
        return execute_dense_search(
            index=self.index,
            query=query,
            top_k=top_k,
            allowlist=allowlist,
            repository=self.repository,
        )
