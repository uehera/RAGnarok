"""
Lexical Search Execution Subsystem.

Executes query tokenization and inverted index retrieval using BM25s.
Resolves internal document positions back to child chunk UUIDs, enriches records
with relational metadata (if repository is injected), and serializes candidates into
the Child Result Contract schema.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

import structlog

from ragnarok.core.config import LexicalRetrievalConfig
from ragnarok.core.models import ChildChunk
from ragnarok.indexing.lexical import LexicalIndex
from ragnarok.retrieval.contract import ChildResultCandidate

logger = structlog.get_logger(__name__)


class ChunkLookupProtocol(Protocol):
    """Protocol for fetching child chunk metadata records by UUIDs."""

    def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> dict[UUID, ChildChunk]:
        """Retrieves child chunks matching a list of chunk UUIDs."""
        ...


def execute_lexical_search(
    index: LexicalIndex,
    query: str,
    top_k: int = 50,
    repository: ChunkLookupProtocol | None = None,
) -> list[ChildResultCandidate]:
    """
    Pure procedural function executing lexical retrieval against a BM25s index.

    Parameters:
        index: Initialized or loaded LexicalIndex instance.
        query: Raw search query text.
        top_k: Maximum number of lexical candidates to retrieve.
        repository: Optional relational repository to enrich parent_chunk_id and content.

    Returns:
        list[ChildResultCandidate]: Ranked child chunk candidates ordered by descending BM25 score.
    """
    cleaned_query = query.strip()
    if not cleaned_query or index.is_empty:
        return []

    # 1. Search BM25s index (resolves positions to UUIDs)
    raw_results = index.search(query=cleaned_query, top_k=top_k)
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
                "Failed to enrich lexical chunks from repository",
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
            bm25_rank=rank_idx,
            bm25_score=score,
            score_type="bm25_score",
            score_direction="higher_is_better",
        )
        candidates.append(candidate)

    logger.info(
        "Lexical search executed successfully",
        query=cleaned_query,
        candidates_found=len(candidates),
        top_score=candidates[0].bm25_score if candidates else None,
    )
    return candidates


class LexicalSearcher:
    """
    Stateful service executing lexical search queries.
    Encapsulates LexicalIndex, optional relational repository, and configuration.
    """

    def __init__(
        self,
        index: LexicalIndex,
        repository: ChunkLookupProtocol | None = None,
        config: LexicalRetrievalConfig | None = None,
    ) -> None:
        self.index = index
        self.repository = repository
        self.config = config if config is not None else LexicalRetrievalConfig()

    def search(self, query: str, top_k: int = 50) -> list[ChildResultCandidate]:
        """
        Executes lexical search and formats candidates into the Child Result Contract.
        """
        return execute_lexical_search(
            index=self.index,
            query=query,
            top_k=top_k,
            repository=self.repository,
        )
