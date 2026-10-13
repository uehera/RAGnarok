"""
Cross-Encoder Reranking Subsystem.

Re-scores top-N candidate child chunks from RRF fusion against the user query using
a sentence-transformers Cross-Encoder model. Slices final candidates down to top-K.
Implements automatic fail-safe fallback: if the reranker fails or times out,
gracefully returns candidates in original RRF order without crashing.
"""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

import structlog

from ragnarok.core.config import RerankerConfig
from ragnarok.core.models import ChildChunk
from ragnarok.retrieval.contract import ChildResultCandidate

logger = structlog.get_logger(__name__)


class ChunkLookupProtocol(Protocol):
    """Protocol for fetching child chunk metadata records by UUIDs."""

    def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> dict[UUID, ChildChunk]:
        """Retrieves child chunks matching a list of chunk UUIDs."""
        ...


class CrossEncoderProtocol(Protocol):
    """Protocol contract for Cross-Encoder reranking models."""

    def predict(self, sentences: list[list[str]] | list[tuple[str, str]]) -> Any:
        """Computes relevance scores for (query, document) text pairs."""
        ...


class SentenceTransformerCrossEncoder:
    """
    Default Cross-Encoder model wrapper using sentence-transformers.
    Lazy-loads model to prevent startup latency.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def predict(self, sentences: list[list[str]] | list[tuple[str, str]]) -> list[float]:
        if not sentences:
            return []
        model = self._get_model()
        scores = model.predict(sentences)
        return [float(s) for s in scores]


def rerank_candidates(
    query: str,
    candidates: list[ChildResultCandidate],
    model: CrossEncoderProtocol,
    top_n: int = 50,
    top_k: int = 10,
    repository: ChunkLookupProtocol | None = None,
) -> list[ChildResultCandidate]:
    """
    Pure procedural function executing Cross-Encoder reranking over top-N candidates.

    Parameters:
        query: User search query string.
        candidates: RRF-fused or pre-ranked candidates.
        model: Initialized Cross-Encoder instance implementing CrossEncoderProtocol.
        top_n: Number of candidates from the input list to score.
        top_k: Final number of top candidates to retain.
        repository: Optional chunk repository to fetch missing chunk content.

    Returns:
        list[ChildResultCandidate]: Re-scored candidates sliced to top-K, ordered descending by rerank_score.
                                   Falls back to original input order on exception.
    """
    if not candidates:
        return []

    # Select top-N candidates for reranking
    candidates_pool = [c.model_copy() for c in candidates[:top_n]]
    final_k = min(top_k, len(candidates_pool))

    # Enrich missing content from repository if available
    missing_ids = [c.chunk_id for c in candidates_pool if not c.content]
    if missing_ids and repository is not None:
        try:
            chunks_map = repository.get_chunks_by_ids(missing_ids)
            for cand in candidates_pool:
                if cand.chunk_id in chunks_map:
                    chunk = chunks_map[cand.chunk_id]
                    cand.content = chunk.content
                    if not cand.parent_chunk_id:
                        cand.parent_chunk_id = chunk.parent_chunk_id
                    if not cand.document_id:
                        cand.document_id = chunk.document_id
        except Exception as exc:
            logger.warning(
                "Failed to fetch missing chunk contents from repository for reranking",
                error=str(exc),
            )

    # Build (query, document) pairs for cross-encoder scoring
    pairs: list[tuple[str, str]] = []
    for cand in candidates_pool:
        doc_text = cand.content if cand.content is not None else ""
        pairs.append((query, doc_text))

    try:
        raw_scores = model.predict(pairs)
        scores_list = [float(s) for s in raw_scores]

        # Attach rerank score to candidates
        for cand, score in zip(candidates_pool, scores_list, strict=True):
            cand.rerank_score = score

        # Sort descending by rerank score
        sorted_candidates = sorted(
            candidates_pool,
            key=lambda c: c.rerank_score if c.rerank_score is not None else float("-inf"),
            reverse=True,
        )

        # Slice to final top-K and assign 1-based rerank_rank
        surviving = sorted_candidates[:final_k]
        for rank_idx, cand in enumerate(surviving, start=1):
            cand.rerank_rank = rank_idx
            cand.score_type = "rerank_score"

        logger.info(
            "Cross-Encoder reranking succeeded",
            top_n_scored=len(candidates_pool),
            top_k_retained=len(surviving),
            best_rerank_score=surviving[0].rerank_score if surviving else None,
        )
        return surviving

    except Exception as exc:
        logger.warning(
            "Cross-Encoder reranking encountered an error; falling back to RRF candidate order",
            error=str(exc),
            candidates_count=len(candidates_pool),
        )
        # Automatic graceful fallback: return top-K in original candidate order without crashing
        fallback_candidates = candidates[:final_k]
        return fallback_candidates


class CrossEncoderReranker:
    """
    Stateful service executing Cross-Encoder reranking.
    Encapsulates RerankerConfig, CrossEncoder model, and optional relational repository.
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
        model: CrossEncoderProtocol | None = None,
        repository: ChunkLookupProtocol | None = None,
    ) -> None:
        self.config = config if config is not None else RerankerConfig()
        self.repository = repository
        self.model = (
            model
            if model is not None
            else SentenceTransformerCrossEncoder(model_name=self.config.model)
        )

    def rerank(
        self,
        query: str,
        candidates: list[ChildResultCandidate],
        top_n: int | None = None,
        top_k: int | None = None,
    ) -> list[ChildResultCandidate]:
        """
        Reranks candidate child chunks if reranking is enabled.
        If disabled, returns top-K candidates directly in their original order.
        """
        target_top_k = top_k if top_k is not None else self.config.top_k
        target_top_n = top_n if top_n is not None else self.config.top_n

        # If reranker is disabled by configuration, return top-K in original RRF order
        if not self.config.enabled:
            logger.info("Reranker disabled in configuration; passing candidates in RRF order")
            return candidates[:target_top_k]

        return rerank_candidates(
            query=query,
            candidates=candidates,
            model=self.model,
            top_n=target_top_n,
            top_k=target_top_k,
            repository=self.repository,
        )
