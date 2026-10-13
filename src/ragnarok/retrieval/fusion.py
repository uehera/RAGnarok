"""
Reciprocal Rank Fusion (RRF) Subsystem.

Merges complementary ranked candidate lists from BM25s lexical search and TurboVec
dense vector search using Reciprocal Rank Fusion (RRF).
Calculates fusion scores strictly from rank positions:
    RRF(d) = sum(1.0 / (k + rank_r(d)))
Never normalizes or directly adds raw BM25 scores and cosine similarity scores.
Retains all native scores and ranks for provenance and downstream auditing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import structlog

from ragnarok.core.config import HybridRetrievalConfig
from ragnarok.retrieval.contract import ChildResultCandidate
from ragnarok.retrieval.dense_search import DenseSearcher
from ragnarok.retrieval.lexical_search import LexicalSearcher

logger = structlog.get_logger(__name__)


def reciprocal_rank_fusion(
    lexical_candidates: list[ChildResultCandidate],
    dense_candidates: list[ChildResultCandidate],
    k: int = 60,
) -> list[ChildResultCandidate]:
    """
    Pure procedural function performing Reciprocal Rank Fusion (RRF) across two ranked candidate lists.

    Parameters:
        lexical_candidates: Ordered candidates from lexical search (BM25s).
        dense_candidates: Ordered candidates from dense vector search (TurboVec).
        k: RRF rank smoothing constant (default: 60).

    Returns:
        list[ChildResultCandidate]: Unified candidates sorted descending by RRF fusion_score,
                                   with 1-based fusion_rank assigned.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be positive, got {k}")

    # Map chunk_id -> unified ChildResultCandidate
    merged: dict[UUID, ChildResultCandidate] = {}
    rrf_scores: dict[UUID, float] = {}

    # 1. Process Lexical candidates
    for rank_idx, candidate in enumerate(lexical_candidates, start=1):
        uid = candidate.chunk_id
        effective_bm25_rank = candidate.bm25_rank if candidate.bm25_rank is not None else rank_idx
        lex_score_contrib = 1.0 / (k + effective_bm25_rank)

        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + lex_score_contrib

        if uid not in merged:
            merged[uid] = ChildResultCandidate(
                chunk_id=uid,
                parent_chunk_id=candidate.parent_chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                bm25_rank=effective_bm25_rank,
                bm25_score=candidate.bm25_score,
                score_direction="higher_is_better",
            )
        else:
            merged[uid].bm25_rank = effective_bm25_rank
            merged[uid].bm25_score = candidate.bm25_score
            if not merged[uid].content and candidate.content:
                merged[uid].content = candidate.content
            if not merged[uid].parent_chunk_id and candidate.parent_chunk_id:
                merged[uid].parent_chunk_id = candidate.parent_chunk_id
            if not merged[uid].document_id and candidate.document_id:
                merged[uid].document_id = candidate.document_id

    # 2. Process Dense candidates
    for rank_idx, candidate in enumerate(dense_candidates, start=1):
        uid = candidate.chunk_id
        effective_dense_rank = (
            candidate.dense_rank if candidate.dense_rank is not None else rank_idx
        )
        dense_score_contrib = 1.0 / (k + effective_dense_rank)

        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + dense_score_contrib

        if uid not in merged:
            merged[uid] = ChildResultCandidate(
                chunk_id=uid,
                parent_chunk_id=candidate.parent_chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                dense_rank=effective_dense_rank,
                dense_score=candidate.dense_score,
                score_direction="higher_is_better",
            )
        else:
            merged[uid].dense_rank = effective_dense_rank
            merged[uid].dense_score = candidate.dense_score
            if not merged[uid].content and candidate.content:
                merged[uid].content = candidate.content
            if not merged[uid].parent_chunk_id and candidate.parent_chunk_id:
                merged[uid].parent_chunk_id = candidate.parent_chunk_id
            if not merged[uid].document_id and candidate.document_id:
                merged[uid].document_id = candidate.document_id

    # 3. Sort candidates by descending RRF score
    sorted_uids = sorted(
        merged.keys(),
        key=lambda u: rrf_scores[u],
        reverse=True,
    )

    # 4. Assign fusion_score, fusion_rank (1-based), and primary score_type
    fused_candidates: list[ChildResultCandidate] = []
    for rank_1_based, uid in enumerate(sorted_uids, start=1):
        cand = merged[uid]
        cand.fusion_score = rrf_scores[uid]
        cand.fusion_rank = rank_1_based
        cand.score_type = "rrf_score"
        fused_candidates.append(cand)

    logger.info(
        "Reciprocal Rank Fusion completed",
        lexical_input_count=len(lexical_candidates),
        dense_input_count=len(dense_candidates),
        fused_total_count=len(fused_candidates),
        k=k,
        top_fusion_score=fused_candidates[0].fusion_score if fused_candidates else None,
    )

    return fused_candidates


class HybridFusionService:
    """
    Stateful service coordinating parallel dual retrieval (Lexical + Dense)
    and Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        lexical_searcher: LexicalSearcher,
        dense_searcher: DenseSearcher,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self.lexical_searcher = lexical_searcher
        self.dense_searcher = dense_searcher
        self.config = config if config is not None else HybridRetrievalConfig()

    def search(
        self,
        query: str,
        top_k: int = 50,
        allowlist: list[UUID] | None = None,
    ) -> list[ChildResultCandidate]:
        """
        Executes lexical and dense searches concurrently and merges results with RRF.

        Parameters:
            query: Search query string.
            top_k: Maximum number of merged candidates to return.
            allowlist: Optional list of allowed child chunk UUIDs.

        Returns:
            list[ChildResultCandidate]: RRF-ranked candidate records.
        """
        # Execute concurrent searches across lexical and dense engines
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_lex = executor.submit(self.lexical_searcher.search, query, top_k)
            future_dense = executor.submit(self.dense_searcher.search, query, top_k, allowlist)
            lexical_results = future_lex.result()
            dense_results = future_dense.result()

        fused = reciprocal_rank_fusion(
            lexical_candidates=lexical_results,
            dense_candidates=dense_results,
            k=self.config.k,
        )

        return fused[:top_k]
