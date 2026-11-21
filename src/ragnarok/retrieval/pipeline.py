"""
Query Retrieval & Context Assembly Pipeline Coordinator.

Integrates:
1. Lexical (BM25s), Dense (TurboVec), or Hybrid (RRF) Retrieval
2. Cross-Encoder Reranking
3. Parent Context Expansion and Rank-Ordered Token Budgeting
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import structlog

from ragnarok.context.contract import AssembledContextPayload
from ragnarok.context.service import ContextManagementService
from ragnarok.core.config import AppConfig
from ragnarok.indexing.dense import DenseIndex
from ragnarok.indexing.lexical import LexicalIndex
from ragnarok.retrieval.contract import ChildResultCandidate
from ragnarok.retrieval.dense_search import DenseSearcher
from ragnarok.retrieval.fusion import HybridFusionService
from ragnarok.retrieval.lexical_search import LexicalSearcher
from ragnarok.retrieval.reranker import (
    SentenceTransformerCrossEncoder,
    rerank_candidates,
)
from ragnarok.storage.database import DatabasePool
from ragnarok.storage.repository import RelationalRepository

logger = structlog.get_logger(__name__)


def extract_document_filter(query: str, repo: RelationalRepository | None) -> str | None:
    """
    Extracts a target document filter if the query explicitly references
    a document filename or stem present in the repository.
    """
    if not repo:
        return None
    try:
        all_docs = repo.list_all_documents() if hasattr(repo, "list_all_documents") else []
        query_lower = query.lower()
        for doc in all_docs:
            fn = doc.filename.lower()
            if fn in query_lower:
                return doc.filename
            stem = fn.rsplit(".", 1)[0]
            if len(stem) > 4 and stem in query_lower:
                return doc.filename
    except Exception as exc:
        logger.debug("Failed to extract doc_filter from query", error=str(exc))
    return None


def retrieve_and_assemble_context(
    query: str,
    config: AppConfig,
    strategy: str = "hybrid",
    rerank: bool = True,
    budget: int = 0,
    doc_filter: str | None = None,
) -> tuple[AssembledContextPayload, list[ChildResultCandidate], dict[str, float]]:
    """
    Executes end-to-end retrieval, reranking, parent expansion, and budgeting:
    1. Loads BM25s and TurboVec indexes from data/indexes/.
    2. Runs Lexical, Dense, or Hybrid (RRF) search.
    3. Re-scores candidates using Cross-Encoder (if enabled).
    4. Expands parent chunks from DuckDB and applies token budgeting.

    Parameters:
        query: User input retrieval query.
        config: Application configuration.
        strategy: Retrieval strategy ("hybrid", "dense", "lexical").
        rerank: Whether to execute Cross-Encoder reranking.
        budget: Token budget override (0 uses config default).
        doc_filter: Optional filename filter to restrict retrieval to a specific document.

    Returns:
        tuple[AssembledContextPayload, list[ChildResultCandidate], dict[str, float]]:
            (assembled_payload, ranked_candidates, stage_latencies)
    """
    timings: dict[str, float] = {
        "retrieval_ms": 0.0,
        "rerank_ms": 0.0,
        "context_ms": 0.0,
    }

    repo: RelationalRepository | None = None
    try:
        db_url = os.getenv("DATABASE_URL", config.secrets.database_url)
        pool = DatabasePool(db_url)
        repo = RelationalRepository(pool)
    except Exception as exc:
        logger.warning("Database unavailable for retrieval pipeline", error=str(exc))
        return (
            AssembledContextPayload(
                total_tokens=0,
                max_tokens_budget=budget if budget > 0 else config.context_management.max_tokens,
                selected_parent_count=0,
                omitted_parent_count=0,
                assembled_context_text="",
                provenance_map={},
            ),
            [],
            timings,
        )

    # 1. Retrieval Stage
    start_retrieval = time.perf_counter()

    if doc_filter is None and repo:
        doc_filter = extract_document_filter(query, repo)
        if doc_filter:
            logger.info("Auto-detected target document filter from query", doc_filter=doc_filter)

    bm25_path = Path("data/indexes/bm25")
    dense_path = Path("data/indexes/dense")

    candidates: list[ChildResultCandidate] = []
    top_n = config.reranking.top_n if rerank else config.reranking.top_k
    search_k = max(top_n * 3, 40) if doc_filter else top_n

    has_bm25 = bm25_path.exists() and any(f.name != ".gitkeep" for f in bm25_path.iterdir())
    has_dense = dense_path.exists() and any(f.name != ".gitkeep" for f in dense_path.iterdir())

    lex_searcher: LexicalSearcher | None = None
    if has_bm25:
        try:
            lex_idx = LexicalIndex.load(bm25_path, config=config.lexical_retrieval)
            lex_searcher = LexicalSearcher(
                index=lex_idx, repository=repo, config=config.lexical_retrieval
            )
        except Exception as exc:
            logger.warning("Failed to load BM25 index for retrieval", error=str(exc))

    dense_searcher: DenseSearcher | None = None
    if has_dense:
        try:
            dense_idx = DenseIndex.load(dense_path, config=config.dense_retrieval)
            dense_searcher = DenseSearcher(
                index=dense_idx, repository=repo, config=config.dense_retrieval
            )
        except Exception as exc:
            logger.warning("Failed to load dense index for retrieval", error=str(exc))

    if strategy == "hybrid" and lex_searcher and dense_searcher:
        fusion_svc = HybridFusionService(
            lexical_searcher=lex_searcher,
            dense_searcher=dense_searcher,
            config=config.hybrid_retrieval,
        )
        candidates = fusion_svc.search(query=query, top_k=search_k)
    elif strategy == "dense" and dense_searcher:
        candidates = dense_searcher.search(query=query, top_k=search_k)
    elif (strategy in ("bm25", "lexical") and lex_searcher) or lex_searcher:
        candidates = lex_searcher.search(query=query, top_k=search_k)
    elif dense_searcher:
        candidates = dense_searcher.search(query=query, top_k=search_k)

    timings["retrieval_ms"] = (time.perf_counter() - start_retrieval) * 1000.0

    # Filter candidates by document scope if active
    if doc_filter and candidates and repo:
        doc_pattern = doc_filter.strip().lower()
        all_docs = repo.list_all_documents() if hasattr(repo, "list_all_documents") else []
        matching_doc_ids = {
            d.document_id for d in all_docs if doc_pattern in d.filename.lower() or d.filename.lower() in doc_pattern
        }
        if matching_doc_ids:
            candidates = [c for c in candidates if c.document_id in matching_doc_ids]
        else:
            # Fallback exact lookup
            exact_doc = repo.get_document_by_filename(doc_filter)
            if exact_doc:
                candidates = [c for c in candidates if c.document_id == exact_doc.document_id]
            else:
                candidates = []
        candidates = candidates[:top_n]

    # 2. Cross-Encoder Reranking Stage
    start_rerank = time.perf_counter()
    if rerank and candidates and config.reranking.enabled:
        try:
            reranker_model = SentenceTransformerCrossEncoder(model_name=config.reranking.model)
            candidates = rerank_candidates(
                query=query,
                candidates=candidates,
                model=reranker_model,
                top_n=config.reranking.top_n,
                top_k=config.reranking.top_k,
                repository=repo,
            )
        except Exception as exc:
            logger.warning("Reranking failed; proceeding with retrieval order", error=str(exc))
    else:
        candidates = candidates[: config.reranking.top_k]

    timings["rerank_ms"] = (time.perf_counter() - start_rerank) * 1000.0

    # 3. Parent Context Expansion & Budgeting Stage
    start_ctx = time.perf_counter()
    ctx_config = config.context_management.model_copy()
    if budget > 0:
        ctx_config.max_tokens = budget

    ctx_service = ContextManagementService.from_config(
        parent_lookup=repo,
        config=ctx_config,
        guardrails_config=config.guardrails,
    )
    payload = ctx_service.build_context(candidates, target_doc=doc_filter)
    timings["context_ms"] = (time.perf_counter() - start_ctx) * 1000.0

    return payload, candidates, timings
