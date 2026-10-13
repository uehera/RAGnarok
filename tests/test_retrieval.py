"""
Unit and integration tests for Testing Point 7: Retrieval Execution, Hybrid Fusion & Reranking.

Verifies:
- Child Result Contract schema validation, score direction locking, and provenance serialization.
- BM25s lexical search returning valid child UUIDs with 1-based bm25_rank and native bm25_score.
- TurboVec dense vector search returning valid child UUIDs with allowlist filtering.
- Reciprocal Rank Fusion (RRF):
  - Calculates candidate order purely based on 1-based rank positions with constant k=60.
  - Zero contribution from retrievers where a chunk does not appear.
  - Preserves native BM25 scores and cosine similarity scores without raw score summation.
- Cross-Encoder Reranking:
  - Re-scores top-N candidates and sorts descending down to top-K.
  - Slices to top-K when enabled.
  - Gracefully falls back to RRF candidate order when reranking is disabled or on exception.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from ragnarok.core.config import HybridRetrievalConfig, RerankerConfig
from ragnarok.core.models import ChildChunk
from ragnarok.indexing.dense import DenseIndex, EmbedderProtocol, l2_normalize
from ragnarok.indexing.lexical import LexicalIndex
from ragnarok.retrieval.contract import ChildResultCandidate
from ragnarok.retrieval.dense_search import DenseSearcher, execute_dense_search
from ragnarok.retrieval.fusion import HybridFusionService, reciprocal_rank_fusion
from ragnarok.retrieval.lexical_search import LexicalSearcher, execute_lexical_search
from ragnarok.retrieval.reranker import (
    CrossEncoderProtocol,
    CrossEncoderReranker,
    rerank_candidates,
)


class MockDeterministicEmbedder(EmbedderProtocol):
    """Deterministic offline embedder for reproducible testing without network access."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = []
        for text in texts:
            rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
            raw_vec = rng.standard_normal(self.dimension).astype(np.float32)
            vectors.append(raw_vec)

        arr = np.stack(vectors, axis=0)
        return l2_normalize(arr).astype(np.float32)


class MockCrossEncoder(CrossEncoderProtocol):
    """Deterministic mock Cross-Encoder returning pre-programmed relevance scores."""

    def __init__(self, score_mapping: dict[str, float] | None = None) -> None:
        self.score_mapping = score_mapping or {}
        self.call_count = 0

    def predict(self, sentences: list[list[str]] | list[tuple[str, str]]) -> list[float]:
        self.call_count += 1
        scores: list[float] = []
        for pair in sentences:
            doc_text = pair[1]
            # Match score by substring or default to length-based score
            matched = False
            for key, score in self.score_mapping.items():
                if key in doc_text:
                    scores.append(score)
                    matched = True
                    break
            if not matched:
                scores.append(0.5)
        return scores


class FailingCrossEncoder(CrossEncoderProtocol):
    """Mock Cross-Encoder that simulates an unexpected runtime exception or model timeout."""

    def predict(self, sentences: list[list[str]] | list[tuple[str, str]]) -> Any:
        raise TimeoutError("Simulated Cross-Encoder inference timeout")


class InMemoryChunkRepository:
    """In-memory repository implementing ChunkLookupProtocol for testing."""

    def __init__(self, chunks: list[ChildChunk]) -> None:
        self.chunks_by_id = {c.chunk_id: c for c in chunks}

    def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> dict[UUID, ChildChunk]:
        return {uid: self.chunks_by_id[uid] for uid in chunk_ids if uid in self.chunks_by_id}


# ============================================================================
# 1. Result Contract Tests
# ============================================================================


def test_child_result_candidate_schema_and_ranks() -> None:
    """Verifies ChildResultCandidate field defaults, score direction, and rank resolution."""
    c_id = uuid4()
    p_id = uuid4()
    d_id = uuid4()

    # BM25 only candidate
    cand_bm25 = ChildResultCandidate(
        chunk_id=c_id,
        parent_chunk_id=p_id,
        document_id=d_id,
        content="Sample chunk content",
        bm25_rank=2,
        bm25_score=14.5,
        score_type="bm25_score",
    )
    assert cand_bm25.score_direction == "higher_is_better"
    assert cand_bm25.final_rank == 2
    assert cand_bm25.final_score == 14.5

    # Dense only candidate
    cand_dense = ChildResultCandidate(
        chunk_id=c_id,
        dense_rank=1,
        dense_score=0.89,
        score_type="similarity_score",
    )
    assert cand_dense.final_rank == 1
    assert cand_dense.final_score == 0.89

    # Fused candidate
    cand_fused = ChildResultCandidate(
        chunk_id=c_id,
        bm25_rank=3,
        bm25_score=11.2,
        dense_rank=2,
        dense_score=0.82,
        fusion_rank=1,
        fusion_score=0.032,
        score_type="rrf_score",
    )
    assert cand_fused.final_rank == 1
    assert cand_fused.final_score == 0.032

    # Reranked candidate (rerank overrides fusion)
    cand_reranked = ChildResultCandidate(
        chunk_id=c_id,
        bm25_rank=1,
        dense_rank=2,
        fusion_rank=1,
        fusion_score=0.032,
        rerank_rank=1,
        rerank_score=0.98,
        score_type="rerank_score",
    )
    assert cand_reranked.final_rank == 1
    assert cand_reranked.final_score == 0.98

    # Provenance dictionary contains all fields
    prov = cand_reranked.to_provenance_dict()
    assert prov["chunk_id"] == str(c_id)
    assert prov["bm25_rank"] == 1
    assert prov["dense_rank"] == 2
    assert prov["fusion_rank"] == 1
    assert prov["rerank_rank"] == 1
    assert prov["rerank_score"] == 0.98


# ============================================================================
# 2. Lexical Search Execution Tests
# ============================================================================


def test_lexical_search_execution() -> None:
    """Verifies BM25s lexical search returns valid candidates with ranks, scores, and enrichment."""
    doc_id = uuid4()
    p_id = uuid4()
    chunks = [
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="PostgreSQL metadata store handles parent-child relational cascading deletes.",
            chunk_index=0,
            heading_path=["Storage", "PostgreSQL"],
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="TurboVec provides high performance dense vector similarity search with cosine distance.",
            chunk_index=1,
            heading_path=["Retrieval", "TurboVec"],
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="BM25s indexing library produces lexical inverted indexes using english stemmer.",
            chunk_index=2,
            heading_path=["Retrieval", "BM25"],
        ),
    ]

    lexical_idx = LexicalIndex()
    lexical_idx.build(chunks)
    repo = InMemoryChunkRepository(chunks)

    searcher = LexicalSearcher(index=lexical_idx, repository=repo)

    # Search for PostgreSQL terms
    results = searcher.search(query="PostgreSQL cascading deletes", top_k=2)
    assert len(results) > 0
    top = results[0]

    assert top.chunk_id == chunks[0].chunk_id
    assert top.bm25_rank == 1
    assert top.bm25_score is not None
    assert top.bm25_score > 0
    assert top.score_type == "bm25_score"
    assert top.parent_chunk_id == p_id
    assert top.document_id == doc_id
    assert "PostgreSQL" in (top.content or "")

    # Direct procedural function call verification
    proc_results = execute_lexical_search(
        lexical_idx, "PostgreSQL cascading deletes", top_k=2, repository=repo
    )
    assert len(proc_results) > 0
    assert proc_results[0].chunk_id == chunks[0].chunk_id

    # Empty query returns empty list
    assert searcher.search(query="", top_k=10) == []
    assert execute_lexical_search(lexical_idx, query="") == []


# ============================================================================
# 3. Dense Search Execution Tests
# ============================================================================


def test_dense_search_execution_and_allowlist() -> None:
    """Verifies TurboVec dense search returns candidates and enforces metadata allowlists."""
    doc_id = uuid4()
    p_id = uuid4()
    chunks = [
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="The quick brown fox jumps over the lazy dog.",
            chunk_index=0,
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="Artificial intelligence and dense embeddings represent semantic meaning.",
            chunk_index=1,
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="Relational transactions guarantee ACID properties for storage.",
            chunk_index=2,
        ),
    ]

    embedder = MockDeterministicEmbedder(dimension=384)
    dense_idx = DenseIndex(embedder=embedder)
    dense_idx.build(chunks)
    repo = InMemoryChunkRepository(chunks)

    searcher = DenseSearcher(index=dense_idx, repository=repo)

    # 1. Free search across all chunks
    results = searcher.search(query="semantic vector embeddings", top_k=3)
    assert len(results) == 3
    assert all(r.dense_rank in [1, 2, 3] for r in results)
    assert all(r.dense_score is not None for r in results)
    assert all(r.score_type == "similarity_score" for r in results)

    # 2. Search with allowlist containing only chunk 0 and chunk 2
    allowlist = [chunks[0].chunk_id, chunks[2].chunk_id]
    filtered_results = searcher.search(
        query="semantic vector embeddings", top_k=3, allowlist=allowlist
    )
    assert len(filtered_results) == 2
    returned_ids = {r.chunk_id for r in filtered_results}
    assert returned_ids == set(allowlist)
    assert chunks[1].chunk_id not in returned_ids

    # 3. Search with empty allowlist -> immediate empty return without search
    empty_allowlist_results = searcher.search(query="any query", top_k=3, allowlist=[])
    assert empty_allowlist_results == []

    # Direct procedural function call verification
    proc_dense = execute_dense_search(
        dense_idx, "semantic vector", top_k=2, allowlist=allowlist, repository=repo
    )
    assert len(proc_dense) == 2
    assert {c.chunk_id for c in proc_dense} == set(allowlist)

    # 4. Empty query returns empty list
    assert searcher.search(query="", top_k=3) == []
    assert execute_dense_search(dense_idx, query="", top_k=3) == []


# ============================================================================
# 4. Reciprocal Rank Fusion (RRF) Tests
# ============================================================================


def test_reciprocal_rank_fusion_mathematical_precision() -> None:
    """
    Verifies that RRF calculates candidate order purely based on 1-based rank positions
    with formula sum(1.0 / (k + rank)), preserves native scores without raw summation,
    and assigns 0 contribution when absent from a retriever.
    """
    id_a = uuid4()
    id_b = uuid4()
    id_c = uuid4()
    id_d = uuid4()

    # List 1 (Lexical): [A (rank 1), B (rank 2), C (rank 3)]
    lexical_candidates = [
        ChildResultCandidate(chunk_id=id_a, bm25_rank=1, bm25_score=25.0),
        ChildResultCandidate(chunk_id=id_b, bm25_rank=2, bm25_score=18.0),
        ChildResultCandidate(chunk_id=id_c, bm25_rank=3, bm25_score=12.0),
    ]

    # List 2 (Dense): [B (rank 1), A (rank 2), D (rank 3)]
    # Note: raw dense scores are cosine similarities in [-1, 1], completely different scale from BM25 (12-25)
    dense_candidates = [
        ChildResultCandidate(chunk_id=id_b, dense_rank=1, dense_score=0.95),
        ChildResultCandidate(chunk_id=id_a, dense_rank=2, dense_score=0.91),
        ChildResultCandidate(chunk_id=id_d, dense_rank=3, dense_score=0.80),
    ]

    k = 60
    fused = reciprocal_rank_fusion(
        lexical_candidates=lexical_candidates,
        dense_candidates=dense_candidates,
        k=k,
    )

    # Mathematical expected scores:
    # A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    expected_score_a = (1.0 / 61) + (1.0 / 62)
    # B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    expected_score_b = (1.0 / 62) + (1.0 / 61)
    # C: 1/(60+3) + 0.0 = 1/63 (appears only in lexical)
    expected_score_c = 1.0 / 63
    # D: 0.0 + 1/(60+3) = 1/63 (appears only in dense)
    expected_score_d = 1.0 / 63

    fused_by_id = {c.chunk_id: c for c in fused}

    # 1. Assert exact mathematical RRF scores
    assert np.isclose(fused_by_id[id_a].fusion_score or 0.0, expected_score_a)
    assert np.isclose(fused_by_id[id_b].fusion_score or 0.0, expected_score_b)
    assert np.isclose(fused_by_id[id_c].fusion_score or 0.0, expected_score_c)
    assert np.isclose(fused_by_id[id_d].fusion_score or 0.0, expected_score_d)

    # 2. Both top chunks A and B have higher RRF scores than C and D
    score_a = fused_by_id[id_a].fusion_score
    score_b = fused_by_id[id_b].fusion_score
    score_c = fused_by_id[id_c].fusion_score
    score_d = fused_by_id[id_d].fusion_score
    assert (
        score_a is not None and score_b is not None and score_c is not None and score_d is not None
    )
    assert score_a > score_c
    assert score_b > score_d

    # 3. Assert raw scores are NOT summed together
    # If raw scores were summed, A would have 25.0 + 0.91 = 25.91, but fusion_score is ~0.0325
    assert score_a < 1.0
    # Native scores must be preserved intact on the candidate
    assert fused_by_id[id_a].bm25_score == 25.0
    assert fused_by_id[id_a].dense_score == 0.91

    # 4. Zero contribution: C has dense_score=None and dense_rank=None
    assert fused_by_id[id_c].dense_score is None
    assert fused_by_id[id_c].dense_rank is None
    assert fused_by_id[id_c].bm25_score == 12.0

    # 5. Zero contribution: D has bm25_score=None and bm25_rank=None
    assert fused_by_id[id_d].bm25_score is None
    assert fused_by_id[id_d].bm25_rank is None
    assert fused_by_id[id_d].dense_score == 0.80

    # 6. Final fusion ranks are sequential 1-based integers
    assert [c.fusion_rank for c in fused] == [1, 2, 3, 4]

    # 7. Invalid k <= 0 raises ValueError
    with pytest.raises(ValueError, match="RRF constant k must be positive"):
        reciprocal_rank_fusion(lexical_candidates, dense_candidates, k=0)


def test_hybrid_fusion_service_concurrent_execution() -> None:
    """Verifies HybridFusionService concurrently executes lexical and dense searches and fuses results."""
    doc_id = uuid4()
    p_id = uuid4()
    chunks = [
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="Alpha text",
            chunk_index=0,
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content="Beta text",
            chunk_index=1,
        ),
    ]

    embedder = MockDeterministicEmbedder(dimension=384)
    dense_idx = DenseIndex(embedder=embedder)
    dense_idx.build(chunks)

    lexical_idx = LexicalIndex()
    lexical_idx.build(chunks)

    lex_searcher = LexicalSearcher(index=lexical_idx)
    dense_searcher = DenseSearcher(index=dense_idx)

    service = HybridFusionService(
        lexical_searcher=lex_searcher,
        dense_searcher=dense_searcher,
        config=HybridRetrievalConfig(k=60),
    )

    results = service.search(query="Alpha text", top_k=2)
    assert len(results) == 2
    assert results[0].fusion_rank == 1
    assert results[1].fusion_rank == 2
    assert results[0].fusion_score is not None


# ============================================================================
# 5. Cross-Encoder Reranker Tests
# ============================================================================


def test_cross_encoder_reranking_reorders_top_n_to_top_k() -> None:
    """
    Verifies that Cross-Encoder scores (query, candidate.content) pairs,
    successfully re-orders candidates, and slices to top_k.
    """
    id_1 = uuid4()
    id_2 = uuid4()
    id_3 = uuid4()

    # Initial RRF candidates: Rank 1 is id_1, Rank 2 is id_2, Rank 3 is id_3
    candidates = [
        ChildResultCandidate(
            chunk_id=id_1, content="Weak relevance content", fusion_rank=1, fusion_score=0.035
        ),
        ChildResultCandidate(
            chunk_id=id_2, content="Moderate relevance content", fusion_rank=2, fusion_score=0.032
        ),
        ChildResultCandidate(
            chunk_id=id_3,
            content="Highly relevant exact target match",
            fusion_rank=3,
            fusion_score=0.030,
        ),
    ]

    # Cross-encoder gives highest score to id_3 ("Highly relevant") and lowest to id_1
    mock_model = MockCrossEncoder(
        score_mapping={
            "Highly relevant": 0.98,
            "Moderate": 0.65,
            "Weak": 0.12,
        }
    )

    reranker = CrossEncoderReranker(
        config=RerankerConfig(enabled=True, top_n=3, top_k=2),
        model=mock_model,
    )

    # Execute rerank for top_k=2
    reranked = reranker.rerank(query="find target match", candidates=candidates, top_n=3, top_k=2)

    assert len(reranked) == 2
    # Chunk 3 was promoted from rank 3 to rank 1
    assert reranked[0].chunk_id == id_3
    assert reranked[0].rerank_rank == 1
    assert reranked[0].rerank_score == 0.98
    assert reranked[0].score_type == "rerank_score"

    # Chunk 2 is rank 2
    assert reranked[1].chunk_id == id_2
    assert reranked[1].rerank_rank == 2
    assert reranked[1].rerank_score == 0.65

    # Original fusion_score and fusion_rank are preserved for provenance
    assert reranked[0].fusion_rank == 3
    assert reranked[0].fusion_score == 0.030

    # Direct procedural function call verification
    proc_reranked = rerank_candidates(
        query="find target match", candidates=candidates, model=mock_model, top_n=3, top_k=2
    )
    assert len(proc_reranked) == 2
    assert proc_reranked[0].chunk_id == id_3
    assert proc_reranked[0].rerank_rank == 1


def test_cross_encoder_disabled_passes_rrf_order() -> None:
    """Verifies that when reranker is disabled, top-K candidates are returned in original RRF order."""
    candidates = [
        ChildResultCandidate(
            chunk_id=uuid4(), content=f"Chunk {i}", fusion_rank=i, fusion_score=1.0 / (60 + i)
        )
        for i in range(1, 6)
    ]

    mock_model = MockCrossEncoder()
    reranker = CrossEncoderReranker(
        config=RerankerConfig(enabled=False, top_n=5, top_k=3),
        model=mock_model,
    )

    result = reranker.rerank(query="test", candidates=candidates, top_k=3)
    assert len(result) == 3
    # Model was never called
    assert mock_model.call_count == 0
    # Original order preserved
    assert [c.chunk_id for c in result] == [c.chunk_id for c in candidates[:3]]


def test_cross_encoder_failure_gracefully_falls_back_to_rrf_order() -> None:
    """
    Verifies that if the Cross-Encoder model fails (throws exception/timeout),
    the system logs a warning and gracefully falls back to the original RRF order without crashing.
    """
    candidates = [
        ChildResultCandidate(
            chunk_id=uuid4(), content="Chunk A", fusion_rank=1, fusion_score=0.035
        ),
        ChildResultCandidate(
            chunk_id=uuid4(), content="Chunk B", fusion_rank=2, fusion_score=0.032
        ),
        ChildResultCandidate(
            chunk_id=uuid4(), content="Chunk C", fusion_rank=3, fusion_score=0.028
        ),
    ]

    failing_model = FailingCrossEncoder()
    reranker = CrossEncoderReranker(
        config=RerankerConfig(enabled=True, top_n=3, top_k=2),
        model=failing_model,
    )

    # Must NOT raise TimeoutError; must return top-2 in RRF order
    fallback_result = reranker.rerank(query="test query", candidates=candidates, top_n=3, top_k=2)

    assert len(fallback_result) == 2
    assert fallback_result[0].chunk_id == candidates[0].chunk_id
    assert fallback_result[1].chunk_id == candidates[1].chunk_id
    assert fallback_result[0].fusion_rank == 1
    assert fallback_result[1].fusion_rank == 2
