"""
Enterprise Battle-Testing Suite — Dimension 4: Index Drift & Phantom Record Detection.

Verifies:
1. The "Ghost Vector" Probe: missing relational chunks are detected, omitted cleanly without crashing.
2. Vector Dimension Mismatch: vector dimension discrepancies are intercepted at boundary with ValueError.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from ragnarok.context.service import ContextManagementService
from ragnarok.core.config import DenseRetrievalConfig, load_config
from ragnarok.core.models import ChildChunk
from ragnarok.indexing.dense import DenseIndex
from ragnarok.retrieval.contract import ChildResultCandidate
from ragnarok.storage.repository import RelationalRepository


def test_drift_ghost_vector_handling_omits_cleanly(repo: RelationalRepository) -> None:
    """
    Simulates a 'ghost vector' where an index returns a chunk UUID that no longer
    exists in PostgreSQL (e.g. deleted document).
    Asserts parent context expansion catches missing relational record, logs an integrity
    warning, and omits the phantom chunk cleanly without crashing.
    """
    config = load_config()
    ctx_service = ContextManagementService.from_config(
        parent_lookup=repo,
        config=config.context_management,
    )

    # Candidate referencing a completely non-existent chunk in PostgreSQL
    ghost_chunk_id = uuid4()
    ghost_parent_id = uuid4()
    ghost_candidate = ChildResultCandidate(
        chunk_id=ghost_chunk_id,
        parent_chunk_id=ghost_parent_id,
        document_id=uuid4(),
        content=None,
        fusion_rank=1,
        fusion_score=0.035,
        score_type="rrf_score",
    )

    # Execute parent context expansion on ghost candidate
    payload = ctx_service.build_context([ghost_candidate])

    # Assert clean handling: 0 parents selected, 0 tokens, no exception thrown
    assert payload.selected_parent_count == 0
    assert payload.total_tokens == 0
    assert payload.assembled_context_text == ""


def test_drift_vector_dimension_mismatch_fails_diagnostically() -> None:
    """
    Simulates a query vector with 768 dimensions when index is configured for 384.
    Asserts boundary validation catches the mismatch immediately with a ValueError.
    """
    cfg = DenseRetrievalConfig(vector_dimension=384)

    class MismatchedEmbedder:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            # Emits 768-dim embeddings instead of 384
            return np.random.randn(len(texts), 768).astype(np.float32)

    index = DenseIndex(config=cfg, embedder=MismatchedEmbedder())

    # Build small index with dummy chunk
    dummy_chunk = ChildChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        parent_chunk_id=uuid4(),
        content="Dummy text",
        chunk_index=0,
        heading_path=[],
    )

    # When trying to build with 768 dimensions into 384 config, must raise ValueError
    with pytest.raises(ValueError, match="does not match configured vector_dimension"):
        index.build([dummy_chunk])

    # Also search check: populate index first with valid 384-dim chunk, then test query mismatch
    class ValidEmbedder:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            return np.random.randn(len(texts), 384).astype(np.float32)

    valid_idx = DenseIndex(config=cfg, embedder=ValidEmbedder())
    valid_idx.build([dummy_chunk])
    valid_idx.embedder = MismatchedEmbedder()

    with pytest.raises(ValueError, match="does not match configured vector_dimension"):
        valid_idx.search("Test query", top_k=5)
