"""
Unit and integration tests for Testing Point 5: Dual Indexing Subsystem (BM25s & TurboVec).

Verifies:
- BM25s index creation, tokenization settings, serialization, and disk reload.
- Ordered integer-to-UUID 1:1 mapping consistency for lexical search.
- TurboVec dense vector index creation, dimension validation, serialization, and disk reload.
- 1:1 bidirectional injective mapping between PostgreSQL UUIDs and 64-bit TurboVec IDs.
- Cosine similarity search and metadata allowlist filtering.
- Standalone ReindexService regenerating both indexes directly from PostgreSQL chunks table.
- Job state transition to COMPLETED with valid completed_at timestamp.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from ragnarok.core.config import AppConfig, DenseRetrievalConfig, LexicalRetrievalConfig
from ragnarok.core.models import ChildChunk, Document, ParentChunk
from ragnarok.indexing.dense import DenseIndex, EmbedderProtocol, VectorIdMap, l2_normalize
from ragnarok.indexing.lexical import LexicalIndex, tokenize_corpus
from ragnarok.indexing.reindexer import ReindexService
from ragnarok.storage.repository import RelationalRepository


class DeterministicEmbedder(EmbedderProtocol):
    """
    Deterministic offline embedder producing unit-normalized vectors
    of fixed dimensionality for fast, reproducible tests without network access.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = []
        for text in texts:
            # Deterministic pseudo-embedding from text hash
            rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
            raw_vec = rng.standard_normal(self.dimension).astype(np.float32)
            vectors.append(raw_vec)

        arr = np.stack(vectors, axis=0)
        return l2_normalize(arr).astype(np.float32)


def test_bm25s_tokenization_settings() -> None:
    """
    Verifies that tokenize_corpus applies lowercase conversion,
    stop words removal, and English stemming.
    """
    sample = ["The quick brown foxes were jumping over lazier dogs."]
    config = LexicalRetrievalConfig(
        case="lowercase",
        stop_words_removed=True,
        stemming_enabled=True,
    )
    tokens = tokenize_corpus(sample, config=config)

    # Convert returned structure to list of token strings
    token_list = list(tokens[0])
    # Lowercase verified: no capital letters
    for t in token_list:
        assert t == t.lower()

    # Stopwords verified: "the", "were", "over" should be filtered out
    assert "the" not in token_list
    assert "were" not in token_list
    assert "over" not in token_list

    # Stemming verified: "foxes" stemmed to "fox", "jumping" to "jump"
    assert any("fox" in t for t in token_list)
    assert any("jump" in t for t in token_list)


def test_bm25s_index_creation_serialization_and_reload(tmp_path: Path) -> None:
    """
    Verifies that BM25s index is created, serialized to disk with ordered UUIDs,
    and reloaded producing identical search rankings and scores.
    """
    index_dir = tmp_path / "bm25_index"
    doc_id = uuid4()

    chunks = [
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=uuid4(),
            content="PostgreSQL provides ACID transactions and relational persistence.",
            chunk_index=0,
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=uuid4(),
            content="TurboVec provides high-speed vector indexing and cosine similarity search.",
            chunk_index=1,
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=uuid4(),
            content="Docling parses multi-column layout, headings, and complex table matrices.",
            chunk_index=2,
        ),
    ]

    # 1. Build and search in-memory index
    idx = LexicalIndex()
    idx.build(chunks)
    assert not idx.is_empty
    assert len(idx.chunk_ids) == 3

    results_orig = idx.search("relational transactions", top_k=2)
    assert len(results_orig) > 0
    # Top match must be chunk 0
    top_chunk_id, top_score = results_orig[0]
    assert top_chunk_id == chunks[0].chunk_id
    assert top_score > 0.0

    # 2. Serialize to disk
    idx.save(index_dir)
    assert (index_dir / "chunk_ids.json").exists()

    # 3. Reload from disk
    reloaded_idx = LexicalIndex.load(index_dir)
    assert not reloaded_idx.is_empty
    assert len(reloaded_idx.chunk_ids) == 3
    assert reloaded_idx.chunk_ids == idx.chunk_ids

    # 4. Assert identical search rankings and scores
    results_reloaded = reloaded_idx.search("relational transactions", top_k=2)
    assert len(results_reloaded) == len(results_orig)
    for (orig_id, orig_sc), (rel_id, rel_sc) in zip(results_orig, results_reloaded, strict=True):
        assert orig_id == rel_id
        assert pytest.approx(orig_sc, rel=1e-5) == rel_sc


def test_vector_id_map_bidirectional_bijection(tmp_path: Path) -> None:
    """
    Verifies that VectorIdMap maintains an injective, loss-free mapping
    between UUIDs and 64-bit unsigned integer IDs across save and load.
    """
    id_map = VectorIdMap()
    uuids = [uuid4() for _ in range(10)]

    for u in uuids:
        handle = id_map.register(u)
        assert handle > 0
        # Idempotent registration
        assert id_map.register(u) == handle
        # Bidirectional resolution
        assert id_map.get_uint64(u) == handle
        assert id_map.get_uuid(handle) == u

    map_path = tmp_path / "id_map.json"
    id_map.save(map_path)

    reloaded_map = VectorIdMap.load(map_path)
    assert len(reloaded_map) == 10
    for u in uuids:
        handle = id_map.get_uint64(u)
        assert handle is not None
        assert reloaded_map.get_uint64(u) == handle
        assert reloaded_map.get_uuid(handle) == u


def test_turbovec_dense_index_build_reload_and_allowlist(tmp_path: Path) -> None:
    """
    Verifies TurboVec dense index build, dimension validation, serialization,
    reload, and metadata allowlist filtering.
    """
    index_dir = tmp_path / "dense_index"
    embedder = DeterministicEmbedder(dimension=384)
    config = DenseRetrievalConfig(
        vector_dimension=384,
        batch_size=10,
        similarity_metric="cosine",
    )
    doc_id = uuid4()

    chunks = [
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=uuid4(),
            content="Vector similarity search using TurboVec",
            chunk_index=0,
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=uuid4(),
            content="PostgreSQL metadata store and ACID transactions.",
            chunk_index=1,
        ),
        ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=uuid4(),
            content="Document layout parsing with Docling engine.",
            chunk_index=2,
        ),
    ]

    # 1. Dimension validation mismatch error
    bad_config = DenseRetrievalConfig(vector_dimension=512)
    bad_idx = DenseIndex(config=bad_config, embedder=embedder)
    with pytest.raises(ValueError, match="does not match configured vector_dimension"):
        bad_idx.build(chunks)

    # 2. Build valid index
    dense_idx = DenseIndex(config=config, embedder=embedder)
    dense_idx.build(chunks)
    assert not dense_idx.is_empty
    assert len(dense_idx.index) == 3

    # Search without allowlist
    results_orig = dense_idx.search("Vector similarity search using TurboVec", top_k=2)
    assert len(results_orig) == 2
    top_chunk_id, top_score = results_orig[0]
    # In deterministic embedder, exact text match yields highest cosine score (~1.0)
    assert top_chunk_id == chunks[0].chunk_id
    assert top_score > 0.9

    # 3. Allowlist filtering
    # Restrict search only to chunk 1 and chunk 2
    allowlist = [chunks[1].chunk_id, chunks[2].chunk_id]
    filtered_results = dense_idx.search(
        "Vector similarity search using TurboVec",
        top_k=2,
        allowlist=allowlist,
    )
    assert len(filtered_results) > 0
    filtered_ids = {res[0] for res in filtered_results}
    assert chunks[0].chunk_id not in filtered_ids
    assert filtered_ids.issubset(set(allowlist))

    # Empty allowlist or no candidate matches must return [] immediately
    empty_results = dense_idx.search(
        "Vector similarity search",
        top_k=2,
        allowlist=[uuid4()],  # Non-existent UUID
    )
    assert empty_results == []

    # 4. Serialize to disk and reload
    dense_idx.save(index_dir)
    assert (index_dir / "index.tvim").exists()
    assert (index_dir / "id_map.json").exists()

    reloaded_dense = DenseIndex.load(index_dir, config=config, embedder=embedder)
    assert not reloaded_dense.is_empty
    assert len(reloaded_dense.index) == 3

    # Assert identical search results and similarity scores
    results_reloaded = reloaded_dense.search(
        "Vector similarity search using TurboVec",
        top_k=2,
    )
    assert len(results_reloaded) == len(results_orig)
    for (orig_id, orig_sc), (rel_id, rel_sc) in zip(results_orig, results_reloaded, strict=True):
        assert orig_id == rel_id
        assert pytest.approx(orig_sc, abs=1e-2) == rel_sc


def test_reindex_service_standalone_regeneration(
    repo: RelationalRepository,
    tmp_path: Path,
) -> None:
    """
    Verifies that ReindexService regenerates both BM25s and TurboVec indexes
    directly from PostgreSQL child chunks without touching raw files,
    and transitions processing_jobs.stage to COMPLETED with completed_at set.
    """
    bm25_dir = tmp_path / "indexes" / "bm25"
    dense_dir = tmp_path / "indexes" / "dense"

    doc_id = uuid4()
    doc = Document(
        document_id=doc_id,
        filename="reindex_doc.md",
        raw_file_path=f"data/raw/{doc_id}.md",
        parsed_file_path=f"data/parsed/{doc_id}.md",
        file_type="text/markdown",
        file_size_bytes=2048,
        checksum=f"reindex_chk_{uuid4().hex[:12]}",
        status="parsed",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="chunking")

    parent = ParentChunk(
        document_id=doc_id,
        content="Parent container for re-indexing verification.",
        parent_index=0,
        heading_path=["Overview"],
    )
    children = [
        ChildChunk(
            document_id=doc_id,
            parent_chunk_id=parent.parent_chunk_id,
            content="Child passage 1 explaining dual indexing architecture in RAGnarok.",
            chunk_index=0,
            heading_path=["Overview"],
        ),
        ChildChunk(
            document_id=doc_id,
            parent_chunk_id=parent.parent_chunk_id,
            content="Child passage 2 detailing lexical inverted index generation with BM25s.",
            chunk_index=1,
            heading_path=["Overview"],
        ),
    ]

    repo.persist_chunks_atomically(
        document_id=doc_id,
        parent_chunks=[parent],
        child_chunks=children,
        job_id=job.job_id,
    )

    # Initialize ReindexService with deterministic embedder
    embedder = DeterministicEmbedder(dimension=384)
    app_config = AppConfig()
    app_config.dense_retrieval.vector_dimension = 384

    reindexer = ReindexService(
        repository=repo,
        config=app_config,
        bm25_index_dir=bm25_dir,
        dense_index_dir=dense_dir,
        embedder=embedder,
    )

    # Execute standalone re-indexing
    indexed_count = reindexer.reindex_all(job_id=job.job_id)
    assert indexed_count >= 2

    # Assert files created on disk
    assert (bm25_dir / "chunk_ids.json").exists()
    assert (dense_dir / "index.tvim").exists()
    assert (dense_dir / "id_map.json").exists()

    # Assert job state machine finalized to COMPLETED
    updated_job = repo.get_job(job.job_id)
    assert updated_job is not None
    assert updated_job.stage == "completed"
    assert updated_job.completed_at is not None
    assert updated_job.error_message is None

    # Assert reloaded indexes execute search correctly
    loaded_lexical = LexicalIndex.load(bm25_dir)
    loaded_dense = DenseIndex.load(dense_dir, config=app_config.dense_retrieval, embedder=embedder)

    lex_res = loaded_lexical.search("BM25s lexical inverted index", top_k=1)
    assert len(lex_res) == 1
    assert lex_res[0][0] == children[1].chunk_id

    dense_res = loaded_dense.search(children[0].content, top_k=1)
    assert len(dense_res) == 1
    assert dense_res[0][0] == children[0].chunk_id
