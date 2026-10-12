"""
Unit and integration tests for Testing Point 3: Parent-Child Chunking & Token Sizing.

Verifies:
- Structural heading boundary splitting.
- Parent chunks adhering to parent_target_size bound.
- Hierarchical breadcrumb preservation in heading_path.
- Sequential zero-based parent_index values.
- Child chunks adhering to child_target_size and child_overlap bounds.
- Sequential zero-based chunk_index values across the document.
- Parent-child foreign key linkage (parent_chunk_id).
- Token counter backend verification and consistency between AutoTokenizer and Tiktoken fallback.
"""

from __future__ import annotations

from uuid import uuid4

from ragnarok.core.config import ChunkingConfig
from ragnarok.ingestion.chunker import (
    AutoTokenizerCounter,
    TiktokenCounter,
    TwoStageChunker,
    resolve_token_counter,
)


def test_token_counter_backend_verification() -> None:
    """
    Verifies counting behavior across AutoTokenizer and TiktokenCounter.
    Asserts non-zero counts and counting consistency on representative text.
    """
    sample_text = (
        "RAGnarok combines high-performance dense vector search with lexical BM25 ranking "
        "and strict parent-child chunking."
    )

    tiktoken_counter = TiktokenCounter(encoding_name="cl100k_base")
    t_count = tiktoken_counter.count_tokens(sample_text)
    assert t_count > 0

    # AutoTokenizer fallback resolution
    auto_counter = AutoTokenizerCounter(
        model_name="BAAI/bge-small-en-v1.5",
        fallback_counter=tiktoken_counter,
    )
    a_count = auto_counter.count_tokens(sample_text)
    assert a_count > 0

    # Resolve token counter helper
    resolved = resolve_token_counter(size_counter="tiktoken", fallback_counter="tiktoken")
    assert resolved.count_tokens(sample_text) == t_count


def test_two_stage_chunking_hierarchy_and_indices() -> None:
    """
    Verifies that Level 1 splits strictly along heading boundaries,
    preserves hierarchical heading paths, and assigns zero-based sequential indices.
    """
    markdown_sample = """# Introduction
This is the root introduction block.

## Architecture
The system architecture consists of storage, retrieval, and synthesis.

### Relational Storage
PostgreSQL stores metadata, parent chunks, and child chunks.

### Dense Indexing
TurboVec handles dense embeddings and allowlist filters.

# Operations
Operational guidelines for running RAGnarok in production.
"""
    config = ChunkingConfig(
        parent_target_size=1000,
        child_target_size=150,
        child_overlap=20,
        heading_levels=["#", "##", "###"],
    )
    chunker = TwoStageChunker(config=config, token_counter=TiktokenCounter())
    doc_id = uuid4()

    parents, children = chunker.chunk_markdown(markdown_sample, document_id=doc_id)

    # 1. Verify parent chunks
    assert len(parents) == 5
    for i, p in enumerate(parents):
        assert p.document_id == doc_id
        assert p.parent_index == i

    # Verify heading paths
    assert parents[0].heading_path == ["Introduction"]
    assert "Introduction" in parents[0].content

    assert parents[1].heading_path == ["Introduction", "Architecture"]
    assert "Architecture" in parents[1].content

    assert parents[2].heading_path == ["Introduction", "Architecture", "Relational Storage"]
    assert "Relational Storage" in parents[2].content

    assert parents[3].heading_path == ["Introduction", "Architecture", "Dense Indexing"]
    assert "Dense Indexing" in parents[3].content

    assert parents[4].heading_path == ["Operations"]
    assert "Operations" in parents[4].content

    # 2. Verify child chunks
    assert len(children) >= 5
    parent_ids = {p.parent_chunk_id for p in parents}

    for j, c in enumerate(children):
        assert c.document_id == doc_id
        assert c.chunk_index == j
        # Every child carries a valid parent_chunk_id referencing one of the parents
        assert c.parent_chunk_id in parent_ids
        assert len(c.content.strip()) > 0


def test_child_token_sizing_and_overlap_boundaries() -> None:
    """
    Verifies that oversized text is subdivided into child chunks adhering
    strictly to child_target_size (e.g. 128-256 tokens) and child_overlap.
    """
    # Create long text under a single heading
    long_paragraph = (
        "PostgreSQL provides enterprise-grade ACID transactions, cascading deletes, "
        "and indexing support for JSONB, vectors, and relational metadata. " * 30
    )
    markdown_content = f"# Deep Dive\n\n{long_paragraph}"

    child_target_size = 150
    child_overlap = 30
    config = ChunkingConfig(
        parent_target_size=2000,
        child_target_size=child_target_size,
        child_overlap=child_overlap,
        heading_levels=["#"],
    )
    counter = TiktokenCounter()
    chunker = TwoStageChunker(config=config, token_counter=counter)
    doc_id = uuid4()

    parents, children = chunker.chunk_markdown(markdown_content, document_id=doc_id)

    assert len(parents) == 1
    assert len(children) > 1

    # Verify each child is bounded by child_target_size
    for child in children:
        tokens = counter.count_tokens(child.content)
        assert tokens <= child_target_size + 10  # small margin for sentence/word boundary

    # Verify sequential zero-based indices
    for idx, child in enumerate(children):
        assert child.chunk_index == idx

    # Verify adjacent children from subdivided text share overlapping content
    overlap_found = False
    for k in range(len(children) - 1):
        child_a = children[k].content
        child_b = children[k + 1].content

        # Extract words from child_a's tail and child_b's head to assert text repetition
        words_a = child_a.split()
        words_b = child_b.split()
        if len(set(words_a[-15:]).intersection(set(words_b[:15]))) > 0:
            overlap_found = True
            break
    assert overlap_found


def test_oversized_parent_splitting_preserves_heading_path() -> None:
    """
    Verifies that an oversized section (> parent_target_size) is subdivided into
    multiple bounded parent units, all retaining the same heading_path.
    """
    huge_paragraph = (
        "The quick brown fox jumps over the lazy dog and explores the dense forest. " * 100
    )
    markdown_content = f"# Mammoth Section\n\n{huge_paragraph}"

    counter = TiktokenCounter()
    parent_target_size = 200
    config = ChunkingConfig(
        parent_target_size=parent_target_size,
        child_target_size=80,
        child_overlap=10,
        heading_levels=["#"],
    )
    chunker = TwoStageChunker(config=config, token_counter=counter)
    doc_id = uuid4()

    parents, children = chunker.chunk_markdown(markdown_content, document_id=doc_id)

    # Must be split into multiple bounded parents
    assert len(parents) > 1

    # Each parent must be bounded and inherit the heading_path
    for p in parents:
        tokens = counter.count_tokens(p.content)
        assert tokens <= parent_target_size + 10
        assert p.heading_path == ["Mammoth Section"]

    # Parent indices must be strictly sequential zero-based
    for idx, p in enumerate(parents):
        assert p.parent_index == idx
