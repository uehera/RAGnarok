"""
Unit and integration tests for Testing Point 6: Pre-Retrieval Optimization & Dynamic Routing.

Verifies:
- Self-RAG Decision Gate separating general knowledge from corpus-specific technical queries.
- Query Rewriter producing a single clarified query while preserving exact technical symbols and identifiers.
- Adaptive Retrieval Router performing deterministic signal analysis to select Lexical, Dense, or Hybrid strategies.
- Metadata Filtering Engine extracting explicit constraints and generating valid TurboVec integer allowlists from PostgreSQL.
- Immediate empty candidate sets when zero chunks match metadata constraints.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from ragnarok.core.config import (
    AdaptiveRetrievalConfig,
    QueryRewritingConfig,
    SelfRagConfig,
)
from ragnarok.core.models import ChildChunk, Document, ParentChunk
from ragnarok.indexing.dense import VectorIdMap
from ragnarok.retrieval.filter import (
    MetadataFilterEngine,
    extract_constraints,
)
from ragnarok.retrieval.rewriter import QueryRewriter
from ragnarok.retrieval.router import (
    AdaptiveRouter,
    RetrievalStrategy,
)
from ragnarok.retrieval.self_rag import (
    LLMClientProtocol,
    SelfRagDecision,
    SelfRagGate,
)
from ragnarok.storage.repository import RelationalRepository


class MockLLMClient(LLMClientProtocol):
    """Deterministic mock LLM client for testing without network requests."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.call_history: list[dict[str, Any]] = []

    def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        self.call_history.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
        )
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")

        for pattern, reply in self.responses.items():
            if pattern in user_content:
                return reply

        # Default fallback response
        return '{"decision": "RETRIEVE", "reasoning": "Default response"}'


# ============================================================================
# 1. Self-RAG Decision Gate Tests
# ============================================================================


def test_self_rag_gate_general_vs_corpus_queries() -> None:
    """
    Verifies that Self-RAG gate routes general knowledge queries to DIRECT_GENERATE
    and corpus-specific technical questions to RETRIEVE.
    """
    mock_responses = {
        "what is the capital of France": '{"decision": "DIRECT_GENERATE", "reasoning": "General world knowledge geography"}',
        "how does TurboVec IdMapIndex handle allowlists": '{"decision": "RETRIEVE", "reasoning": "Corpus-specific vector indexing question"}',
        "hello there": '{"decision": "DIRECT_GENERATE", "reasoning": "Greeting"}',
        "explain the PostgreSQL schema for parent_chunks": '{"decision": "RETRIEVE", "reasoning": "Documentation question"}',
    }
    client = MockLLMClient(mock_responses)
    gate = SelfRagGate(config=SelfRagConfig(enabled=True), llm_client=client)

    # General knowledge queries
    res_gen1 = gate.evaluate("what is the capital of France?")
    assert res_gen1.decision == SelfRagDecision.DIRECT_GENERATE

    res_gen2 = gate.evaluate("hello there")
    assert res_gen2.decision == SelfRagDecision.DIRECT_GENERATE

    # Corpus-specific queries
    res_corp1 = gate.evaluate("how does TurboVec IdMapIndex handle allowlists?")
    assert res_corp1.decision == SelfRagDecision.RETRIEVE

    res_corp2 = gate.evaluate("explain the PostgreSQL schema for parent_chunks")
    assert res_corp2.decision == SelfRagDecision.RETRIEVE


def test_self_rag_gate_disabled_and_error_handling() -> None:
    """
    Verifies that when Self-RAG is disabled, all queries default to RETRIEVE,
    and exceptions fail-safe to RETRIEVE without crashing.
    """
    # 1. Disabled configuration
    disabled_gate = SelfRagGate(config=SelfRagConfig(enabled=False))
    res_disabled = disabled_gate.evaluate("what is 2 + 2?")
    assert res_disabled.decision == SelfRagDecision.RETRIEVE

    # 2. Error handling fallback
    class CrashingClient(LLMClientProtocol):
        def completion(self, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("Simulated LLM network failure")

    crashing_gate = SelfRagGate(
        config=SelfRagConfig(enabled=True),
        llm_client=CrashingClient(),
    )
    res_crash = crashing_gate.evaluate("test query")
    assert res_crash.decision == SelfRagDecision.RETRIEVE
    assert "Fail-safe" in res_crash.reasoning


# ============================================================================
# 2. Query Rewriter Subsystem Tests
# ============================================================================


def test_query_rewriter_preserves_identifiers_and_single_query() -> None:
    """
    Verifies that QueryRewriter expands ambiguous pronouns into a single query
    while strictly preserving technical identifiers and error codes.
    """
    mock_responses = {
        "How does it index them?": "How does BM25s index child chunk tokens in data/indexes/bm25/?",
        "Why did it fail with 404?": "Why did DoclingParser fail with HTTP 404 in ingestion_pipeline.py?",
        # Malformed response trying to output multi-query list
        "Multi-query attempt": "1. First query for BM25s\n2. Second query for TurboVec\n- Third query",
    }
    client = MockLLMClient(mock_responses)
    rewriter = QueryRewriter(
        config=QueryRewritingConfig(enabled=True),
        llm_client=client,
    )

    # 1. Pronoun expansion preserving path and BM25s symbol
    res1 = rewriter.rewrite("How does it index them?")
    assert res1 == "How does BM25s index child chunk tokens in data/indexes/bm25/?"
    assert "BM25s" in res1
    assert "data/indexes/bm25/" in res1

    # 2. Error code and class name preserved
    res2 = rewriter.rewrite("Why did it fail with 404?")
    assert res2 == "Why did DoclingParser fail with HTTP 404 in ingestion_pipeline.py?"
    assert "DoclingParser" in res2
    assert "404" in res2

    # 3. Multi-query output collapsed to single query
    res3 = rewriter.rewrite("Multi-query attempt")
    assert res3 == "First query for BM25s"
    assert "\n" not in res3


def test_query_rewriter_disabled_and_fallback() -> None:
    """
    Verifies that QueryRewriter passes original queries unchanged when disabled
    or on exception.
    """
    # 1. Disabled
    disabled_rewriter = QueryRewriter(config=QueryRewritingConfig(enabled=False))
    orig = "What is parent_chunks table?"
    assert disabled_rewriter.rewrite(orig) == orig

    # 2. Crashing client fallback
    class CrashingClient(LLMClientProtocol):
        def completion(self, *args: Any, **kwargs: Any) -> Any:
            raise TimeoutError("LLM timeout")

    error_rewriter = QueryRewriter(
        config=QueryRewritingConfig(enabled=True),
        llm_client=CrashingClient(),
    )
    assert error_rewriter.rewrite(orig) == orig


# ============================================================================
# 3. Adaptive Retrieval Router Tests
# ============================================================================


def test_adaptive_router_signal_classification() -> None:
    """
    Verifies that exact technical identifiers route to Lexical (BM25s),
    conceptual questions route to Dense (TurboVec),
    and mixed queries route to Hybrid.
    """
    router = AdaptiveRouter(config=AdaptiveRetrievalConfig(enabled=True))

    # Exact Lexical Signals
    lexical_queries = [
        "Find definition of RelationalRepository in repository.py",
        "Where is chunk_index >= 5 updated?",
        "Lookup error code ERR_DATABASE_DISCONNECTED",
        "Search for 'exact quoted token' in documents",
        "Inspect docling_parse.pyd C-extension",
    ]
    for q in lexical_queries:
        strategy = router.route(q)
        assert strategy == RetrievalStrategy.LEXICAL, f"Query '{q}' should route to LEXICAL"

    # Semantic / Conceptual Signals
    dense_queries = [
        "Why do we decouple child searching from parent context?",
        "Explain the high level architecture and design philosophy",
        "Provide an overview of the two-stage chunking tradeoffs",
        "Understand the relationship between documents and jobs",
    ]
    for q in dense_queries:
        strategy = router.route(q)
        assert strategy == RetrievalStrategy.DENSE, f"Query '{q}' should route to DENSE"

    # Mixed Signals -> Hybrid
    mixed_queries = [
        "Why does IdMapIndex use cosine similarity instead of dot product?",
        "Explain how the chunk_id UUID is mapped to a 64-bit uint64 integer",
        "Summarize the architecture of BM25s inverted index generation",
    ]
    for q in mixed_queries:
        strategy = router.route(q)
        assert strategy == RetrievalStrategy.HYBRID, f"Query '{q}' should route to HYBRID"

    # Neutral / ambiguous query -> Default configured strategy (HYBRID)
    neutral_query = "database information"
    assert router.route(neutral_query) == RetrievalStrategy.HYBRID


def test_adaptive_router_disabled_configuration() -> None:
    """
    Verifies that when adaptive retrieval is disabled, the router strictly
    returns the configured default_strategy regardless of query content.
    """
    router_dense_default = AdaptiveRouter(
        config=AdaptiveRetrievalConfig(enabled=False, default_strategy="dense")
    )
    # Query with strong exact-match identifier
    assert (
        router_dense_default.route("find definition of RelationalRepository")
        == RetrievalStrategy.DENSE
    )


# ============================================================================
# 4. Metadata Filtering Engine Tests
# ============================================================================


def test_metadata_constraint_extraction() -> None:
    """
    Verifies extraction of explicit metadata constraints (filename, document ID,
    chunk index, chunk range) from user queries.
    """
    # 1. Filename extraction
    c1 = extract_constraints("Search for transactions in file 'architecture.md'")
    assert c1.filename == "architecture.md"
    assert c1.has_constraints

    c2 = extract_constraints("file: setup.toml check installation")
    assert c2.filename == "setup.toml"

    # 2. Document ID extraction
    test_uuid = UUID("12345678-1234-5678-1234-567812345678")
    c3 = extract_constraints(f"query details for doc_id: {test_uuid}")
    assert c3.document_id == test_uuid

    # 3. Chunk range extraction
    c4 = extract_constraints("review chunks 0 to 4 in setup.md")
    assert c4.filename == "setup.md"
    assert c4.min_chunk_index == 0
    assert c4.max_chunk_index == 4

    # 4. Single chunk index extraction
    c5 = extract_constraints("inspect chunk 7 for syntax error")
    assert c5.chunk_index == 7

    # 5. Query without metadata constraints
    c_none = extract_constraints("What is the dual indexing pipeline?")
    assert not c_none.has_constraints


def test_metadata_filter_engine_postgresql_allowlist_generation(
    repo: RelationalRepository,
) -> None:
    """
    Verifies that MetadataFilterEngine queries PostgreSQL for matching chunk records,
    translates them into TurboVec 64-bit integer allowlists, and returns empty allowlist
    when zero records match.
    """
    doc_id = uuid4()
    doc = Document(
        document_id=doc_id,
        filename="filter_test_doc.md",
        raw_file_path=f"data/raw/{doc_id}.md",
        parsed_file_path=f"data/parsed/{doc_id}.md",
        file_type="text/markdown",
        file_size_bytes=1024,
        checksum=f"filter_chk_{uuid4().hex[:12]}",
        status="parsed",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="chunking")

    parent = ParentChunk(
        document_id=doc_id,
        content="Parent section for metadata filtering test.",
        parent_index=0,
        heading_path=["Filter Test"],
    )
    children = [
        ChildChunk(
            document_id=doc_id,
            parent_chunk_id=parent.parent_chunk_id,
            content=f"Child passage index {i} for allowlist test.",
            chunk_index=i,
            heading_path=["Filter Test"],
        )
        for i in range(5)
    ]
    repo.persist_chunks_atomically(
        document_id=doc_id,
        parent_chunks=[parent],
        child_chunks=children,
        job_id=job.job_id,
    )

    # Initialize VectorIdMap with bijection for the 5 child chunks
    id_map = VectorIdMap()
    for c in children:
        id_map.add(c.chunk_id)

    filter_engine = MetadataFilterEngine(repository=repo, id_map=id_map)

    # 1. Query with filename constraint
    q_file = "Search vector similarities in file: filter_test_doc.md"
    constraints, allowlist = filter_engine.filter_query(q_file)
    assert constraints.filename == "filter_test_doc.md"
    assert allowlist is not None
    assert len(allowlist) == 5
    for c in children:
        handle = id_map.get_handle(c.chunk_id)
        assert handle in allowlist

    # 2. Query with filename + chunk range constraint (chunks 1 to 3)
    q_range = "check chunks 1 to 3 in file 'filter_test_doc.md'"
    _, range_allowlist = filter_engine.filter_query(q_range)
    assert range_allowlist is not None
    assert len(range_allowlist) == 3
    expected_handles = {id_map.get_handle(children[i].chunk_id) for i in [1, 2, 3]}
    assert set(range_allowlist) == expected_handles

    # 3. Query with non-existent filename -> empty allowlist ([])
    q_missing = "search details in file non_existent_file.pdf"
    _, missing_allowlist = filter_engine.filter_query(q_missing)
    assert missing_allowlist == []

    # 4. Query with non-matching chunk index -> empty allowlist ([])
    q_out_of_bounds = "inspect chunk 99 in file filter_test_doc.md"
    _, oob_allowlist = filter_engine.filter_query(q_out_of_bounds)
    assert oob_allowlist == []

    # 5. Query with no constraints -> None (full vector search allowed)
    q_free = "How does vector quantizer operate?"
    _, free_allowlist = filter_engine.filter_query(q_free)
    assert free_allowlist is None
