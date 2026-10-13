"""
Comprehensive unit and integration tests for Phase 5: Parent Context Expansion & Context Management.
(Testing Point 8 in dev/workspace/tests_plan.md).

Verifies:
1. Deduplication of multiple child matches sharing the same parent chunk.
2. Strict ordering of parent blocks in ascending order of best_child_rank (highest relevance first).
3. Budget enforcement: Total context tokens never exceed context_management.max_tokens.
4. Whole-parent preservation: Zero slicing, partial truncation, or summarization of parents.
5. Strict non-backfilling: If a higher-ranked parent exceeds remaining budget, selection halts
   immediately without cherry-picking smaller lower-ranked parents.
6. Provenance lookup table maps every [Source X] to underlying document ID, parent chunk ID, and child chunk IDs.
7. Graceful handling of missing parent records in database.
8. Empty candidate inputs produce safe empty payloads.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from ragnarok.context.budget import TokenBudgetEngine
from ragnarok.context.contract import (
    ExpandedParentContext,
    MatchedChildReference,
)
from ragnarok.context.expansion import (
    ParentContextExpander,
    expand_parent_contexts,
)
from ragnarok.context.formatter import (
    ContextFormatter,
)
from ragnarok.context.service import ContextManagementService
from ragnarok.core.models import ChildChunk, Document, ParentChunk
from ragnarok.retrieval.contract import ChildResultCandidate
from ragnarok.storage.repository import RelationalRepository


class MockParentLookup:
    """Mock implementation of ParentLookupProtocol for deterministic unit testing."""

    def __init__(
        self,
        parents: dict[UUID, ParentChunk] | None = None,
        documents: dict[UUID, Document] | None = None,
    ) -> None:
        self.parents = parents or {}
        self.documents = documents or {}

    def get_parent_chunks_by_ids(self, parent_chunk_ids: list[UUID]) -> dict[UUID, ParentChunk]:
        return {pid: self.parents[pid] for pid in parent_chunk_ids if pid in self.parents}

    def get_documents_by_ids(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        return {did: self.documents[did] for did in document_ids if did in self.documents}


class SimpleWordCounter:
    """Deterministic token counter treating words as tokens for predictable budget tests."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


def test_parent_deduplication_and_best_child_rank() -> None:
    """
    Verifies that multiple child chunks referencing the same parent chunk ID
    are consolidated into a single parent entry, and the parent inherits the
    rank of its best-matching child (best_child_rank).
    """
    doc_id = uuid4()
    parent_1_id = uuid4()
    parent_2_id = uuid4()

    doc = Document(
        document_id=doc_id,
        filename="architectural_specs.md",
        raw_file_path="data/raw/architectural_specs.md",
        parsed_file_path="data/parsed/architectural_specs.md",
        file_type="markdown",
        file_size_bytes=1024,
        checksum="chk123",
    )

    parent_1 = ParentChunk(
        parent_chunk_id=parent_1_id,
        document_id=doc_id,
        content="Parent 1 content covering dual indexing and BM25s mechanics.",
        parent_index=0,
        heading_path=["Architecture", "Dual Indexing"],
    )

    parent_2 = ParentChunk(
        parent_chunk_id=parent_2_id,
        document_id=doc_id,
        content="Parent 2 content discussing vector search and TurboVec.",
        parent_index=1,
        heading_path=["Architecture", "Vector Search"],
    )

    lookup = MockParentLookup(
        parents={parent_1_id: parent_1, parent_2_id: parent_2},
        documents={doc_id: doc},
    )

    # 3 children: Child A (Rank 2) -> Parent 1
    #             Child B (Rank 1) -> Parent 2
    #             Child C (Rank 3) -> Parent 1
    candidates = [
        ChildResultCandidate(
            chunk_id=uuid4(),
            parent_chunk_id=parent_1_id,
            document_id=doc_id,
            final_rank=2,
            rerank_rank=2,
            rerank_score=0.91,
            score_type="rerank_score",
        ),
        ChildResultCandidate(
            chunk_id=uuid4(),
            parent_chunk_id=parent_2_id,
            document_id=doc_id,
            final_rank=1,
            rerank_rank=1,
            rerank_score=0.98,
            score_type="rerank_score",
        ),
        ChildResultCandidate(
            chunk_id=uuid4(),
            parent_chunk_id=parent_1_id,
            document_id=doc_id,
            final_rank=3,
            rerank_rank=3,
            rerank_score=0.85,
            score_type="rerank_score",
        ),
    ]

    expander = ParentContextExpander(parent_lookup=lookup)
    expanded = expander.expand(candidates)

    # Assertions
    # 1. Deduplicated: 3 children collapse to 2 unique parents
    assert len(expanded) == 2

    # 2. Strict best-child rank ordering: Parent 2 had Child B (Rank 1), Parent 1 had Child A (Rank 2)
    assert expanded[0].parent_chunk_id == parent_2_id
    assert expanded[0].best_child_rank == 1
    assert len(expanded[0].matched_children) == 1

    assert expanded[1].parent_chunk_id == parent_1_id
    assert expanded[1].best_child_rank == 2  # min(2, 3) = 2
    assert len(expanded[1].matched_children) == 2
    assert expanded[1].matched_children[0].rank == 2
    assert expanded[1].matched_children[1].rank == 3

    # Document filename resolved
    assert expanded[0].filename == "architectural_specs.md"
    assert expanded[1].filename == "architectural_specs.md"


def test_missing_parent_record_in_storage_is_safely_omitted() -> None:
    """
    Verifies that if a child references a parent ID missing in the database,
    the missing parent is safely omitted without raising unhandled exceptions.
    """
    doc_id = uuid4()
    known_parent_id = uuid4()
    missing_parent_id = uuid4()

    known_parent = ParentChunk(
        parent_chunk_id=known_parent_id,
        document_id=doc_id,
        content="Valid parent text",
        parent_index=0,
        heading_path=["Intro"],
    )

    lookup = MockParentLookup(
        parents={known_parent_id: known_parent},
        documents={},
    )

    candidates = [
        ChildResultCandidate(
            chunk_id=uuid4(),
            parent_chunk_id=known_parent_id,
            final_rank=1,
            rerank_rank=1,
        ),
        ChildResultCandidate(
            chunk_id=uuid4(),
            parent_chunk_id=missing_parent_id,
            final_rank=2,
            rerank_rank=2,
        ),
    ]

    expanded = expand_parent_contexts(candidates, lookup)
    assert len(expanded) == 1
    assert expanded[0].parent_chunk_id == known_parent_id


def test_token_budgeting_whole_parent_preservation_and_no_backfilling() -> None:
    """
    Verifies that:
    1. Parents are greedily accumulated until max_tokens.
    2. Zero parent text is partially sliced or truncated.
    3. Non-backfilling invariant: When a higher-ranked parent exceeds remaining budget,
       selection halts immediately, even if a lower-ranked smaller parent would fit.
    """
    counter = SimpleWordCounter()

    # Create 3 parents:
    # Parent 1: 5 words (fits, total=5)
    # Parent 2: 8 words (exceeds budget when limit is 10, needed 5+8=13 > 10)
    # Parent 3: 2 words (would fit 5+2=7 <= 10, but must NOT be included!)
    parent_1 = ExpandedParentContext(
        parent_chunk_id=uuid4(),
        document_id=uuid4(),
        filename="doc1.md",
        parent_index=0,
        content="one two three four five",  # 5 words
        best_child_rank=1,
        matched_children=[MatchedChildReference(chunk_id=uuid4(), rank=1)],
    )

    parent_2 = ExpandedParentContext(
        parent_chunk_id=uuid4(),
        document_id=uuid4(),
        filename="doc2.md",
        parent_index=1,
        content="alpha beta gamma delta epsilon zeta eta theta",  # 8 words
        best_child_rank=2,
        matched_children=[MatchedChildReference(chunk_id=uuid4(), rank=2)],
    )

    parent_3 = ExpandedParentContext(
        parent_chunk_id=uuid4(),
        document_id=uuid4(),
        filename="doc3.md",
        parent_index=2,
        content="small chunk",  # 2 words
        best_child_rank=3,
        matched_children=[MatchedChildReference(chunk_id=uuid4(), rank=3)],
    )

    parents = [parent_1, parent_2, parent_3]
    budget_limit = 10

    engine = TokenBudgetEngine(max_tokens=budget_limit, token_counter=counter)
    result = engine.apply_budget(parents)

    # Assertions
    # 1. Total tokens <= max_tokens
    assert result.total_tokens <= budget_limit
    assert result.total_tokens == 5

    # 2. Only Parent 1 is selected; Parent 2 and Parent 3 are omitted
    assert len(result.selected_parents) == 1
    assert result.selected_parents[0].parent_chunk_id == parent_1.parent_chunk_id
    assert result.selected_parents[0].content == "one two three four five"

    # 3. Non-backfilling verified: Parent 3 was NOT included
    assert len(result.omitted_parents) == 2
    assert result.omitted_parents[0].parent_chunk_id == parent_2.parent_chunk_id
    assert result.omitted_parents[1].parent_chunk_id == parent_3.parent_chunk_id


def test_structured_context_formatting_and_provenance_map() -> None:
    """
    Verifies that ContextFormatter produces standardized [Source X] delimited text
    and builds an accurate provenance lookup table.
    """
    p1_id = uuid4()
    p2_id = uuid4()
    doc1_id = uuid4()
    doc2_id = uuid4()
    c1_id = uuid4()
    c2_id = uuid4()

    parent_1 = ExpandedParentContext(
        parent_chunk_id=p1_id,
        document_id=doc1_id,
        filename="storage_spec.md",
        parent_index=0,
        heading_path=["Storage", "PostgreSQL"],
        content="PostgreSQL 16 handles metadata and parent chunks.",
        best_child_rank=1,
        matched_children=[
            MatchedChildReference(chunk_id=c1_id, rank=1, score=0.95, score_type="rerank_score")
        ],
    )

    parent_2 = ExpandedParentContext(
        parent_chunk_id=p2_id,
        document_id=doc2_id,
        filename="retrieval_spec.md",
        parent_index=2,
        heading_path=["Retrieval", "RRF Fusion"],
        content="RRF uses constant k=60 to fuse lexical and dense ranks.",
        best_child_rank=2,
        matched_children=[
            MatchedChildReference(chunk_id=c2_id, rank=2, score=0.88, score_type="fusion_score")
        ],
    )

    selected = [parent_1, parent_2]
    omitted: list[ExpandedParentContext] = []
    token_counts = [7, 10]

    formatter = ContextFormatter(block_delimiter="\n\n---\n\n")
    payload = formatter.format(
        selected_parents=selected,
        omitted_parents=omitted,
        max_tokens_budget=1000,
        total_tokens=17,
        token_counts=token_counts,
    )

    # 1. Structural schema assertions
    assert payload.total_tokens == 17
    assert payload.max_tokens_budget == 1000
    assert payload.selected_parent_count == 2
    assert payload.omitted_parent_count == 0
    assert len(payload.context_blocks) == 2

    # 2. Formatted text verification
    assert (
        "[Source 1] (Document: storage_spec.md, Section: Storage > PostgreSQL)"
        in payload.assembled_context_text
    )
    assert "PostgreSQL 16 handles metadata and parent chunks." in payload.assembled_context_text
    assert "---" in payload.assembled_context_text
    assert (
        "[Source 2] (Document: retrieval_spec.md, Section: Retrieval > RRF Fusion)"
        in payload.assembled_context_text
    )
    assert (
        "RRF uses constant k=60 to fuse lexical and dense ranks." in payload.assembled_context_text
    )

    # 3. Provenance map verification
    prov = payload.provenance_map
    assert "[Source 1]" in prov
    assert prov["[Source 1]"]["document_id"] == str(doc1_id)
    assert prov["[Source 1]"]["filename"] == "storage_spec.md"
    assert prov["[Source 1]"]["parent_chunk_id"] == str(p1_id)
    assert prov["[Source 1]"]["heading_path"] == ["Storage", "PostgreSQL"]
    assert prov["[Source 1]"]["best_child_rank"] == 1
    assert len(prov["[Source 1]"]["matched_children"]) == 1
    assert prov["[Source 1]"]["matched_children"][0]["chunk_id"] == str(c1_id)
    assert prov["[Source 1]"]["matched_children"][0]["rank"] == 1
    assert prov["[Source 1]"]["matched_children"][0]["score"] == 0.95

    assert "[Source 2]" in prov
    assert prov["[Source 2]"]["parent_chunk_id"] == str(p2_id)
    assert prov["[Source 2]"]["matched_children"][0]["chunk_id"] == str(c2_id)


def test_empty_candidates_returns_safe_empty_payload() -> None:
    """Verifies that an empty retrieval candidate list produces an empty context payload."""
    lookup = MockParentLookup()
    counter = SimpleWordCounter()

    service = ContextManagementService(
        expander=ParentContextExpander(lookup),
        budget_engine=TokenBudgetEngine(max_tokens=2000, token_counter=counter),
    )

    payload = service.build_context([])

    assert payload.total_tokens == 0
    assert payload.selected_parent_count == 0
    assert payload.omitted_parent_count == 0
    assert payload.assembled_context_text == ""
    assert payload.context_blocks == []
    assert payload.provenance_map == {}


def test_full_context_management_service_end_to_end() -> None:
    """
    End-to-end integration test verifying the full pipeline:
    Candidates -> Expansion -> Deduplication -> Ranking -> Budgeting -> Formatting.
    """
    doc_id = uuid4()
    p1 = uuid4()
    p2 = uuid4()

    doc = Document(
        document_id=doc_id,
        filename="guide.pdf",
        raw_file_path="data/raw/guide.pdf",
        parsed_file_path="data/parsed/guide.pdf",
        file_type="pdf",
        file_size_bytes=4096,
        checksum="chk999",
    )

    parent_1 = ParentChunk(
        parent_chunk_id=p1,
        document_id=doc_id,
        content="Section one details.",
        parent_index=0,
        heading_path=["Ch1"],
    )
    parent_2 = ParentChunk(
        parent_chunk_id=p2,
        document_id=doc_id,
        content="Section two details.",
        parent_index=1,
        heading_path=["Ch2"],
    )

    lookup = MockParentLookup(
        parents={p1: parent_1, p2: parent_2},
        documents={doc_id: doc},
    )

    candidates = [
        ChildResultCandidate(
            chunk_id=uuid4(),
            parent_chunk_id=p1,
            document_id=doc_id,
            final_rank=1,
            rerank_rank=1,
            rerank_score=0.99,
        ),
        ChildResultCandidate(
            chunk_id=uuid4(),
            parent_chunk_id=p2,
            document_id=doc_id,
            final_rank=2,
            rerank_rank=2,
            rerank_score=0.88,
        ),
    ]

    service = ContextManagementService.from_config(
        parent_lookup=lookup,
        token_counter=SimpleWordCounter(),
    )

    payload = service.build_context(candidates)

    assert payload.selected_parent_count == 2
    assert payload.omitted_parent_count == 0
    assert len(payload.context_blocks) == 2
    assert payload.context_blocks[0].parent_chunk_id == p1
    assert payload.context_blocks[1].parent_chunk_id == p2
    assert "[Source 1]" in payload.provenance_map
    assert "[Source 2]" in payload.provenance_map


def test_live_postgresql_parent_expansion_integration(repo: RelationalRepository) -> None:
    """
    Integration test verifying that ParentContextExpander works directly with
    RelationalRepository against a live PostgreSQL database instance.
    """
    doc_id = uuid4()
    doc = Document(
        document_id=doc_id,
        filename="live_integration_spec.md",
        raw_file_path=f"data/raw/{doc_id}.md",
        parsed_file_path=f"data/parsed/{doc_id}.md",
        file_type="markdown",
        file_size_bytes=2048,
        checksum=f"live_chk_{uuid4().hex[:12]}",
        status="parsed",
    )
    repo.insert_document(doc)

    p1_id = uuid4()
    p2_id = uuid4()
    parent_1 = ParentChunk(
        parent_chunk_id=p1_id,
        document_id=doc_id,
        content="Live PostgreSQL Parent 1 content.",
        parent_index=0,
        heading_path=["Live", "Section 1"],
    )
    parent_2 = ParentChunk(
        parent_chunk_id=p2_id,
        document_id=doc_id,
        content="Live PostgreSQL Parent 2 content.",
        parent_index=1,
        heading_path=["Live", "Section 2"],
    )

    c1_id = uuid4()
    c2_id = uuid4()
    child_1 = ChildChunk(
        chunk_id=c1_id,
        document_id=doc_id,
        parent_chunk_id=p1_id,
        content="Child 1 content.",
        chunk_index=0,
        heading_path=["Live", "Section 1"],
    )
    child_2 = ChildChunk(
        chunk_id=c2_id,
        document_id=doc_id,
        parent_chunk_id=p2_id,
        content="Child 2 content.",
        chunk_index=1,
        heading_path=["Live", "Section 2"],
    )

    repo.persist_chunks_atomically(
        document_id=doc_id,
        parent_chunks=[parent_1, parent_2],
        child_chunks=[child_1, child_2],
    )

    candidates = [
        ChildResultCandidate(
            chunk_id=c1_id,
            parent_chunk_id=p1_id,
            document_id=doc_id,
            final_rank=1,
            rerank_rank=1,
            rerank_score=0.99,
        ),
        ChildResultCandidate(
            chunk_id=c2_id,
            parent_chunk_id=p2_id,
            document_id=doc_id,
            final_rank=2,
            rerank_rank=2,
            rerank_score=0.90,
        ),
    ]

    expander = ParentContextExpander(parent_lookup=repo)
    expanded = expander.expand(candidates)

    assert len(expanded) == 2
    assert expanded[0].parent_chunk_id == p1_id
    assert expanded[0].filename == "live_integration_spec.md"
    assert expanded[0].content == "Live PostgreSQL Parent 1 content."
    assert expanded[0].best_child_rank == 1

    assert expanded[1].parent_chunk_id == p2_id
    assert expanded[1].best_child_rank == 2
