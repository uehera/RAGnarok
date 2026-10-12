"""
Unit and integration tests for Testing Point 4: PostgreSQL Storage Success & Relational Integrity.

Verifies:
- Atomic transaction boundaries:
  - Commits parents and children atomically.
  - Full rollback on failure (simulated error at chunk 50) leaving zero orphaned records.
- Uniqueness constraints:
  - (document_id, parent_index) unique constraint enforcement.
  - (document_id, chunk_index) unique constraint enforcement.
- Cascading delete integrity:
  - Deleting a document cascades to delete parent_chunks, chunks, and processing_jobs.
  - Deleting a parent_chunk cascades to delete all child chunks referencing it.
"""

from __future__ import annotations

from uuid import uuid4

import psycopg.errors
import pytest

from ragnarok.core.models import ChildChunk, Document, ParentChunk
from ragnarok.storage.repository import RelationalRepository


def test_atomic_persistence_and_rollback_on_failure(
    repo: RelationalRepository,
) -> None:
    """
    Simulates a failure during persistence at chunk 50.
    Asserts that the transaction rolls back completely with zero orphaned records
    in parent_chunks or chunks.
    """
    doc_id = uuid4()
    doc = Document(
        document_id=doc_id,
        filename="atomic_test.md",
        raw_file_path=f"data/raw/{doc_id}.md",
        parsed_file_path=f"data/parsed/{doc_id}.md",
        file_type="text/markdown",
        file_size_bytes=4096,
        checksum=f"atomic_chk_{uuid4().hex[:12]}",
        status="parsed",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="parsing")

    # Generate 5 parents and 60 child chunks
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for p_idx in range(5):
        p_id = uuid4()
        parents.append(
            ParentChunk(
                parent_chunk_id=p_id,
                document_id=doc_id,
                content=f"Parent chunk context block {p_idx}",
                parent_index=p_idx,
                heading_path=[f"Section {p_idx}"],
            )
        )
        for c_offset in range(12):
            c_idx = (p_idx * 12) + c_offset
            children.append(
                ChildChunk(
                    chunk_id=uuid4(),
                    document_id=doc_id,
                    parent_chunk_id=p_id,
                    content=f"Child chunk content block {c_idx}",
                    chunk_index=c_idx,
                    heading_path=[f"Section {p_idx}"],
                )
            )

    assert len(children) == 60

    # 1. Simulate failure at chunk 50
    with pytest.raises(RuntimeError, match="Simulated failure at chunk index 50"):
        repo.persist_chunks_atomically(
            document_id=doc_id,
            parent_chunks=parents,
            child_chunks=children,
            job_id=job.job_id,
            simulate_failure_at_chunk=50,
        )

    # Assert complete rollback: zero parent chunks and zero child chunks stored
    assert repo.get_parent_chunks(doc_id) == []
    assert repo.get_child_chunks(doc_id) == []

    # Job stage should not have advanced to chunking
    persisted_job = repo.get_job(job.job_id)
    assert persisted_job is not None
    assert persisted_job.stage == "parsing"

    # 2. Persist cleanly without failure
    repo.persist_chunks_atomically(
        document_id=doc_id,
        parent_chunks=parents,
        child_chunks=children,
        job_id=job.job_id,
    )

    # Assert clean commit
    persisted_parents = repo.get_parent_chunks(doc_id)
    persisted_children = repo.get_child_chunks(doc_id)
    assert len(persisted_parents) == 5
    assert len(persisted_children) == 60

    persisted_job_after = repo.get_job(job.job_id)
    assert persisted_job_after is not None
    assert persisted_job_after.stage == "chunking"


def test_unique_ordering_constraints(
    repo: RelationalRepository,
) -> None:
    """
    Verifies that inserting duplicate (document_id, parent_index) or
    (document_id, chunk_index) raises a database unique violation.
    """
    doc_id = uuid4()
    doc = Document(
        document_id=doc_id,
        filename="unique_test.md",
        raw_file_path=f"data/raw/{doc_id}.md",
        parsed_file_path=f"data/parsed/{doc_id}.md",
        file_type="text/markdown",
        file_size_bytes=1024,
        checksum=f"unique_chk_{uuid4().hex[:12]}",
        status="parsed",
    )
    repo.insert_document(doc)

    parent1 = ParentChunk(
        document_id=doc_id,
        content="Parent 0",
        parent_index=0,
    )
    parent2 = ParentChunk(
        document_id=doc_id,
        content="Duplicate Parent 0",
        parent_index=0,  # Duplicate parent_index for same document_id
    )

    # Duplicate parent_index must raise UniqueViolation
    with pytest.raises(psycopg.errors.UniqueViolation):
        repo.persist_chunks_atomically(
            document_id=doc_id,
            parent_chunks=[parent1, parent2],
            child_chunks=[],
        )

    # Valid parent insertion
    repo.persist_chunks_atomically(
        document_id=doc_id,
        parent_chunks=[parent1],
        child_chunks=[],
    )

    child1 = ChildChunk(
        document_id=doc_id,
        parent_chunk_id=parent1.parent_chunk_id,
        content="Child 0",
        chunk_index=0,
    )
    child2 = ChildChunk(
        document_id=doc_id,
        parent_chunk_id=parent1.parent_chunk_id,
        content="Duplicate Child 0",
        chunk_index=0,  # Duplicate chunk_index for same document_id
    )

    # Duplicate chunk_index must raise UniqueViolation
    with pytest.raises(psycopg.errors.UniqueViolation):
        repo.persist_chunks_atomically(
            document_id=doc_id,
            parent_chunks=[],
            child_chunks=[child1, child2],
        )


def test_cascading_deletes_integrity(
    repo: RelationalRepository,
) -> None:
    """
    Verifies that:
    1. Deleting a parent chunk cascades to delete all child chunks referencing it.
    2. Deleting a document cascades to delete all parent chunks, child chunks, and jobs.
    """
    doc_id = uuid4()
    doc = Document(
        document_id=doc_id,
        filename="cascade_test.md",
        raw_file_path=f"data/raw/{doc_id}.md",
        parsed_file_path=f"data/parsed/{doc_id}.md",
        file_type="text/markdown",
        file_size_bytes=2048,
        checksum=f"cascade_chk_{uuid4().hex[:12]}",
        status="parsed",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="parsing")

    p1 = ParentChunk(document_id=doc_id, content="Parent 1", parent_index=0)
    p2 = ParentChunk(document_id=doc_id, content="Parent 2", parent_index=1)

    c1 = ChildChunk(
        document_id=doc_id, parent_chunk_id=p1.parent_chunk_id, content="Child 1A", chunk_index=0
    )
    c2 = ChildChunk(
        document_id=doc_id, parent_chunk_id=p1.parent_chunk_id, content="Child 1B", chunk_index=1
    )
    c3 = ChildChunk(
        document_id=doc_id, parent_chunk_id=p2.parent_chunk_id, content="Child 2A", chunk_index=2
    )

    repo.persist_chunks_atomically(
        document_id=doc_id,
        parent_chunks=[p1, p2],
        child_chunks=[c1, c2, c3],
    )

    assert len(repo.get_parent_chunks(doc_id)) == 2
    assert len(repo.get_child_chunks(doc_id)) == 3

    # 1. Delete parent chunk p1: c1 and c2 must be cascade-deleted; p2 and c3 remain
    assert repo.delete_parent_chunk(p1.parent_chunk_id) is True
    remaining_parents = repo.get_parent_chunks(doc_id)
    remaining_children = repo.get_child_chunks(doc_id)
    assert len(remaining_parents) == 1
    assert remaining_parents[0].parent_chunk_id == p2.parent_chunk_id
    assert len(remaining_children) == 1
    assert remaining_children[0].chunk_id == c3.chunk_id

    # 2. Delete document: all remaining parents, children, and processing jobs must be deleted
    assert repo.delete_document(doc_id) is True
    assert repo.get_document(doc_id) is None
    assert repo.get_parent_chunks(doc_id) == []
    assert repo.get_child_chunks(doc_id) == []
    assert repo.get_job(job.job_id) is None
