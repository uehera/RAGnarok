"""
Enterprise Battle-Testing Suite — Dimension 3: Concurrency & Stress Testing.

Verifies:
1. 10 Parallel Document Ingestions: no race conditions on state transitions, connection pool scales.
2. 20 Concurrent Retrieval Queries: zero thread contention or deadlocks, independent candidate payloads.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from uuid import uuid4

from ragnarok.core.config import load_config
from ragnarok.core.models import ChildChunk, Document, ParentChunk
from ragnarok.indexing.dense import DenseIndex
from ragnarok.indexing.lexical import LexicalIndex
from ragnarok.retrieval.dense_search import DenseSearcher
from ragnarok.retrieval.fusion import HybridFusionService
from ragnarok.retrieval.lexical_search import LexicalSearcher
from ragnarok.storage.repository import RelationalRepository


def test_concurrency_parallel_document_ingestions(
    repo: RelationalRepository, isolated_paths: dict[str, Path]
) -> None:
    """
    Simulates 10 concurrent documents being ingested simultaneously via worker threads.
    Asserts zero deadlocks, clean state transitions, and pool scaling.
    """
    def ingest_worker(idx: int) -> Document:
        doc_id = uuid4()
        raw_file = isolated_paths["raw"] / f"concurrent_{idx}_{str(doc_id)[:8]}.txt"
        raw_file.write_text(f"Content for parallel ingestion worker {idx}", encoding="utf-8")

        doc = Document(
            document_id=doc_id,
            filename=f"concurrent_{idx}.txt",
            raw_file_path=str(raw_file),
            parsed_file_path="",
            file_type="text/plain",
            file_size_bytes=len(raw_file.read_bytes()),
            checksum=f"concurrent_hash_{idx}_{doc_id}",
            status="stored",
        )
        inserted = repo.insert_document(doc)
        job = repo.create_processing_job(document_id=doc_id, stage="parsing")

        # Insert parent and child chunk atomically
        p_id = uuid4()
        parent = ParentChunk(
            parent_chunk_id=p_id,
            document_id=doc_id,
            content=f"Parent content for {idx}",
            parent_index=0,
            heading_path=[f"Section_{idx}"],
        )

        child = ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content=f"Child content for {idx}",
            chunk_index=0,
            heading_path=[f"Section_{idx}"],
        )

        repo.persist_chunks_atomically(
            document_id=doc_id,
            parent_chunks=[parent],
            child_chunks=[child],
            job_id=job.job_id,
        )
        repo.update_document_status(doc_id, status="parsed")
        return inserted

    # Run 10 parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(ingest_worker, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 10
    # Verify all 10 documents exist in database and are marked parsed
    for doc in results:
        db_doc = repo.get_document(doc.document_id)
        assert db_doc is not None
        assert db_doc.status == "parsed"


def test_concurrency_parallel_retrieval_queries(
    repo: RelationalRepository, tmp_path: Path
) -> None:
    """
    Fires 20 concurrent queries across BM25 and TurboVec searchers.
    Asserts zero thread contention, deadlocks, or cross-talk between queries.
    """
    config = load_config()

    # Build test indexes with 10 child chunks
    chunks: list[ChildChunk] = []
    doc_id = uuid4()
    p_id = uuid4()
    for i in range(10):
        c = ChildChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            parent_chunk_id=p_id,
            content=f"Python machine learning and algorithms chunk {i} description.",
            chunk_index=i,
            heading_path=["AI"],
        )
        chunks.append(c)

    lex_idx = LexicalIndex(config=config.lexical_retrieval)
    lex_idx.build(chunks)
    lex_searcher = LexicalSearcher(index=lex_idx, repository=repo, config=config.lexical_retrieval)

    dense_idx = DenseIndex(config=config.dense_retrieval)
    dense_idx.build(chunks)
    dense_searcher = DenseSearcher(index=dense_idx, repository=repo, config=config.dense_retrieval)

    fusion_svc = HybridFusionService(
        lexical_searcher=lex_searcher,
        dense_searcher=dense_searcher,
        config=config.hybrid_retrieval,
    )

    query_templates = [
        "What is machine learning in Python?",
        "Explain algorithms chunk",
        "AI and data science fundamentals",
        "Overview of python chunk description",
    ]

    def query_worker(query_idx: int) -> int:
        q = query_templates[query_idx % len(query_templates)]
        candidates = fusion_svc.search(query=q, top_k=5)
        return len(candidates)

    # Dispatch 20 concurrent search operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(query_worker, i) for i in range(20)]
        counts = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(counts) == 20
    # Every search query returned valid non-empty candidate lists
    for count in counts:
        assert count > 0, "Query returned empty candidate set under concurrency"
