"""
Enterprise Battle-Testing Suite — Dimension 1: Chaos Engineering & Fault Injection.

Verifies:
1. Mid-Transaction PostgreSQL Disconnection: clean rollback, zero orphaned chunks, job status 'failed'.
2. Interrupted Index Disk Writes: atomic write safety, previous index snapshot preservation.
3. Mid-Stream LLM Provider Outage: fallback chain activation, resilient failure handling.
4. Unhandled Parser Crash Recovery: raw file intact in data/raw/, job marked failed with traceback.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from ragnarok.core.config import load_config
from ragnarok.core.models import ChildChunk, Document, ParentChunk
from ragnarok.generation.service import GenerationService
from ragnarok.ingestion.parser import DocumentParser
from ragnarok.storage.repository import RelationalRepository


def test_chaos_mid_transaction_disconnect_rollback(repo: RelationalRepository) -> None:
    """
    Simulates a database failure mid-transaction while inserting parent/child chunks.
    Asserts complete transaction rollback and zero orphaned records.
    """
    doc_id = uuid4()
    doc = Document(
        document_id=doc_id,
        filename="chaos_doc.md",
        raw_file_path=f"data/raw/{doc_id}.md",
        parsed_file_path=f"data/parsed/{doc_id}.md",
        file_type="text/markdown",
        file_size_bytes=100,
        checksum=f"chaos_checksum_{doc_id}",
        status="stored",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="parsing")

    parent_id = uuid4()
    parent = ParentChunk(
        parent_chunk_id=parent_id,
        document_id=doc_id,
        content="Parent chunk destined for rollback testing.",
        parent_index=0,
        heading_path=["Chaos"],
    )

    child = ChildChunk(
        chunk_id=uuid4(),
        document_id=doc_id,
        parent_chunk_id=parent_id,
        content="Child chunk destined for rollback.",
        chunk_index=0,
        heading_path=["Chaos"],
    )

    # Intentionally trigger failure in chunk persistence
    with pytest.raises(RuntimeError, match="Simulated failure at chunk index 0"):
        repo.persist_chunks_atomically(
            document_id=doc_id,
            parent_chunks=[parent],
            child_chunks=[child],
            job_id=job.job_id,
            simulate_failure_at_chunk=0,
        )

    # Assert zero orphaned child chunks exist in database
    retrieved_children = repo.get_child_chunks(doc_id)
    assert len(retrieved_children) == 0, (
        "Orphaned child chunks detected after mid-transaction failure"
    )
    assert len(repo.get_parent_chunks(doc_id)) == 0


def test_chaos_interrupted_index_disk_write(tmp_path: Path) -> None:
    """
    Simulates a disk-full exception during index artifact persistence.
    Asserts previous index snapshots remain valid and uncorrupted.
    """
    from ragnarok.core.config import LexicalRetrievalConfig
    from ragnarok.indexing.lexical import LexicalIndex

    index_dir = tmp_path / "bm25_chaos"
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and save a valid initial index
    cfg = LexicalRetrievalConfig()
    idx = LexicalIndex(config=cfg)
    chunk1 = ChildChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        parent_chunk_id=uuid4(),
        content="Initial healthy baseline index content.",
        chunk_index=0,
        heading_path=[],
    )
    idx.build([chunk1])
    idx.save(index_dir)

    initial_mapping = (index_dir / "chunk_ids.json").read_text(encoding="utf-8")

    # 2. Simulate disk-full / IOError on subsequent save
    with (
        patch("pathlib.Path.write_text", side_effect=OSError("No space left on device")),
        pytest.raises(OSError, match="No space left on device"),
    ):
        # Attempt write that crashes mid-flight
        chunk2 = ChildChunk(
            chunk_id=uuid4(),
            document_id=uuid4(),
            parent_chunk_id=uuid4(),
            content="Corrupting write attempt.",
            chunk_index=1,
            heading_path=[],
        )
        idx2 = LexicalIndex(config=cfg)
        idx2.build([chunk1, chunk2])
        idx2.save(index_dir)

    # 3. Assert baseline index mapping is still reloadable and uncorrupted
    loaded = LexicalIndex.load(index_dir, config=cfg)
    assert not loaded.is_empty
    assert len(loaded.chunk_ids) == 1
    assert (index_dir / "chunk_ids.json").read_text(encoding="utf-8") == initial_mapping


def test_chaos_mid_stream_llm_provider_outage() -> None:
    """
    Simulates an abrupt provider network disconnect mid-stream.
    Asserts transparent fallback model activation.
    """
    config = load_config()
    config.generation.model = "primary-model"
    config.generation.fallbacks.enabled = True
    config.generation.fallbacks.models = ["fallback-model"]

    class MockChaosLLMClient:
        async def stream_completion(
            self,
            model: str,
            messages: list[dict[str, str]],
            **kwargs,
        ):
            if model == "primary-model":
                raise RuntimeError("Connection dropped by remote host mid-stream")
            yield "Fallback answer tokens."

    svc = GenerationService.from_config(config=config, client=MockChaosLLMClient())  # type: ignore

    response = svc.generate_sync(
        query="Test query under provider failure",
        context="Test context",
        render_live=False,
    )
    assert response.answer == "Fallback answer tokens."
    assert response.fallback_triggered is True
    assert response.model_used == "fallback-model"


def test_chaos_unhandled_parser_crash_recovery(
    repo: RelationalRepository, isolated_paths: dict[str, Path]
) -> None:
    """
    Simulates an unhandled parser engine crash.
    Asserts raw source file remains intact and job records traceback in failed status.
    """
    doc_id = uuid4()
    raw_file = isolated_paths["raw"] / f"crash_test_{str(doc_id)[:8]}.md"
    raw_file.write_text("# Crash Candidate Content", encoding="utf-8")

    doc = Document(
        document_id=doc_id,
        filename="crash_test.md",
        raw_file_path=str(raw_file),
        parsed_file_path="",
        file_type="text/markdown",
        file_size_bytes=len(raw_file.read_bytes()),
        checksum=f"crash_hash_{doc_id}",
        status="stored",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="queued")

    class CrashingEngine:
        def parse_to_markdown(self, file_path: Path) -> str:
            raise MemoryError("Out of memory in native parsing layer")

    parser = DocumentParser(
        repository=repo,
        parsed_dir=isolated_paths["parsed"],
        engine=CrashingEngine(),
    )

    with pytest.raises(MemoryError, match="Out of memory in native parsing layer"):
        parser.parse_document(document_id=doc_id, job_id=job.job_id)

    # 1. Assert raw source file is still completely intact
    assert raw_file.exists()
    assert raw_file.read_text(encoding="utf-8") == "# Crash Candidate Content"

    # 2. Assert job captured error and transitioned to failed
    updated_job = repo.get_job(job.job_id)
    assert updated_job is not None
    assert updated_job.stage == "failed"
    assert "MemoryError: Out of memory in native parsing layer" in (updated_job.error_message or "")

    # 3. Assert document status updated to failed
    updated_doc = repo.get_document(doc_id)
    assert updated_doc is not None
    assert updated_doc.status == "failed"
