"""
Unit and integration tests for Testing Point 1: Ingestion, Validation & Staging.

Verifies:
- File receipt into staging.
- Streaming SHA-256 calculation.
- Duplicate detection and duplicate rejection (purging staging file).
- Permanent storage in data/raw/<document_id>.<ext>.
- Rejection of invalid, unreadable, or empty files.
- Initial job registration in processing_jobs with stage 'QUEUED'.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ragnarok.ingestion.staging import (
    DuplicateDocumentError,
    StagingService,
    compute_sha256,
    detect_file_type,
    validate_staging_file,
)
from ragnarok.storage.repository import RelationalRepository


def test_streaming_sha256_calculation(tmp_path: Path) -> None:
    """Verifies that compute_sha256 produces exact cryptographic hashes."""
    content = b"RAGnarok high-performance streaming checksum verification test payload."
    test_file = tmp_path / "test_hash.txt"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    computed_hash = compute_sha256(test_file, chunk_size=16)

    assert computed_hash == expected_hash


def test_detect_file_type_mapping(tmp_path: Path) -> None:
    """Verifies MIME type resolution across supported document extensions."""
    assert detect_file_type(Path("doc.md")) == "text/markdown"
    assert detect_file_type(Path("doc.txt")) == "text/plain"
    assert detect_file_type(Path("doc.pdf")) == "application/pdf"
    assert (
        detect_file_type(Path("doc.docx"))
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_validate_staging_file_rejections(tmp_path: Path) -> None:
    """Verifies rejection of non-existent, empty, and unsupported files."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError, match="File not found"):
        validate_staging_file(tmp_path / "missing.pdf")

    # 2. Empty file (0 bytes)
    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="File is empty"):
        validate_staging_file(empty_file)

    # 3. Unsupported format
    unsupported = tmp_path / "script.exe"
    unsupported.write_bytes(b"MZ\x90\x00")
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_staging_file(unsupported, supported_formats=[".pdf", ".docx", ".md", ".txt"])


def test_staging_service_unique_onboarding(
    repo: RelationalRepository,
    isolated_paths: dict[str, Path],
) -> None:
    """
    Verifies that a valid, unique file is staged, moved to data/raw/<doc_id>.<ext>,
    and registered in documents and processing_jobs with stage 'QUEUED'.
    """
    service = StagingService(
        repository=repo,
        staging_dir=isolated_paths["staging"],
        raw_dir=isolated_paths["raw"],
        parsed_dir=isolated_paths["parsed"],
        supported_formats=[".md", ".txt", ".pdf", ".docx"],
    )

    # Create incoming document
    source_content = "# System Overview\n\nWelcome to the RAGnarok architectural documentation."
    source_file = isolated_paths["staging"] / "architecture_guide.md"
    source_file.write_bytes(source_content.encode("utf-8"))

    # Process staged file
    result = service.process_file(source_file)

    assert not result.is_duplicate
    assert result.document is not None
    assert result.job is not None

    doc = result.document
    job = result.job

    # Verify document fields
    assert doc.status == "stored"
    assert doc.filename == "architecture_guide.md"
    assert doc.checksum == hashlib.sha256(source_content.encode("utf-8")).hexdigest()

    # Verify filesystem movement: source in staging is gone, raw file exists
    assert not source_file.exists()
    raw_path = Path(doc.raw_file_path)
    assert raw_path.exists()
    assert raw_path.read_text(encoding="utf-8") == source_content
    assert raw_path.name == f"architecture_guide_{str(doc.document_id)[:8]}.md"

    # Verify processing job created with stage QUEUED
    assert job.stage == "queued"
    assert job.document_id == doc.document_id
    assert job.error_message is None

    # Verify PostgreSQL persistence
    db_doc = repo.get_document(doc.document_id)
    assert db_doc is not None
    assert db_doc.checksum == doc.checksum

    db_job = repo.get_job(job.job_id)
    assert db_job is not None
    assert db_job.stage == "queued"


def test_staging_duplicate_detection_and_rejection(
    repo: RelationalRepository,
    isolated_paths: dict[str, Path],
) -> None:
    """
    Verifies that re-uploading an identical file is detected as a duplicate,
    emits warning, deletes the staging file, and halts further database entry.
    """
    service = StagingService(
        repository=repo,
        staging_dir=isolated_paths["staging"],
        raw_dir=isolated_paths["raw"],
        parsed_dir=isolated_paths["parsed"],
    )

    content = b"Unique content intended to test duplicate rejection logic in RAGnarok."
    file1 = isolated_paths["staging"] / "first_upload.txt"
    file1.write_bytes(content)

    # Ingest original file
    res1 = service.process_file(file1)
    assert not res1.is_duplicate
    assert res1.document is not None
    orig_doc_id = res1.document.document_id

    # Create duplicate in staging
    file2 = isolated_paths["staging"] / "second_upload_duplicate.txt"
    file2.write_bytes(content)

    # Process duplicate file
    res2 = service.process_file(file2)
    assert res2.is_duplicate
    assert res2.existing_document_id == orig_doc_id
    assert res2.document is None

    # Staging file must be purged
    assert not file2.exists()

    # Duplicate with raise_on_duplicate=True
    file3 = isolated_paths["staging"] / "third_upload_raise.txt"
    file3.write_bytes(content)
    with pytest.raises(DuplicateDocumentError) as exc_info:
        service.process_file(file3, raise_on_duplicate=True)

    assert exc_info.value.existing_document_id == orig_doc_id
    assert not file3.exists()
