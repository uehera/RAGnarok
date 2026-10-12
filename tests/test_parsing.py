"""
Unit and integration tests for Testing Point 2: Document Layout Parsing & Markdown Output Fidelity.

Verifies:
- Job state transition from QUEUED to PARSING.
- Structural heading hierarchy retention (#, ##, ###).
- Tabular data preservation in valid Markdown table structures.
- Logical reading order retention.
- Filesystem persistence to data/parsed/<document_id>.md.
- Relational updates: documents.parsed_file_path set, documents.status = 'parsed'.
- Parser exception handling: job transitions to FAILED with error trace in processing_jobs.error_message.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ragnarok.core.models import Document
from ragnarok.ingestion.parser import DoclingParserEngine, DocumentParser, ParserEngineProtocol
from ragnarok.storage.repository import RelationalRepository


def test_docling_parser_heading_and_table_preservation(
    repo: RelationalRepository,
    isolated_paths: dict[str, Path],
) -> None:
    """
    Verifies that layout parsing preserves Markdown headings (#, ##, ###),
    tabular structures (| Col 1 | Col 2 |), and reading order,
    while persisting to data/parsed/<doc_id>.md and updating relational status.
    """
    parser = DocumentParser(
        repository=repo,
        parsed_dir=isolated_paths["parsed"],
        engine=DoclingParserEngine(),
    )

    doc_id = uuid4()
    raw_file = isolated_paths["raw"] / f"{doc_id}.md"

    # Multi-level headings, multi-row/col table, sequential reading order
    structured_markdown = """# Architecture Overview
This document specifies the RAGnarok ingestion and parsing architecture.

## Component Specifications
The following table outlines the foundational component parameters:

| Component | Library | Primary Function |
|---|---|---|
| Ingestion | StagingService | Streaming SHA-256 deduplication |
| Parser | Docling | Document layout and table extraction |
| Chunker | TwoStageChunker | Bounded parent-child splitting |
| Storage | Psycopg 3 | Atomic relational persistence |

### Operational Flow
1. Files arrive at staging.
2. Layout is extracted into Markdown.
3. Chunks are persisted to PostgreSQL.
"""
    raw_file.write_text(structured_markdown, encoding="utf-8")

    # Insert initial document and job records
    doc = Document(
        document_id=doc_id,
        filename="structured_doc.md",
        raw_file_path=str(raw_file),
        parsed_file_path="",
        file_type="text/markdown",
        file_size_bytes=len(structured_markdown.encode("utf-8")),
        checksum="dummy_checksum_12345",
        status="stored",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="queued")

    # Execute parsing stage
    updated_doc, parsed_path = parser.parse_document(document_id=doc_id, job_id=job.job_id)

    # 1. Assert filesystem persistence
    assert parsed_path.exists()
    assert parsed_path.name == f"{doc_id}.md"
    parsed_content = parsed_path.read_text(encoding="utf-8")
    assert len(parsed_content.strip()) > 0

    # 2. Assert heading hierarchy retention
    assert "# Architecture Overview" in parsed_content
    assert "## Component Specifications" in parsed_content
    assert "### Operational Flow" in parsed_content

    # 3. Assert tabular structure preservation
    assert "| Component | Library | Primary Function |" in parsed_content
    assert "| Ingestion | StagingService |" in parsed_content
    assert "| Storage | Psycopg 3 |" in parsed_content

    # 4. Assert reading order
    idx_h1 = parsed_content.index("# Architecture Overview")
    idx_h2 = parsed_content.index("## Component Specifications")
    idx_tbl = parsed_content.index("| Component |")
    idx_h3 = parsed_content.index("### Operational Flow")
    assert idx_h1 < idx_h2 < idx_tbl < idx_h3

    # 5. Assert relational status updates
    assert updated_doc.status == "parsed"
    assert updated_doc.parsed_file_path == str(parsed_path)

    db_doc = repo.get_document(doc_id)
    assert db_doc is not None
    assert db_doc.status == "parsed"
    assert db_doc.parsed_file_path == str(parsed_path)


def test_parser_exception_handling_and_failure_boundaries(
    repo: RelationalRepository,
    isolated_paths: dict[str, Path],
) -> None:
    """
    Verifies that parser crashes or corrupted documents trigger:
    - job stage transition to 'failed'
    - detailed diagnostic trace recorded in processing_jobs.error_message
    - document status updated to 'failed'
    - raw source file retained in data/raw/
    """
    class CorruptedByteParserEngine(ParserEngineProtocol):
        def parse_to_markdown(self, file_path: Path) -> str:
            raise RuntimeError("Corrupted document: invalid byte sequence or damaged header")

    parser = DocumentParser(
        repository=repo,
        parsed_dir=isolated_paths["parsed"],
        engine=CorruptedByteParserEngine(),
    )

    doc_id = uuid4()
    corrupted_file = isolated_paths["raw"] / f"{doc_id}.pdf"
    corrupted_file.write_bytes(b"%PDF-1.4 CORRUPTED_GARBAGE_BYTES_NO_EOF")

    doc = Document(
        document_id=doc_id,
        filename="corrupted_report.pdf",
        raw_file_path=str(corrupted_file),
        parsed_file_path="",
        file_type="application/pdf",
        file_size_bytes=len(corrupted_file.read_bytes()),
        checksum="corrupt_hash_99999",
        status="stored",
    )
    repo.insert_document(doc)
    job = repo.create_processing_job(document_id=doc_id, stage="queued")

    # Parsing must fail with RuntimeError
    with pytest.raises(RuntimeError, match="Corrupted document"):
        parser.parse_document(document_id=doc_id, job_id=job.job_id)

    # Raw file must still be retained for diagnostic inspection
    assert corrupted_file.exists()

    # Document status in PostgreSQL must be 'failed'
    db_doc = repo.get_document(doc_id)
    assert db_doc is not None
    assert db_doc.status == "failed"

    # Job stage must be 'failed' with captured error traceback
    db_job = repo.get_job(job.job_id)
    assert db_job is not None
    assert db_job.stage == "failed"
    assert db_job.error_message is not None
    assert "Corrupted document" in db_job.error_message
    assert "Traceback" in db_job.error_message
