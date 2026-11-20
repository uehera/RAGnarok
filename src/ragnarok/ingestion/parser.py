"""
Document Layout Parsing Subsystem.

Extracts reading order, structural tables, and heading hierarchies (#, ##, ###)
using PyMuPDF (pymupdf4llm) for PDFs and direct UTF-8 file reads for Markdown and plain text.
Persists normalized Markdown artifacts to data/parsed/<doc_id>.md, updates relational status,
and captures parser exceptions in processing_jobs.error_message.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Protocol
from uuid import UUID

import structlog

from ragnarok.core.models import Document
from ragnarok.storage.repository import RelationalRepository

logger = structlog.get_logger(__name__)


class ParserEngineProtocol(Protocol):
    """Protocol contract for interchangeable document layout parsing backends."""

    def parse_to_markdown(self, file_path: Path) -> str:
        """Parses a raw document into normalized Markdown string."""
        ...


class PyMuPDFParserEngine:
    """
    Lightweight document parsing engine using PyMuPDF (pymupdf4llm) and direct file reads.
    Extracts layout reading order, markdown heading hierarchies, and structural tables from PDFs,
    and directly reads UTF-8 content from Markdown and plain text files.
    """

    def parse_to_markdown(self, file_path: Path) -> str:
        """Converts file into normalized Markdown."""
        suffix = file_path.suffix.lower()
        # Direct pass-through for Markdown and Plaintext files with zero parser overhead
        if suffix in (".md", ".txt"):
            return file_path.read_text(encoding="utf-8")

        if suffix == ".pdf":
            import pymupdf4llm

            markdown_text = pymupdf4llm.to_markdown(str(file_path))
            return markdown_text

        raise ValueError(f"Unsupported file format for parsing: {suffix}")


# Backward-compatibility alias
DoclingParserEngine = PyMuPDFParserEngine


class DocumentParser:
    """
    Stateful service managing the layout parsing stage (Stage 2).
    Transitions job state, persists Markdown artifacts, and handles failure boundaries.
    """

    def __init__(
        self,
        repository: RelationalRepository,
        parsed_dir: Path | str = "data/parsed",
        engine: ParserEngineProtocol | None = None,
    ) -> None:
        self.repository = repository
        self.parsed_dir = Path(parsed_dir)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine if engine is not None else PyMuPDFParserEngine()

    def parse_document(
        self,
        document_id: UUID,
        job_id: UUID | None = None,
    ) -> tuple[Document, Path]:
        """
        Executes layout parsing for a stored document:
        1. Transitions processing_job to 'parsing' and document to 'parsing'.
        2. Parses document layout using the configured parser engine.
        3. Writes normalized Markdown to data/parsed/<document_id>.md.
        4. Updates document status to 'parsed' and records parsed_file_path.
        5. On error: records full traceback into processing_jobs.error_message,
           transitions job to 'failed', document to 'failed', and re-raises exception.

        Returns:
            tuple[Document, Path]: (updated_document, path_to_parsed_markdown)
        """
        doc = self.repository.get_document(document_id)
        if doc is None:
            raise ValueError(f"Document {document_id} not found in database")

        # Resolve processing job if not explicitly provided
        if job_id is None:
            jobs = self.repository.get_jobs_for_document(document_id)
            if jobs:
                job_id = jobs[-1].job_id

        # 1. Transition to PARSING stage
        self.repository.update_document_status(document_id, status="parsing")
        if job_id is not None:
            self.repository.update_job_stage(job_id, stage="parsing")

        raw_path = Path(doc.raw_file_path)
        if not raw_path.exists():
            if (Path("data/raw") / doc.filename).exists():
                import shutil

                shutil.copy2(Path("data/raw") / doc.filename, raw_path)
            elif Path(doc.filename).exists():
                import shutil

                shutil.copy2(Path(doc.filename), raw_path)

        if doc.parsed_file_path and doc.parsed_file_path.strip():
            parsed_target_path = Path(doc.parsed_file_path)
        else:
            parsed_target_path = self.parsed_dir / f"{document_id}.md"

        try:
            logger.info(
                "Starting document layout parsing",
                document_id=str(document_id),
                raw_path=str(raw_path),
            )

            # 2. Extract layout and normalized markdown (or reuse existing parsed artifact)
            candidate_files = [
                self.parsed_dir / f"{raw_path.stem}.md",
                self.parsed_dir / f"{Path(doc.filename).stem}.md",
            ]
            prefix = doc.filename.split("_")[0]
            for p in self.parsed_dir.glob("*.md"):
                if p.name.startswith(prefix) and p.stat().st_size > 0:
                    candidate_files.append(p)

            candidate_md = next(
                (c for c in candidate_files if c.exists() and c.stat().st_size > 0), None
            )
            if candidate_md is not None:
                logger.info(
                    "Reusing existing parsed markdown artifact",
                    path=str(candidate_md),
                    document_id=str(document_id),
                )
                markdown_content = candidate_md.read_text(encoding="utf-8")
            else:
                markdown_content = self.engine.parse_to_markdown(raw_path)

            if not markdown_content.strip():
                raise ValueError("Parser returned empty content from source document")

            # 3. Persist markdown artifact to disk
            parsed_target_path.write_text(markdown_content, encoding="utf-8")

            # 4. Update document status in database
            self.repository.update_document_status(
                document_id=document_id,
                status="parsed",
                parsed_file_path=str(parsed_target_path),
            )

            updated_doc = self.repository.get_document(document_id)
            assert updated_doc is not None

            logger.info(
                "Document layout parsing completed successfully",
                document_id=str(document_id),
                parsed_path=str(parsed_target_path),
                size_bytes=len(markdown_content.encode("utf-8")),
            )

            return updated_doc, parsed_target_path

        except Exception as exc:
            error_trace = traceback.format_exc()
            logger.error(
                "Document layout parsing failed",
                document_id=str(document_id),
                error=str(exc),
            )

            # 5. Capture error in processing_jobs and update statuses to failed
            self.repository.update_document_status(document_id, status="failed")
            if job_id is not None:
                self.repository.update_job_stage(
                    job_id=job_id,
                    stage="failed",
                    error_message=error_trace,
                )

            raise


def parse_document_to_markdown(file_path: Path, engine: ParserEngineProtocol | None = None) -> str:
    """
    Pure procedural function that converts a document file into normalized Markdown.
    """
    eng = engine if engine is not None else PyMuPDFParserEngine()
    return eng.parse_to_markdown(file_path)
