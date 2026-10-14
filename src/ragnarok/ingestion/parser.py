"""
Document Layout Parsing Subsystem.

Wraps Docling to extract reading order, structural tables, and heading hierarchies (#, ##, ###).
Persists normalized Markdown artifacts to data/parsed/<doc_id>.md, updates relational status,
and captures parser exceptions in processing_jobs.error_message.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Protocol
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


class DoclingParserEngine:
    """
    Default parsing engine using Docling.
    Extracts layout reading order, markdown heading hierarchies, and structural tables.
    """

    def __init__(self) -> None:
        self._converter: Any = None

    def _get_converter(self) -> Any:
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def parse_to_markdown(self, file_path: Path) -> str:
        """Converts file into normalized Markdown using Docling."""
        suffix = file_path.suffix.lower()
        # Fast path for raw Markdown and Plaintext files
        if suffix in (".md", ".txt"):
            return file_path.read_text(encoding="utf-8")

        converter = self._get_converter()
        conv_result = converter.convert(str(file_path))
        markdown_text = conv_result.document.export_to_markdown()
        return markdown_text


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
        self.engine = engine if engine is not None else DoclingParserEngine()

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

            # 2. Extract layout and normalized markdown
            markdown_content = self.engine.parse_to_markdown(raw_path)
            if not markdown_content.strip():
                raise ValueError("Parser returned empty content from source document")

            # 3. Persist markdown artifact to disk
            parsed_target_path.write_text(markdown_content, encoding="utf-8")

            # 4. Update document status in PostgreSQL
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
    eng = engine if engine is not None else DoclingParserEngine()
    return eng.parse_to_markdown(file_path)
