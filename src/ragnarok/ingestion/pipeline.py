"""
Ingestion Pipeline Coordinator Subsystem.

Coordinates end-to-end processing across all 5 stages of the ingestion pipeline:
- Stage 1: Validation, Deduplication & Staging (StagingService)
- Stage 2: Document Layout Parsing (DocumentParser)
- Stage 3: Two-Stage Parent-Child Chunking (TwoStageChunker)
- Stage 4: Atomic Relational Persistence (RelationalRepository)
- Stage 5: Dual Indexing Subsystem (ReindexService)
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import structlog

from ragnarok.core.config import AppConfig
from ragnarok.core.models import Document
from ragnarok.indexing.reindexer import ReindexService
from ragnarok.ingestion.chunker import TwoStageChunker
from ragnarok.ingestion.parser import DocumentParser
from ragnarok.ingestion.staging import StagingService
from ragnarok.storage.repository import RelationalRepository

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """
    Coordinates document onboarding through all ingestion, chunking, and indexing stages.
    """

    def __init__(
        self,
        repository: RelationalRepository,
        config: AppConfig,
        staging_service: StagingService | None = None,
        parser: DocumentParser | None = None,
        chunker: TwoStageChunker | None = None,
        reindexer: ReindexService | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self.staging_service = (
            staging_service
            if staging_service is not None
            else StagingService(
                repository=repository,
                staging_dir=config.ingestion.staging_directory,
                raw_dir=config.ingestion.raw_directory,
                supported_formats=config.ingestion.supported_formats,
            )
        )
        self.parser = (
            parser
            if parser is not None
            else DocumentParser(
                repository=repository,
                parsed_dir=config.ingestion.parsed_directory,
            )
        )
        self.chunker = (
            chunker
            if chunker is not None
            else TwoStageChunker(
                config=config.chunking,
                model_name=config.embedding.model,
            )
        )
        self.reindexer = (
            reindexer
            if reindexer is not None
            else ReindexService(
                repository=repository,
                config=config,
            )
        )

    def process_file(self, file_path: Path) -> tuple[Document | None, bool]:
        """
        Processes a single file through Stages 1 to 4:
        - Stages and calculates SHA-256 hash.
        - If already fully chunked in PostgreSQL, safely skips as duplicate.
        - Otherwise parses layout to Markdown (Stage 2).
        - Generates parent and child chunks (Stage 3).
        - Atomically persists chunks to PostgreSQL (Stage 4).

        Returns:
            tuple[Document | None, bool]: (document, is_duplicate)
        """
        # Stage 1: Validation, Deduplication & Staging
        staged_path = self.staging_service.stage_file(file_path)
        result = self.staging_service.process_file(staged_path)

        job_id: UUID | None = None
        if result.is_duplicate:
            existing_doc_id = result.existing_document_id
            if existing_doc_id is not None:
                existing_chunks = self.repository.query_chunk_ids_by_metadata(
                    document_id=existing_doc_id
                )
                if existing_chunks:
                    # Genuinely already fully ingested and chunked
                    existing_doc = self.repository.get_document(existing_doc_id)
                    return existing_doc, True

                # Document was previously staged but never chunked! Resume processing it.
                doc_id = existing_doc_id
                jobs = self.repository.get_jobs_for_document(doc_id)
                job_id = jobs[-1].job_id if jobs else None
            else:
                return None, True
        else:
            if result.document is None:
                return None, False
            doc_id = result.document.document_id
            job_id = result.job.job_id if result.job else None

        # Stage 2: Layout Parsing
        updated_doc, parsed_markdown_path = self.parser.parse_document(
            document_id=doc_id,
            job_id=job_id,
        )

        # Stage 3: Two-Stage Parent-Child Chunking
        markdown_content = parsed_markdown_path.read_text(encoding="utf-8")
        parent_chunks, child_chunks = self.chunker.chunk_markdown(
            markdown_content=markdown_content,
            document_id=doc_id,
        )

        # Stage 4: Atomic Relational Persistence
        self.repository.persist_chunks_atomically(
            document_id=doc_id,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            job_id=job_id,
        )

        logger.info(
            "Document ingested, chunked, and persisted to relational store",
            document_id=str(doc_id),
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
        )

        return updated_doc, False

    def reindex(self) -> int:
        """Stage 5: Builds BM25s and TurboVec indexes from all chunks in PostgreSQL."""
        return self.reindexer.reindex_all()
