"""
Standalone Re-Indexing Subsystem.

Rebuilds BM25s lexical inverted index and TurboVec dense vector index directly
from DuckDB child chunk text without re-parsing raw source files.
Finalizes processing job lifecycles with state transitions to COMPLETED.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import structlog

from ragnarok.core.config import AppConfig
from ragnarok.indexing.dense import DenseIndex, EmbedderProtocol
from ragnarok.indexing.lexical import LexicalIndex
from ragnarok.storage.repository import RelationalRepository

logger = structlog.get_logger(__name__)


class ReindexService:
    """
    Coordinates standalone regeneration of both lexical and dense indexes.
    Sources text exclusively from DuckDB chunks table.
    """

    def __init__(
        self,
        repository: RelationalRepository,
        config: AppConfig | None = None,
        bm25_index_dir: Path | str = "data/indexes/bm25",
        dense_index_dir: Path | str = "data/indexes/dense",
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self.repository = repository
        self.config = config if config is not None else AppConfig()
        self.bm25_index_dir = Path(bm25_index_dir)
        self.dense_index_dir = Path(dense_index_dir)
        self.embedder = embedder

        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)
        self.dense_index_dir.mkdir(parents=True, exist_ok=True)

    def reindex_all(self, job_id: UUID | None = None) -> int:
        """
        Executes complete re-indexing across all child chunks in DuckDB:
        1. Reads finalized child chunks from DuckDB.
        2. Transitions processing job to 'indexing' if job_id is provided.
        3. Builds BM25s index and serializes to data/indexes/bm25/.
        4. Builds TurboVec dense index and serializes to data/indexes/dense/.
        5. Marks processing job 'completed' with completed_at timestamp.

        Returns:
            int: Number of chunks indexed.
        """
        if job_id is not None:
            self.repository.update_job_stage(job_id=job_id, stage="indexing")

        chunks = self.repository.get_all_child_chunks()
        logger.info("Retrieved child chunks from DuckDB for re-indexing", count=len(chunks))

        # 1. Build and save lexical index
        lexical_idx = LexicalIndex(config=self.config.lexical_retrieval)
        lexical_idx.build(chunks)
        lexical_idx.save(self.bm25_index_dir)

        # 2. Build and save dense index
        dense_idx = DenseIndex(
            config=self.config.dense_retrieval,
            embedder=self.embedder,
        )
        dense_idx.build(chunks)
        dense_idx.save(self.dense_index_dir)

        # 3. Finalize processing job
        if job_id is not None:
            self.repository.update_job_stage(
                job_id=job_id,
                stage="completed",
                completed=True,
            )

        logger.info("Successfully completed full dual re-indexing", count=len(chunks))
        return len(chunks)

    def reindex_document(self, document_id: UUID, job_id: UUID | None = None) -> int:
        """
        Incrementally indexes chunks for a newly ingested document into existing search indexes.
        Avoids full re-embedding of historical documents.

        1. Fetches child chunks for the target document.
        2. Loads existing TurboVec index from disk (or falls back to full build if empty/missing).
        3. Embeds ONLY the new child chunks and appends them to TurboVec.
        4. Rebuilds BM25s lexical index across all chunks (fast, < 50ms).
        5. Updates job stage to 'completed'.

        Returns:
            int: Total number of chunks in the updated index.
        """
        if job_id is not None:
            self.repository.update_job_stage(job_id=job_id, stage="indexing")

        new_chunks = self.repository.get_child_chunks(document_id)
        if not new_chunks:
            if job_id is not None:
                self.repository.update_job_stage(job_id=job_id, stage="completed", completed=True)
            all_chunks = self.repository.get_all_child_chunks()
            return len(all_chunks)

        # 1. Incremental dense vector update
        dense_idx = DenseIndex.load(
            index_dir=self.dense_index_dir,
            config=self.config.dense_retrieval,
            embedder=self.embedder,
        )
        if dense_idx.is_empty:
            # Fallback to full build if index does not exist yet
            all_chunks = self.repository.get_all_child_chunks()
            dense_idx.build(all_chunks)
        else:
            dense_idx.add_chunks(new_chunks)
        dense_idx.save(self.dense_index_dir)

        # 2. Fast lexical index update
        all_chunks = self.repository.get_all_child_chunks()
        lexical_idx = LexicalIndex(config=self.config.lexical_retrieval)
        lexical_idx.build(all_chunks)
        lexical_idx.save(self.bm25_index_dir)

        # 3. Finalize processing job
        if job_id is not None:
            self.repository.update_job_stage(
                job_id=job_id,
                stage="completed",
                completed=True,
            )

        logger.info(
            "Incrementally updated dual search indexes for document",
            document_id=str(document_id),
            new_chunks=len(new_chunks),
            total_chunks=len(all_chunks),
        )
        return len(all_chunks)
