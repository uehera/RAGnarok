"""
Standalone Re-Indexing Subsystem.

Rebuilds BM25s lexical inverted index and TurboVec dense vector index directly
from PostgreSQL child chunk text without re-parsing raw source files.
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
    Sources text exclusively from PostgreSQL chunks table.
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
        Executes complete re-indexing across all child chunks in PostgreSQL:
        1. Reads finalized child chunks from PostgreSQL.
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
        logger.info("Retrieved child chunks from PostgreSQL for re-indexing", count=len(chunks))

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
