"""
Relational Repository Subsystem.

Provides atomic transactional persistence and metadata queries for documents,
parent chunks, child chunks, and processing jobs using Psycopg 3.
Enforces zero orphaned records, cascading deletes, and unique ordering constraints.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import structlog
from psycopg import sql

from ragnarok.core.models import (
    ChildChunk,
    Document,
    DocumentStatus,
    JobStage,
    ParentChunk,
    ProcessingJob,
)
from ragnarok.storage.database import DatabasePool

logger = structlog.get_logger(__name__)


class RelationalRepository:
    """
    Data access layer for PostgreSQL metadata store.
    Encapsulates all SQL queries and manages atomic transactional boundaries.
    """

    def __init__(self, pool: DatabasePool) -> None:
        self.pool = pool

    def get_document_by_checksum(self, checksum: str) -> Document | None:
        """Looks up a document by its cryptographic SHA-256 checksum."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, filename, raw_file_path, parsed_file_path,
                       file_type, file_size_bytes, checksum, status, created_at
                FROM documents
                WHERE checksum = %s;
                """,
                (checksum,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return Document(
                document_id=row[0],
                filename=row[1],
                raw_file_path=row[2],
                parsed_file_path=row[3],
                file_type=row[4],
                file_size_bytes=row[5],
                checksum=row[6],
                status=row[7],
                created_at=row[8],
            )

    def get_document(self, document_id: UUID) -> Document | None:
        """Retrieves a document record by its primary key UUID."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, filename, raw_file_path, parsed_file_path,
                       file_type, file_size_bytes, checksum, status, created_at
                FROM documents
                WHERE document_id = %s;
                """,
                (document_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return Document(
                document_id=row[0],
                filename=row[1],
                raw_file_path=row[2],
                parsed_file_path=row[3],
                file_type=row[4],
                file_size_bytes=row[5],
                checksum=row[6],
                status=row[7],
                created_at=row[8],
            )

    def get_document_by_filename(self, filename: str) -> Document | None:
        """Looks up a document record by its exact filename."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, filename, raw_file_path, parsed_file_path,
                       file_type, file_size_bytes, checksum, status, created_at
                FROM documents
                WHERE filename = %s;
                """,
                (filename,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return Document(
                document_id=row[0],
                filename=row[1],
                raw_file_path=row[2],
                parsed_file_path=row[3],
                file_type=row[4],
                file_size_bytes=row[5],
                checksum=row[6],
                status=row[7],
                created_at=row[8],
            )

    def get_documents_by_ids(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        """Retrieves document records matching a list of document UUIDs."""
        if not document_ids:
            return {}
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, filename, raw_file_path, parsed_file_path,
                       file_type, file_size_bytes, checksum, status, created_at
                FROM documents
                WHERE document_id = ANY(%s);
                """,
                (document_ids,),
            )
            rows = cur.fetchall()
            return {
                row[0]: Document(
                    document_id=row[0],
                    filename=row[1],
                    raw_file_path=row[2],
                    parsed_file_path=row[3],
                    file_type=row[4],
                    file_size_bytes=row[5],
                    checksum=row[6],
                    status=row[7],
                    created_at=row[8],
                )
                for row in rows
            }

    def query_chunk_ids_by_metadata(
        self,
        document_id: UUID | None = None,
        min_chunk_index: int | None = None,
        max_chunk_index: int | None = None,
        chunk_index: int | None = None,
    ) -> list[UUID]:
        """
        Queries matching child chunk UUIDs filtered by document_id and chunk_index constraints.
        Used for metadata filtering and generating TurboVec integer allowlists.
        """
        clauses: list[sql.Composable] = []
        params: list[Any] = []

        if document_id is not None:
            clauses.append(sql.SQL("document_id = %s"))
            params.append(document_id)

        if chunk_index is not None:
            clauses.append(sql.SQL("chunk_index = %s"))
            params.append(chunk_index)
        else:
            if min_chunk_index is not None:
                clauses.append(sql.SQL("chunk_index >= %s"))
                params.append(min_chunk_index)
            if max_chunk_index is not None:
                clauses.append(sql.SQL("chunk_index <= %s"))
                params.append(max_chunk_index)

        where_part = sql.SQL("WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")

        query = sql.SQL("SELECT chunk_id FROM chunks {where} ORDER BY chunk_index ASC;").format(
            where=where_part
        )

        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [cast(UUID, row[0]) for row in rows]

    def insert_document(self, doc: Document) -> Document:
        """
        Inserts a new document record.
        Returns the inserted Document with populated created_at timestamp.
        """
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    document_id, filename, raw_file_path, parsed_file_path,
                    file_type, file_size_bytes, checksum, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING created_at;
                """,
                (
                    doc.document_id,
                    doc.filename,
                    doc.raw_file_path,
                    doc.parsed_file_path,
                    doc.file_type,
                    doc.file_size_bytes,
                    doc.checksum,
                    doc.status,
                ),
            )
            row = cur.fetchone()
            if row is not None:
                doc.created_at = row[0]
        return doc

    def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        parsed_file_path: str | None = None,
    ) -> None:
        """Updates document status and optionally the parsed markdown file path."""
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if parsed_file_path is not None:
                cur.execute(
                    """
                    UPDATE documents
                    SET status = %s, parsed_file_path = %s
                    WHERE document_id = %s;
                    """,
                    (status, parsed_file_path, document_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE documents
                    SET status = %s
                    WHERE document_id = %s;
                    """,
                    (status, document_id),
                )

    def delete_document(self, document_id: UUID) -> bool:
        """
        Deletes a document from the database.
        Relies on ON DELETE CASCADE to automatically purge all associated
        parent chunks, child chunks, and processing jobs.
        """
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE document_id = %s;",
                (document_id,),
            )
            return cur.rowcount > 0

    def create_processing_job(
        self,
        document_id: UUID,
        stage: JobStage = "queued",
    ) -> ProcessingJob:
        """Creates an initial processing job record."""
        job = ProcessingJob(document_id=document_id, stage=stage)
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processing_jobs (
                    job_id, document_id, stage, error_message
                )
                VALUES (%s, %s, %s, %s)
                RETURNING started_at;
                """,
                (job.job_id, job.document_id, job.stage, job.error_message),
            )
            row = cur.fetchone()
            if row is not None:
                job.started_at = row[0]
        return job

    def update_job_stage(
        self,
        job_id: UUID,
        stage: JobStage,
        error_message: str | None = None,
        completed: bool = False,
    ) -> None:
        """Updates processing job stage, error messages, and completion timestamp."""
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if completed or stage in ("completed", "failed"):
                cur.execute(
                    """
                    UPDATE processing_jobs
                    SET stage = %s, error_message = %s, completed_at = NOW()
                    WHERE job_id = %s;
                    """,
                    (stage, error_message, job_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE processing_jobs
                    SET stage = %s, error_message = %s
                    WHERE job_id = %s;
                    """,
                    (stage, error_message, job_id),
                )

    def get_job(self, job_id: UUID) -> ProcessingJob | None:
        """Retrieves a processing job record by its job_id."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, document_id, stage, error_message, started_at, completed_at
                FROM processing_jobs
                WHERE job_id = %s;
                """,
                (job_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return ProcessingJob(
                job_id=row[0],
                document_id=row[1],
                stage=row[2],
                error_message=row[3],
                started_at=row[4],
                completed_at=row[5],
            )

    def get_jobs_for_document(self, document_id: UUID) -> list[ProcessingJob]:
        """Retrieves all processing jobs for a specific document."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, document_id, stage, error_message, started_at, completed_at
                FROM processing_jobs
                WHERE document_id = %s
                ORDER BY started_at ASC;
                """,
                (document_id,),
            )
            rows = cur.fetchall()
            return [
                ProcessingJob(
                    job_id=row[0],
                    document_id=row[1],
                    stage=row[2],
                    error_message=row[3],
                    started_at=row[4],
                    completed_at=row[5],
                )
                for row in rows
            ]

    def persist_chunks_atomically(
        self,
        document_id: UUID,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
        job_id: UUID | None = None,
        simulate_failure_at_chunk: int | None = None,
    ) -> None:
        """
        Persists parent chunks and child chunks in a single atomic database transaction.
        If any chunk insertion fails, the entire transaction rolls back completely to prevent
        orphaned records.

        Parameters:
            document_id: UUID of the source document.
            parent_chunks: Ordered list of ParentChunk entities.
            child_chunks: Ordered list of ChildChunk entities with parent_chunk_id pointers.
            job_id: Optional processing job ID to update to 'chunking' stage upon commit.
            simulate_failure_at_chunk: Optional index to simulate an error for chaos testing.
        """
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            for parent in parent_chunks:
                cur.execute(
                    """
                    INSERT INTO parent_chunks (
                        parent_chunk_id, document_id, content, parent_index, heading_path
                    )
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (
                        parent.parent_chunk_id,
                        parent.document_id,
                        parent.content,
                        parent.parent_index,
                        parent.heading_path,
                    ),
                )

            for i, child in enumerate(child_chunks):
                if simulate_failure_at_chunk is not None and i == simulate_failure_at_chunk:
                    raise RuntimeError(
                        f"Simulated failure at chunk index {i} during atomic persistence"
                    )

                cur.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, document_id, parent_chunk_id, content, chunk_index, heading_path
                    )
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        child.chunk_id,
                        child.document_id,
                        child.parent_chunk_id,
                        child.content,
                        child.chunk_index,
                        child.heading_path,
                    ),
                )

            if job_id is not None:
                cur.execute(
                    """
                    UPDATE processing_jobs
                    SET stage = 'chunking'
                    WHERE job_id = %s;
                    """,
                    (job_id,),
                )

        logger.info(
            "Persisted chunks atomically",
            document_id=str(document_id),
            parent_count=len(parent_chunks),
            child_count=len(child_chunks),
        )

    def get_parent_chunks(self, document_id: UUID) -> list[ParentChunk]:
        """Retrieves all parent chunks for a document, ordered by parent_index."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT parent_chunk_id, document_id, content, parent_index, heading_path
                FROM parent_chunks
                WHERE document_id = %s
                ORDER BY parent_index ASC;
                """,
                (document_id,),
            )
            rows = cur.fetchall()
            return [
                ParentChunk(
                    parent_chunk_id=row[0],
                    document_id=row[1],
                    content=row[2],
                    parent_index=row[3],
                    heading_path=cast(list[str], row[4] or []),
                )
                for row in rows
            ]

    def get_child_chunks(self, document_id: UUID) -> list[ChildChunk]:
        """Retrieves all child chunks for a document, ordered by chunk_index."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, parent_chunk_id, content, chunk_index, heading_path
                FROM chunks
                WHERE document_id = %s
                ORDER BY chunk_index ASC;
                """,
                (document_id,),
            )
            rows = cur.fetchall()
            return [
                ChildChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    parent_chunk_id=row[2],
                    content=row[3],
                    chunk_index=row[4],
                    heading_path=cast(list[str], row[5] or []),
                )
                for row in rows
            ]

    def get_all_child_chunks(self) -> list[ChildChunk]:
        """Retrieves all child chunks across all documents, ordered by document_id and chunk_index."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, parent_chunk_id, content, chunk_index, heading_path
                FROM chunks
                ORDER BY document_id ASC, chunk_index ASC;
                """
            )
            rows = cur.fetchall()
            return [
                ChildChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    parent_chunk_id=row[2],
                    content=row[3],
                    chunk_index=row[4],
                    heading_path=cast(list[str], row[5] or []),
                )
                for row in rows
            ]

    def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> dict[UUID, ChildChunk]:
        """Retrieves child chunks matching a list of chunk UUIDs."""
        if not chunk_ids:
            return {}
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, parent_chunk_id, content, chunk_index, heading_path
                FROM chunks
                WHERE chunk_id = ANY(%s);
                """,
                (chunk_ids,),
            )
            rows = cur.fetchall()
            return {
                row[0]: ChildChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    parent_chunk_id=row[2],
                    content=row[3],
                    chunk_index=row[4],
                    heading_path=cast(list[str], row[5] or []),
                )
                for row in rows
            }

    def get_parent_chunks_by_ids(self, parent_chunk_ids: list[UUID]) -> dict[UUID, ParentChunk]:
        """Retrieves parent chunks matching a list of parent chunk UUIDs."""
        if not parent_chunk_ids:
            return {}
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT parent_chunk_id, document_id, content, parent_index, heading_path
                FROM parent_chunks
                WHERE parent_chunk_id = ANY(%s);
                """,
                (parent_chunk_ids,),
            )
            rows = cur.fetchall()
            return {
                row[0]: ParentChunk(
                    parent_chunk_id=row[0],
                    document_id=row[1],
                    content=row[2],
                    parent_index=row[3],
                    heading_path=cast(list[str], row[4] or []),
                )
                for row in rows
            }

    def delete_parent_chunk(self, parent_chunk_id: UUID) -> bool:
        """
        Deletes a parent chunk from the database.
        Cascades automatically to delete all child chunks linked via parent_chunk_id.
        """
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "DELETE FROM parent_chunks WHERE parent_chunk_id = %s;",
                (parent_chunk_id,),
            )
            return cur.rowcount > 0
