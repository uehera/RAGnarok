"""
Core Domain Models and Entities.

Provides strongly typed Pydantic models for documents, parent chunks,
child chunks, and processing job lifecycles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

DocumentStatus = Literal["stored", "parsing", "parsed", "failed"]
JobStage = Literal["queued", "parsing", "chunking", "indexing", "completed", "failed"]


class Document(BaseModel):
    """Represents a source document registered in the system."""

    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    raw_file_path: str
    parsed_file_path: str
    file_type: str
    file_size_bytes: int
    checksum: str
    status: DocumentStatus = "stored"
    created_at: datetime | None = None


class ParentChunk(BaseModel):
    """
    Represents a broader contextual container produced by heading-based splitting.
    Stored exclusively in PostgreSQL; never indexed into BM25s or TurboVec.
    """

    parent_chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    parent_index: int
    heading_path: list[str] = Field(default_factory=list)


class ChildChunk(BaseModel):
    """
    Represents a discrete retrieval unit subdivided from a parent chunk.
    Referenced by dense and lexical indexes, linked back to its parent container.
    """

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    parent_chunk_id: UUID
    content: str
    chunk_index: int
    heading_path: list[str] = Field(default_factory=list)


# Alias for ChildChunk matching table name
Chunk = ChildChunk


class ProcessingJob(BaseModel):
    """Tracks asynchronous ingestion, parsing, chunking, and indexing lifecycles."""

    job_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    stage: JobStage = "queued"
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
