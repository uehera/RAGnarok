"""
Document Ingestion & Staging Subsystem.

Implements streaming SHA-256 calculation, duplicate rejection,
raw file archiving, and initial processing job creation in PostgreSQL.
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from pathlib import Path
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel

from ragnarok.core.models import Document, ProcessingJob
from ragnarok.storage.repository import RelationalRepository

logger = structlog.get_logger(__name__)


class DuplicateDocumentError(ValueError):
    """Raised when an incoming file matches an existing document's SHA-256 checksum."""

    def __init__(self, checksum: str, existing_document_id: UUID) -> None:
        super().__init__(
            f"Duplicate document rejected. Checksum {checksum} exists in document {existing_document_id}"
        )
        self.checksum = checksum
        self.existing_document_id = existing_document_id


class StagingResult(BaseModel):
    """Result of staging and validating an incoming document."""

    document: Document | None = None
    job: ProcessingJob | None = None
    is_duplicate: bool = False
    existing_document_id: UUID | None = None


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Pure procedural function that computes cryptographic SHA-256 hash
    across a file byte stream in memory-efficient chunks.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_file_type(path: Path) -> str:
    """Detects MIME type from file extension with fallback."""
    ext = path.suffix.lower()
    if ext == ".md":
        return "text/markdown"
    if ext == ".txt":
        return "text/plain"
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def validate_staging_file(
    file_path: Path,
    supported_formats: list[str] | None = None,
) -> tuple[str, int]:
    """
    Validates that a file exists, is non-empty, and has a supported format.

    Returns:
        tuple[str, int]: (detected_file_type, file_size_bytes)

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or has an unsupported format.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    size = file_path.stat().st_size
    if size == 0:
        raise ValueError(f"File is empty (0 bytes): {file_path}")

    if supported_formats is not None:
        ext = file_path.suffix.lower()
        normalized_formats = [f.lower() for f in supported_formats]
        if ext not in normalized_formats:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported formats: {supported_formats}"
            )

    file_type = detect_file_type(file_path)
    return file_type, size


class StagingService:
    """
    Stateful service coordinating document receipt, validation,
    SHA-256 deduplication, filesystem storage, and PostgreSQL tracking.
    """

    def __init__(
        self,
        repository: RelationalRepository,
        staging_dir: Path | str = "data/staging",
        raw_dir: Path | str = "data/raw",
        parsed_dir: Path | str = "data/parsed",
        supported_formats: list[str] | None = None,
    ) -> None:
        self.repository = repository
        self.staging_dir = Path(staging_dir)
        self.raw_dir = Path(raw_dir)
        self.parsed_dir = Path(parsed_dir)
        self.supported_formats = (
            supported_formats if supported_formats is not None else [".pdf", ".docx", ".md", ".txt"]
        )

        # Ensure directories exist
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    def stage_file(self, source_path: Path | str) -> Path:
        """
        Receives an incoming document by copying it into the staging directory.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found at: {source}")

        dest = self.staging_dir / source.name
        shutil.copy2(source, dest)
        return dest

    def process_file(
        self,
        file_path: Path | str,
        raise_on_duplicate: bool = False,
    ) -> StagingResult:
        """
        Executes Stage 1 ingestion:
        1. Validates file existence, non-emptiness, and format.
        2. Computes streaming SHA-256 checksum.
        3. Checks deduplication in PostgreSQL.
        4. If duplicate: purges staging file, logs warning, returns StagingResult(is_duplicate=True).
        5. If unique: moves to data/raw/<document_id>.<ext>, inserts document and job records.
        """
        path = Path(file_path)
        # Validate file
        file_type, file_size = validate_staging_file(path, self.supported_formats)

        # Compute SHA-256
        checksum = compute_sha256(path)

        # Check existing document in PostgreSQL
        existing = self.repository.get_document_by_checksum(checksum)
        if existing is not None:
            logger.warning(
                "Duplicate file rejected",
                filename=path.name,
                checksum=checksum,
                existing_document_id=str(existing.document_id),
            )
            # Purge the staging file if it resides in staging directory
            try:
                if path.resolve().parent == self.staging_dir.resolve():
                    path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to remove staging file", error=str(e))

            if raise_on_duplicate:
                raise DuplicateDocumentError(checksum, existing.document_id)

            return StagingResult(
                is_duplicate=True,
                existing_document_id=existing.document_id,
            )

        # Unique file: assign new UUID
        document_id = uuid4()
        suffix = path.suffix.lower()
        raw_target_path = self.raw_dir / f"{document_id}{suffix}"
        parsed_target_path = self.parsed_dir / f"{document_id}.md"

        # Move file permanently to data/raw/<document_id>.<ext>
        shutil.move(str(path), str(raw_target_path))

        # Register document in PostgreSQL
        doc = Document(
            document_id=document_id,
            filename=path.name,
            raw_file_path=str(raw_target_path),
            parsed_file_path=str(parsed_target_path),
            file_type=file_type,
            file_size_bytes=file_size,
            checksum=checksum,
            status="stored",
        )
        doc = self.repository.insert_document(doc)

        # Create initial processing job with stage 'queued'
        job = self.repository.create_processing_job(document_id=doc.document_id, stage="queued")

        logger.info(
            "Document successfully staged and stored",
            document_id=str(doc.document_id),
            filename=doc.filename,
            raw_path=str(raw_target_path),
            checksum=checksum,
            job_id=str(job.job_id),
        )

        return StagingResult(
            document=doc,
            job=job,
            is_duplicate=False,
        )
