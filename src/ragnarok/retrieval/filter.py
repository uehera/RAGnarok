"""
Metadata Filtering Engine Subsystem.

Extracts explicit metadata constraints (filename, document ID, chunk index / sequence range)
from user queries, queries PostgreSQL to resolve matching child chunk records,
and translates matching records into 64-bit TurboVec integer allowlists.
Returns an immediate empty candidate set if zero chunks match constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

import structlog

from ragnarok.indexing.dense import VectorIdMap
from ragnarok.storage.repository import RelationalRepository

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MetadataConstraint:
    """Explicit metadata constraints identified within a user query."""

    filename: str | None = None
    document_id: UUID | None = None
    chunk_index: int | None = None
    min_chunk_index: int | None = None
    max_chunk_index: int | None = None

    @property
    def has_constraints(self) -> bool:
        """Returns True if at least one metadata constraint is present."""
        return (
            self.filename is not None
            or self.document_id is not None
            or self.chunk_index is not None
            or self.min_chunk_index is not None
            or self.max_chunk_index is not None
        )


# Extraction regex patterns
_FILENAME_EXPLICIT_PATTERN = re.compile(
    r"(?:filename|file)\s*[:=]\s*[\"']?([\w\-.]+\.[\w]+)[\"']?",
    re.IGNORECASE,
)
_IN_FROM_FILE_PATTERN = re.compile(
    r"(?:in|from)(?:\s+file)?\s+[\"']?([\w\-.]+\.(?:md|py|toml|json|sql|pdf|txt|csv|yaml|yml|log|tvim|csc))[\"']?",
    re.IGNORECASE,
)
_DOC_ID_PATTERN = re.compile(
    r"(?:document_id|doc_id)\s*[:=]\s*([0-9a-fA-F-]{36})|(?:document|doc)\s+([0-9a-fA-F-]{36})",
    re.IGNORECASE,
)
_CHUNK_RANGE_PATTERN = re.compile(
    r"\bchunks?\s*(?:[:=]|\s)\s*(\d+)\s*(?:-|to)\s*(\d+)|\bchunk\s+range\s+(\d+)\s+to\s+(\d+)",
    re.IGNORECASE,
)
_SINGLE_CHUNK_PATTERN = re.compile(
    r"\bchunk(?:_index)?\s*(?:[:=]|\s)\s*(\d+)\b",
    re.IGNORECASE,
)


def extract_constraints(query: str) -> MetadataConstraint:
    """
    Extracts explicit metadata constraints from a natural-language or structured query.
    """
    filename: str | None = None
    document_id: UUID | None = None
    chunk_index: int | None = None
    min_chunk_index: int | None = None
    max_chunk_index: int | None = None

    # 1. Filename extraction
    file_match = _FILENAME_EXPLICIT_PATTERN.search(query)
    if file_match:
        filename = file_match.group(1).strip()
    else:
        in_file_match = _IN_FROM_FILE_PATTERN.search(query)
        if in_file_match:
            filename = in_file_match.group(1).strip()

    # 2. Document ID extraction
    doc_match = _DOC_ID_PATTERN.search(query)
    if doc_match:
        raw_uuid = doc_match.group(1) or doc_match.group(2)
        try:
            document_id = UUID(raw_uuid.strip())
        except ValueError:
            document_id = None

    # 3. Chunk range extraction (checked before single chunk)
    range_match = _CHUNK_RANGE_PATTERN.search(query)
    if range_match:
        start_str = range_match.group(1) or range_match.group(3)
        end_str = range_match.group(2) or range_match.group(4)
        if start_str and end_str:
            min_chunk_index = int(start_str)
            max_chunk_index = int(end_str)
    else:
        # Single chunk extraction
        single_match = _SINGLE_CHUNK_PATTERN.search(query)
        if single_match:
            chunk_str = single_match.group(1) or single_match.group(2)
            if chunk_str:
                chunk_index = int(chunk_str)

    return MetadataConstraint(
        filename=filename,
        document_id=document_id,
        chunk_index=chunk_index,
        min_chunk_index=min_chunk_index,
        max_chunk_index=max_chunk_index,
    )


class MetadataFilterEngine:
    """
    Resolves natural-language metadata constraints against PostgreSQL
    and translates matching records into TurboVec 64-bit integer allowlists.
    """

    def __init__(
        self,
        repository: RelationalRepository,
        id_map: VectorIdMap,
    ) -> None:
        self.repository = repository
        self.id_map = id_map

    def resolve_chunk_uuids(self, constraints: MetadataConstraint) -> list[UUID]:
        """
        Queries PostgreSQL for child chunk UUIDs satisfying the given metadata constraints.
        Returns empty list if constraints are specified but zero records match.
        """
        if not constraints.has_constraints:
            return []

        doc_id = constraints.document_id

        # If filename specified, resolve to document_id first
        if constraints.filename is not None:
            doc = self.repository.get_document_by_filename(constraints.filename)
            if doc is None:
                logger.info(
                    "Metadata constraint filename not found in documents",
                    filename=constraints.filename,
                )
                return []
            doc_id = doc.document_id

        chunk_uuids = self.repository.query_chunk_ids_by_metadata(
            document_id=doc_id,
            min_chunk_index=constraints.min_chunk_index,
            max_chunk_index=constraints.max_chunk_index,
            chunk_index=constraints.chunk_index,
        )

        logger.info(
            "Resolved metadata constraints in PostgreSQL",
            filename=constraints.filename,
            document_id=str(doc_id) if doc_id else None,
            matching_chunks=len(chunk_uuids),
        )
        return chunk_uuids

    def resolve_integer_allowlist(
        self,
        constraints: MetadataConstraint,
    ) -> list[int] | None:
        """
        Translates matching child chunk UUIDs into TurboVec 64-bit uint64 integer handles.
        Returns:
        - None if no constraints were specified (full index eligible)
        - list[int] containing uint64 handles (which is [] if constraints were specified but 0 matched)
        """
        if not constraints.has_constraints:
            return None

        matching_uuids = self.resolve_chunk_uuids(constraints)
        if not matching_uuids:
            return []

        handles: list[int] = []
        for uid in matching_uuids:
            handle = self.id_map.get_handle(uid)
            if handle is not None:
                handles.append(handle)

        logger.info(
            "Mapped chunk UUIDs to TurboVec integer allowlist",
            uuid_count=len(matching_uuids),
            handle_count=len(handles),
        )
        return handles

    def filter_query(
        self,
        query: str,
    ) -> tuple[MetadataConstraint, list[int] | None]:
        """
        Convenience method: extracts constraints from query and resolves the integer allowlist.
        """
        constraints = extract_constraints(query)
        allowlist = self.resolve_integer_allowlist(constraints)
        return constraints, allowlist
