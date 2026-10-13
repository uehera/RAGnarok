"""
Citation and Footnote Resolution Subsystem.

Parses inline bracketed citations ([1], [2], [3]) from synthesized text,
resolves them 1:1 against the Context Management provenance lookup table,
eliminates ghost citations referencing nonexistent sources, and formats
structured CLI footnotes.
Adheres strictly to dev/workspace/generation_citations.md.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

import structlog

from ragnarok.generation.contract import CitationFootnote

logger = structlog.get_logger(__name__)

# Matches [1], [2], [12], etc.
_CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citation_indices(text: str) -> list[int]:
    """
    Extracts ordered, deduplicated 1-based citation indices from generated text.
    Preserves the order in which citations first appear.
    """
    if not text:
        return []

    seen: set[int] = set()
    ordered_indices: list[int] = []

    for match in _CITATION_MARKER_PATTERN.finditer(text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            ordered_indices.append(idx)

    return ordered_indices


def _lookup_provenance_entry(
    citation_index: int, provenance_map: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Flexible lookup checking multiple key formats in provenance_map:
    '[Source 1]', 'Source 1', '1', or integer 1.
    """
    candidates = [
        f"[Source {citation_index}]",
        f"Source {citation_index}",
        str(citation_index),
        citation_index,
        f"[source {citation_index}]",
    ]
    for key in candidates:
        if key in provenance_map:
            val = provenance_map[key]
            if isinstance(val, dict):
                return val
    return None


def resolve_citations(
    text: str, provenance_map: dict[str, Any]
) -> tuple[list[CitationFootnote], list[int]]:
    """
    Resolves inline bracketed citations against the context provenance table.

    Parameters:
        text: Synthesized answer text containing [1], [2] markers.
        provenance_map: Dictionary mapping source labels to document/chunk metadata.

    Returns:
        A tuple of:
        - valid_citations: List of verified CitationFootnote objects.
        - ghost_citations: List of integer indices that could not be resolved.
    """
    indices = extract_citation_indices(text)
    if not indices:
        return [], []

    valid_citations: list[CitationFootnote] = []
    ghost_citations: list[int] = []

    for idx in indices:
        record = _lookup_provenance_entry(idx, provenance_map)
        if record is None:
            logger.warning(
                "Ghost citation detected; marker does not exist in context provenance",
                citation_index=idx,
            )
            ghost_citations.append(idx)
            continue

        raw_doc_id = record.get("document_id")
        doc_uuid = UUID(str(raw_doc_id)) if raw_doc_id else None

        raw_parent_id = record.get("parent_chunk_id")
        parent_uuid = UUID(str(raw_parent_id)) if raw_parent_id else None

        filename = str(record.get("filename", f"document_{idx}"))
        heading_path = list(record.get("heading_path", []))

        footnote = CitationFootnote(
            citation_index=idx,
            document_id=doc_uuid,
            filename=filename,
            parent_chunk_id=parent_uuid,
            heading_path=heading_path,
        )
        valid_citations.append(footnote)

    logger.info(
        "Resolved citations",
        total_extracted=len(indices),
        valid_citations=len(valid_citations),
        ghost_citations=len(ghost_citations),
    )
    return valid_citations, ghost_citations


def format_footnotes_table(citations: list[CitationFootnote]) -> str:
    """
    Formats structured footnotes table matching Developer CLI layout specification:

    --------------------------------------------------------------------------------
    Sources & Citations
    --------------------------------------------------------------------------------
    [1] filename.pdf
        Section: Retrieval Mechanics > Child Indexing
        Parent Chunk: c3000000-0000-4000-8000-000000000001
    """
    if not citations:
        return ""

    lines = [
        "--------------------------------------------------------------------------------",
        "Sources & Citations",
        "--------------------------------------------------------------------------------",
    ]

    for citation in citations:
        lines.append(citation.format_footnote_text())
        lines.append("")  # Blank spacer line

    return "\n".join(lines).rstrip()
