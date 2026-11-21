"""
In-Flight (Post-Retrieval) Context Security Screening Subsystem.

Defends against Indirect Prompt Injections (IPI), trojan payloads, delimiter escapes,
and adversarial context manipulation that may exist in retrieved candidate passages
(e.g., pre-existing legacy data, multi-tenant databases, or fragmented injection payloads).

Screens retrieved parent context blocks after cross-encoder reranking and prior to prompt assembly.
Malicious evidence chunks are surgically pruned or defanged, ensuring only verified evidence
reaches the LLM generation stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import structlog

if TYPE_CHECKING:
    from ragnarok.context.contract import ExpandedParentContext

from ragnarok.core.config import GuardrailsConfig
from ragnarok.guardrails.document_guard import _DOCUMENT_INJECTION_PATTERNS

logger = structlog.get_logger(__name__)


@dataclass
class ContextScreeningResult:
    """Detailed audit report of an in-flight context screening pass."""

    total_candidates: int
    safe_candidates: int
    dropped_candidates: int
    threats_detected: list[str] = field(default_factory=list)
    dropped_parent_ids: list[UUID] = field(default_factory=list)
    threat_details: list[dict[str, str]] = field(default_factory=list)

    @property
    def has_threats(self) -> bool:
        """Returns True if any injection or adversarial patterns were detected."""
        return len(self.threats_detected) > 0

    @property
    def all_dropped(self) -> bool:
        """Returns True if all candidate evidence blocks were dropped due to threats."""
        return self.total_candidates > 0 and self.safe_candidates == 0


class ContextSecurityGuard:
    """
    Runtime context screening guard evaluating retrieved passages before prompt assembly.
    """

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
        action: Literal["drop", "quarantine", "sanitize"] = "drop",
    ) -> None:
        self.config = config or GuardrailsConfig()
        self.action = action

    def screen_parents(
        self,
        parents: list[ExpandedParentContext],
    ) -> tuple[list[ExpandedParentContext], ContextScreeningResult]:
        """
        Screens a list of expanded parent context sections retrieved from storage.

        Parameters:
            parents: Ordered list of ExpandedParentContext candidates.

        Returns:
            Tuple of:
                - Surviving safe ExpandedParentContext items (with malicious chunks dropped or defanged)
                - ContextScreeningResult detailing detections, actions taken, and dropped chunk IDs.
        """
        if not parents:
            return [], ContextScreeningResult(
                total_candidates=0,
                safe_candidates=0,
                dropped_candidates=0,
            )

        safe_parents: list[ExpandedParentContext] = []
        dropped_ids: list[UUID] = []
        all_threats: list[str] = []
        threat_details: list[dict[str, str]] = []

        for parent in parents:
            content = parent.content
            detected_tags: list[str] = []
            threat_snippets: list[str] = []
            sanitized_content = content

            # Evaluate each injection pattern against parent text
            for tag, pattern in _DOCUMENT_INJECTION_PATTERNS:
                matches = list(pattern.finditer(sanitized_content))
                if matches:
                    detected_tags.append(tag)
                    for m in matches:
                        threat_snippets.append(m.group(0)[:120])
                    if self.action == "sanitize":
                        sanitized_content = pattern.sub(
                            f"[DEFANGED INJECTION DIRECTIVE: {tag}]", sanitized_content
                        )

            if detected_tags:
                all_threats.extend(detected_tags)
                for tag, snippet in zip(detected_tags, threat_snippets, strict=False):
                    threat_details.append({
                        "parent_chunk_id": str(parent.parent_chunk_id),
                        "filename": parent.filename,
                        "tag": tag,
                        "snippet": snippet,
                    })

                logger.warning(
                    "Context Guard: Threat detected in retrieved parent context",
                    parent_chunk_id=str(parent.parent_chunk_id),
                    filename=parent.filename,
                    tags=detected_tags,
                    action=self.action,
                )

                if self.action in ("drop", "quarantine"):
                    # Prune the malicious evidence chunk from surviving candidate pool
                    dropped_ids.append(parent.parent_chunk_id)
                    continue

                if self.action == "sanitize":
                    # Update parent content in place with defanged version
                    parent.content = sanitized_content
                    safe_parents.append(parent)
            else:
                safe_parents.append(parent)

        unique_threats = list(dict.fromkeys(all_threats))
        result = ContextScreeningResult(
            total_candidates=len(parents),
            safe_candidates=len(safe_parents),
            dropped_candidates=len(dropped_ids),
            threats_detected=unique_threats,
            dropped_parent_ids=dropped_ids,
            threat_details=threat_details,
        )

        if result.has_threats:
            logger.info(
                "Context Guard: In-flight screening completed",
                total=result.total_candidates,
                safe=result.safe_candidates,
                dropped=result.dropped_candidates,
                threats=unique_threats,
            )

        return safe_parents, result
