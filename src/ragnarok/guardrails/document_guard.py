"""
Native Document Security Guard Subsystem.

Defends against Indirect Prompt Injections (Trojan Horse attacks) embedded inside
ingested documents (PDFs, Markdown, text). Scans document content for instruction-like
imperative directives, persona hijacking, system overrides, special tokens, and hidden
HTML comments attempting to steer downstream generation.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from ragnarok.core.config import GuardrailsConfig
from ragnarok.guardrails.contract import (
    DocumentGuardDecision,
    DocumentSecurityError,
)

logger = structlog.get_logger(__name__)

# Injection patterns for document content scanning
_DOCUMENT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "IGNORE_INSTRUCTIONS",
        re.compile(r"(?i)(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions"),
    ),
    (
        "PERSONA_HIJACKING",
        re.compile(r"(?i)(?:you\s+are\s+now\s+|act\s+as\s+(?:a\s+)?|dan\s+mode|developer\s+mode)"),
    ),
    (
        "SYSTEM_OVERRIDE",
        re.compile(r"(?i)(?:system\s*prompt|system\s+override)"),
    ),
    (
        "ROLE_HEADER_INJECTION",
        re.compile(r"(?i)\n\s*\n\s*(?:system|user|assistant)\s*:"),
    ),
    (
        "SPECIAL_TOKEN_INJECTION",
        re.compile(r"(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\])"),
    ),
    (
        "HIDDEN_HTML_INJECTION",
        re.compile(r"<!--\s*(?:system|prompt|override|instruction|ignore).*?-->", re.IGNORECASE | re.DOTALL),
    ),
]


def scan_and_defang_document(
    content: str,
    action: str = "sanitize",
    quarantine_dir: Path | str | None = None,
    source_name: str | None = None,
) -> DocumentGuardDecision:
    """
    Scans document text for indirect prompt injection directives.

    Parameters:
        content: Raw parsed document text / markdown.
        action: 'sanitize' (defangs directives into harmless tags),
                'quarantine' (isolates file and metadata to quarantine_dir without raising),
                or 'reject' (quarantines if dir configured, and raises DocumentSecurityError).
        quarantine_dir: Target directory for isolating quarantined files.
        source_name: Optional original file name / path for metadata audit.

    Returns:
        DocumentGuardDecision detailing whether content was clean, what was detected, defanged text,
        and quarantine status.
    """
    if not content:
        return DocumentGuardDecision(
            is_safe=True,
            defanged_content="",
            injections_detected=[],
            quarantined=False,
        )

    detected: list[str] = []
    threat_excerpts: list[str] = []
    sanitized = content

    for tag, pattern in _DOCUMENT_INJECTION_PATTERNS:
        matches = list(pattern.finditer(sanitized))
        if matches:
            detected.append(tag)
            for m in matches:
                threat_excerpts.append(m.group(0)[:120])
            if action == "sanitize":
                sanitized = pattern.sub(f"[DEFANGED INJECTION DIRECTIVE: {tag}]", sanitized)

    is_safe = len(detected) == 0

    if is_safe:
        return DocumentGuardDecision(
            is_safe=True,
            defanged_content=content,
            injections_detected=[],
            quarantined=False,
        )

    # Threat detected! Handle quarantine and rejection policies
    quarantine_saved_path: str | None = None
    if action in ("quarantine", "reject"):
        if quarantine_dir is not None:
            try:
                target_dir = Path(quarantine_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                clean_stem = Path(source_name).stem if source_name else "isolated_doc"
                clean_suffix = (
                    Path(source_name).suffix if source_name and Path(source_name).suffix else ".md"
                )
                quarantine_id = str(uuid4())
                file_name = f"{timestamp_str}_{clean_stem}_{quarantine_id[:8]}{clean_suffix}"
                meta_name = f"{timestamp_str}_{clean_stem}_{quarantine_id[:8]}.json"

                raw_quarantine_path = target_dir / file_name
                meta_quarantine_path = target_dir / meta_name

                raw_quarantine_path.write_text(content, encoding="utf-8")

                sha256_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                metadata = {
                    "quarantine_id": quarantine_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_name": source_name or "unknown",
                    "sha256": sha256_hash,
                    "file_size_bytes": len(content.encode("utf-8")),
                    "injections_detected": detected,
                    "threat_excerpts": threat_excerpts,
                    "action_taken": action,
                    "status": "QUARANTINED",
                    "quarantine_file": file_name,
                }
                meta_quarantine_path.write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )
                quarantine_saved_path = str(raw_quarantine_path)
                logger.warning(
                    "Document Guard: Malicious document isolated to quarantine",
                    source=source_name or "unknown",
                    tags=detected,
                    quarantine_file=str(raw_quarantine_path),
                )
            except Exception as q_err:
                logger.error(
                    "Document Guard: Failed to isolate file to quarantine directory",
                    error=str(q_err),
                )

        if action == "reject":
            first_match = threat_excerpts[0] if threat_excerpts else "directive"
            logger.error(
                "Document Guard: Forbidden indirect prompt injection detected in reject mode",
                tags=detected,
                match=first_match[:50],
                quarantine_path=quarantine_saved_path,
            )
            raise DocumentSecurityError(
                f"Document contains forbidden prompt injection pattern ({detected[0]}): {first_match[:50]}"
            )

        # action == "quarantine"
        return DocumentGuardDecision(
            is_safe=False,
            defanged_content="",
            injections_detected=detected,
            quarantined=True,
            quarantine_path=quarantine_saved_path,
            threat_excerpts=threat_excerpts,
        )

    # action == "sanitize"
    logger.warning(
        "Document Guard: Indirect prompt injection directives defanged",
        tags=detected,
        count=len(detected),
    )
    return DocumentGuardDecision(
        is_safe=False,
        defanged_content=sanitized,
        injections_detected=detected,
        quarantined=False,
        threat_excerpts=threat_excerpts,
    )


class DocumentSecurityGuard:
    """
    Ingestion-time safety guard defending against malicious documents.
    """

    def __init__(self, config: GuardrailsConfig | None = None) -> None:
        self.config = config or GuardrailsConfig()

    def validate(
        self,
        content: str,
        source_name: str | None = None,
    ) -> DocumentGuardDecision:
        """
        Validates, defangs, or quarantines document content according to active configuration.
        """
        if not self.config.enabled or not self.config.enable_document_guard:
            return DocumentGuardDecision(
                is_safe=True,
                defanged_content=content,
                injections_detected=[],
                quarantined=False,
            )

        return scan_and_defang_document(
            content=content,
            action=self.config.document_injection_action,
            quarantine_dir=self.config.quarantine_directory,
            source_name=source_name,
        )
