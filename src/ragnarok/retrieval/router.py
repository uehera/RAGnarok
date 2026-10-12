"""
Adaptive Retrieval Router Subsystem.

Applies deterministic rule-based signal analysis to query tokens to route
queries to the optimal search strategy:
- Exact identifiers, error codes, code symbols, filenames -> Lexical (BM25s)
- Broad conceptual, explanatory, or thematic questions -> Dense (TurboVec)
- Mixed or neutral signals -> Configured default strategy (typically Hybrid via RRF)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import structlog

from ragnarok.core.config import AdaptiveRetrievalConfig

logger = structlog.get_logger(__name__)


class RetrievalStrategy(StrEnum):
    """Retrieval strategies supported by RAGnarok."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class QuerySignals:
    """Detected lexical and semantic signals within a query."""

    has_exact_match: bool
    has_semantic: bool
    exact_signals: list[str]
    semantic_signals: list[str]


# Regex patterns for exact lexical tokens
_QUOTED_PATTERN = re.compile(r'["\']([^"\']+)["\']')
_FILENAME_PATTERN = re.compile(
    r"\b[\w\-.]+\.(?:py|md|toml|json|sql|pdf|txt|csv|yaml|yml|log|tvim|csc)\b",
    re.IGNORECASE,
)
_SNAKE_CASE_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b")
_CAMEL_CASE_PATTERN = re.compile(r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b")
_PASCAL_CASE_PATTERN = re.compile(r"\b[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*\b")
_CALL_OR_ACCESS_PATTERN = re.compile(
    r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\(\)|\.[a-zA-Z_][a-zA-Z0-9_]*)\b"
)
_ERROR_STATUS_PATTERN = re.compile(
    r"\b(?:HTTP\s*)?(?:4\d\d|5\d\d)\b|\b(?:ERR|ERROR|FATAL|FAIL)_[A-Z0-9_]+\b|\b[A-Z][a-zA-Z0-9]*(?:Error|Exception|Fault|Warning)\b"
)
_VERSION_OR_UUID_PATTERN = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)?\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_ACRONYM_TECHNICAL_PATTERN = re.compile(r"\b[A-Z]{2,}[0-9a-zA-Z]*\b")
_COMMON_ALL_CAPS = frozenset({"AND", "OR", "NOT", "THE", "IS", "IN", "AT", "FOR", "TO", "OF", "ON", "BY"})

# Semantic and conceptual indicators
_SEMANTIC_PATTERNS = [
    re.compile(r"\bwhy\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:does|do|can|should|is)\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bthematic\b", re.IGNORECASE),
    re.compile(r"\bconcept(?:ual)?\b", re.IGNORECASE),
    re.compile(r"\barchitecture\b", re.IGNORECASE),
    re.compile(r"\bpurpose\s+of\b", re.IGNORECASE),
    re.compile(r"\brelationship\s+between\b", re.IGNORECASE),
    re.compile(r"\bdifference\s+between\b", re.IGNORECASE),
    re.compile(r"\bsummar(?:y|ize)\b", re.IGNORECASE),
    re.compile(r"\btradeoffs?\b", re.IGNORECASE),
    re.compile(r"\bdesign\s+philosophy\b", re.IGNORECASE),
    re.compile(r"\bbest\s+practices\b", re.IGNORECASE),
    re.compile(r"\bunderstand\b", re.IGNORECASE),
]


class AdaptiveRouter:
    """
    Deterministic rule-based router evaluating query signals.
    """

    def __init__(self, config: AdaptiveRetrievalConfig | None = None) -> None:
        self.config = config if config is not None else AdaptiveRetrievalConfig()

    def analyze_signals(self, query: str) -> QuerySignals:
        """
        Extracts exact-match signals (identifiers, filenames, codes)
        and semantic signals (conceptual explanations, summaries) from the query text.
        """
        exact_tokens: list[str] = []

        # 1. Quoted phrases
        for m in _QUOTED_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))

        # 2. Filenames and extensions
        for m in _FILENAME_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))

        # 3. Code identifiers
        for m in _SNAKE_CASE_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))
        for m in _CAMEL_CASE_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))
        for m in _PASCAL_CASE_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))
        for m in _CALL_OR_ACCESS_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))

        # 4. Error codes and exceptions
        for m in _ERROR_STATUS_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))

        # 5. Versions and UUIDs
        for m in _VERSION_OR_UUID_PATTERN.finditer(query):
            exact_tokens.append(m.group(0))

        # 6. Acronyms and technical symbols (e.g. BM25s, SHA256, RRF)
        for m in _ACRONYM_TECHNICAL_PATTERN.finditer(query):
            tok = m.group(0)
            if tok not in _COMMON_ALL_CAPS:
                exact_tokens.append(tok)

        dedup_exact = list(dict.fromkeys(exact_tokens))

        # Semantic signals
        semantic_tokens: list[str] = []
        for pat in _SEMANTIC_PATTERNS:
            match = pat.search(query)
            if match:
                semantic_tokens.append(match.group(0))

        dedup_semantic = list(dict.fromkeys(semantic_tokens))

        return QuerySignals(
            has_exact_match=len(dedup_exact) > 0,
            has_semantic=len(dedup_semantic) > 0,
            exact_signals=dedup_exact,
            semantic_signals=dedup_semantic,
        )

    def route(self, query: str) -> RetrievalStrategy:
        """
        Selects retrieval strategy based on deterministic signal analysis.
        """
        if not self.config.enabled:
            strategy = RetrievalStrategy(self.config.default_strategy)
            logger.info(
                "Adaptive retrieval disabled; using configured default strategy",
                strategy=strategy.value,
            )
            return strategy

        signals = self.analyze_signals(query)

        # Routing decision table
        if signals.has_exact_match and signals.has_semantic:
            strategy = RetrievalStrategy.HYBRID
        elif signals.has_exact_match and not signals.has_semantic:
            strategy = RetrievalStrategy.LEXICAL
        elif signals.has_semantic and not signals.has_exact_match:
            strategy = RetrievalStrategy.DENSE
        else:
            strategy = RetrievalStrategy(self.config.default_strategy)

        logger.info(
            "Adaptive retrieval routing decision",
            query=query,
            strategy=strategy.value,
            exact_count=len(signals.exact_signals),
            semantic_count=len(signals.semantic_signals),
        )
        return strategy
