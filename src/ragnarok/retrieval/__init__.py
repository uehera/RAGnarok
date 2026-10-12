"""
Retrieval Subsystem of RAGnarok.

Exports pre-retrieval optimization, Self-RAG decision gate, query rewriter,
adaptive strategy router, and metadata filtering engine.
"""

from __future__ import annotations

from ragnarok.retrieval.filter import (
    MetadataConstraint,
    MetadataFilterEngine,
    extract_constraints,
)
from ragnarok.retrieval.rewriter import QueryRewriter
from ragnarok.retrieval.router import (
    AdaptiveRouter,
    QuerySignals,
    RetrievalStrategy,
)
from ragnarok.retrieval.self_rag import (
    DefaultLiteLLMClient,
    LLMClientProtocol,
    SelfRagDecision,
    SelfRagGate,
    SelfRagResult,
)

__all__ = [
    "DefaultLiteLLMClient",
    "LLMClientProtocol",
    "SelfRagDecision",
    "SelfRagGate",
    "SelfRagResult",
    "QueryRewriter",
    "RetrievalStrategy",
    "QuerySignals",
    "AdaptiveRouter",
    "MetadataConstraint",
    "MetadataFilterEngine",
    "extract_constraints",
]
