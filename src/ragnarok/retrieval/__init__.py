"""
Retrieval Subsystem of RAGnarok.

Exports:
- Pre-retrieval optimization (Self-RAG gate, query rewriter, adaptive router, metadata filter)
- Child result contract schema (ChildResultCandidate)
- Lexical search execution (LexicalSearcher, execute_lexical_search)
- Dense vector search execution (DenseSearcher, execute_dense_search)
- Reciprocal Rank Fusion (reciprocal_rank_fusion, HybridFusionService)
- Cross-Encoder reranking (CrossEncoderReranker, SentenceTransformerCrossEncoder, rerank_candidates)
"""

from __future__ import annotations

from ragnarok.retrieval.contract import ChildResultCandidate
from ragnarok.retrieval.dense_search import DenseSearcher, execute_dense_search
from ragnarok.retrieval.filter import (
    MetadataConstraint,
    MetadataFilterEngine,
    extract_constraints,
)
from ragnarok.retrieval.fusion import HybridFusionService, reciprocal_rank_fusion
from ragnarok.retrieval.lexical_search import LexicalSearcher, execute_lexical_search
from ragnarok.retrieval.reranker import (
    CrossEncoderProtocol,
    CrossEncoderReranker,
    SentenceTransformerCrossEncoder,
    rerank_candidates,
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
    # Result Contract
    "ChildResultCandidate",
    # Pre-Retrieval
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
    # Search Execution
    "LexicalSearcher",
    "execute_lexical_search",
    "DenseSearcher",
    "execute_dense_search",
    # Fusion & Reranking
    "reciprocal_rank_fusion",
    "HybridFusionService",
    "CrossEncoderProtocol",
    "CrossEncoderReranker",
    "SentenceTransformerCrossEncoder",
    "rerank_candidates",
]
