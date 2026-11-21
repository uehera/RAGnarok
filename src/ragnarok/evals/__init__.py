"""
Formal Evaluation Subsystem for RAGnarok.

Provides RAGAS and DeepEval evaluation engines measuring Context Precision,
Context Recall, Faithfulness, and Answer Relevancy across golden datasets.
"""

from __future__ import annotations

from ragnarok.evals.ragas_bench import RagasBenchmarkReport, RagasBenchmarkRunner

__all__ = [
    "RagasBenchmarkReport",
    "RagasBenchmarkRunner",
]
