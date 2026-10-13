"""
Answer Generation & Citations Subsystem of RAGnarok.

Exports:
- Prompt formulation (build_generation_messages, format_user_prompt, DEFAULT_GROUNDED_SYSTEM_PROMPT, REFUSAL_MESSAGE)
- Real-time token streaming (AnswerGenerator)
- Multi-model fallback chain (stream_with_fallback, StreamingLLMClientProtocol, DefaultLiteLLMStreamingClient)
- Minimal output sanitization (sanitize_generated_text)
- Citation and footnote resolution (resolve_citations, extract_citation_indices, format_footnotes_table)
- Orchestration service (GenerationService)
- Data contracts (CitationFootnote, LatencyMetrics, TokenUsage, GenerationResponse)
"""

from __future__ import annotations

from ragnarok.generation.citations import (
    extract_citation_indices,
    format_footnotes_table,
    resolve_citations,
)
from ragnarok.generation.contract import (
    CitationFootnote,
    GenerationResponse,
    LatencyMetrics,
    TokenUsage,
)
from ragnarok.generation.fallback import (
    DefaultLiteLLMStreamingClient,
    StreamingLLMClientProtocol,
    stream_with_fallback,
)
from ragnarok.generation.generator import AnswerGenerator
from ragnarok.generation.prompts import (
    DEFAULT_GROUNDED_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
    build_generation_messages,
    format_user_prompt,
)
from ragnarok.generation.sanitizer import sanitize_generated_text
from ragnarok.generation.service import GenerationService

__all__ = [
    # Contracts
    "CitationFootnote",
    "LatencyMetrics",
    "TokenUsage",
    "GenerationResponse",
    # Prompts
    "DEFAULT_GROUNDED_SYSTEM_PROMPT",
    "REFUSAL_MESSAGE",
    "build_generation_messages",
    "format_user_prompt",
    # Streaming & Fallback
    "StreamingLLMClientProtocol",
    "DefaultLiteLLMStreamingClient",
    "stream_with_fallback",
    "AnswerGenerator",
    # Sanitization
    "sanitize_generated_text",
    # Citations
    "extract_citation_indices",
    "resolve_citations",
    "format_footnotes_table",
    # Orchestration Service
    "GenerationService",
]
