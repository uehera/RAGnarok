"""
Generation Orchestration Service.

Provides a unified high-level interface for grounded answer generation,
multi-model fallback resilience, output sanitization, and citation rendering.
"""

from __future__ import annotations

from typing import Any

import structlog
from rich.console import Console

from ragnarok.context.contract import AssembledContextPayload
from ragnarok.core.config import AppConfig, GenerationConfig
from ragnarok.generation.citations import format_footnotes_table
from ragnarok.generation.contract import GenerationResponse
from ragnarok.generation.fallback import (
    DefaultLiteLLMStreamingClient,
    StreamingLLMClientProtocol,
)
from ragnarok.generation.generator import AnswerGenerator

logger = structlog.get_logger(__name__)


class GenerationService:
    """
    High-level orchestration service for Phase 6: Answer Generation & Citations.
    """

    def __init__(
        self,
        generator: AnswerGenerator,
    ) -> None:
        self.generator = generator

    @classmethod
    def from_config(
        cls,
        config: GenerationConfig | AppConfig | None = None,
        client: StreamingLLMClientProtocol | None = None,
        console: Console | None = None,
    ) -> GenerationService:
        """Factory method constructing GenerationService from configuration."""
        if isinstance(config, AppConfig):
            gen_cfg = config.generation
        elif isinstance(config, GenerationConfig):
            gen_cfg = config
        else:
            gen_cfg = GenerationConfig()

        llm_client = client if client is not None else DefaultLiteLLMStreamingClient()
        generator = AnswerGenerator(
            config=gen_cfg,
            client=llm_client,
            console=console,
        )
        return cls(generator=generator)

    async def generate(
        self,
        query: str,
        context: str | AssembledContextPayload,
        provenance_map: dict[str, Any] | None = None,
        render_live: bool = False,
    ) -> GenerationResponse:
        """Asynchronously executes answer generation and citation resolution."""
        return await self.generator.generate(
            query=query,
            context=context,
            provenance_map=provenance_map,
            render_live=render_live,
        )

    def generate_sync(
        self,
        query: str,
        context: str | AssembledContextPayload,
        provenance_map: dict[str, Any] | None = None,
        render_live: bool = False,
    ) -> GenerationResponse:
        """Synchronously executes answer generation and citation resolution."""
        return self.generator.generate_sync(
            query=query,
            context=context,
            provenance_map=provenance_map,
            render_live=render_live,
        )

    def format_terminal_output(self, response: GenerationResponse) -> str:
        """
        Formats generation output matching developer CLI specification:

        ================================================================================
        RAGnarok Synthesis Output
        ================================================================================
        Model: [Active Model] | Latency: [TTFT / Total Time] | Fallback: [Yes/No]

        [Streamed Grounded Answer Text with inline markers like [1], [2]...]

        --------------------------------------------------------------------------------
        Sources & Citations
        --------------------------------------------------------------------------------
        [1] filename.pdf
            Section: ...
        """
        ttft_str = f"{response.latency_metrics.ttft_ms:.1f}ms"
        total_str = f"{response.latency_metrics.total_duration_ms:.1f}ms"
        fallback_str = "Yes" if response.fallback_triggered else "No"

        header_lines = [
            "=" * 80,
            "RAGnarok Synthesis Output",
            "=" * 80,
            f"Model: {response.model_used} | Latency: {ttft_str} TTFT / {total_str} Total | Fallback: {fallback_str}",
            "",
            response.answer,
            "",
        ]

        if response.citations:
            footnotes_str = format_footnotes_table(response.citations)
            header_lines.append(footnotes_str)
            header_lines.append("=" * 80)
        else:
            header_lines.append("=" * 80)

        return "\n".join(header_lines)
