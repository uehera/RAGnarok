"""
Real-Time Asynchronous Token Streaming Generator Subsystem.

Coordinates real-time terminal stdout streaming via Rich Live display,
measures Time-to-First-Token (TTFT), buffers answer text in memory,
applies minimal output sanitization, and binds verified citation footnotes.
Adheres strictly to dev/workspace/generation_citations.md and dev/arc/generation.md.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from rich.console import Console
from rich.live import Live
from rich.text import Text

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None  # type: ignore[assignment]

from ragnarok.context.contract import AssembledContextPayload
from ragnarok.core.config import GenerationConfig
from ragnarok.generation.citations import resolve_citations
from ragnarok.generation.contract import (
    GenerationResponse,
    LatencyMetrics,
    TokenUsage,
)
from ragnarok.generation.fallback import (
    DefaultLiteLLMStreamingClient,
    StreamingLLMClientProtocol,
    stream_with_fallback,
)
from ragnarok.generation.prompts import build_generation_messages
from ragnarok.generation.sanitizer import sanitize_generated_text
from ragnarok.guardrails.output_guard import SpeculativeStreamSanitizer
from ragnarok.guardrails.service import GuardrailsService

logger = structlog.get_logger(__name__)


class AnswerGenerator:
    """
    Stateful service coordinating grounded answer generation, real-time streaming,
    fallback resilience, and citation resolution.
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        client: StreamingLLMClientProtocol | None = None,
        console: Console | None = None,
    ) -> None:
        self.config = config if config is not None else GenerationConfig()
        self.client = client if client is not None else DefaultLiteLLMStreamingClient()
        self.console = console if console is not None else Console()

    def _build_model_list(self) -> list[str]:
        """Constructs the prioritized model list including fallback candidates."""
        models = [self.config.model]
        if self.config.fallbacks.enabled:
            for fallback_model in self.config.fallbacks.models:
                if fallback_model not in models:
                    models.append(fallback_model)
        return models

    async def generate(
        self,
        query: str,
        context: str | AssembledContextPayload,
        provenance_map: dict[str, Any] | None = None,
        render_live: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> GenerationResponse:
        """
        Executes grounded answer synthesis:
        1. Formulates grounded prompt with refusal directives and numbered source blocks.
        2. Streams tokens asynchronously via fallback chain with TTFT measurement.
        3. Buffers and sanitizes synthesized text against credential leaks.
        4. Resolves inline citations [X] against the context provenance table.
        """
        # Resolve context text and provenance dictionary
        if isinstance(context, AssembledContextPayload):
            context_text = context.assembled_context_text
            provenance = provenance_map if provenance_map is not None else context.provenance_map
        else:
            context_text = context
            provenance = provenance_map if provenance_map is not None else {}

        messages = build_generation_messages(query=query, context=context_text, history=history)
        models = self._build_model_list()

        start_time = time.perf_counter()
        first_token_time: float | None = None
        buffer_chunks: list[str] = []

        def on_fallback(failed_model: str, next_model: str, error: Exception) -> None:
            msg = f"[Fallback] Primary model '{failed_model}' failed ({type(error).__name__}); switching to '{next_model}'..."
            logger.warning(msg)
            if render_live and self.console:
                self.console.print(f"[bold yellow]{msg}[/bold yellow]")

        # Establish connection via fallback chain
        token_stream, model_used, fallback_triggered = await stream_with_fallback(
            models=models,
            messages=messages,
            client=self.client,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout_seconds=self.config.timeout_seconds,
            on_fallback_notice=on_fallback,
        )

        # Stream tokens with speculative windowing to prevent credentials leakage
        if render_live and self.console:
            stream_sanitizer = SpeculativeStreamSanitizer(window_size=35)
            display_text = Text()
            with Live(display_text, console=self.console, refresh_per_second=20) as live:
                async for token in token_stream:
                    if first_token_time is None and token.strip():
                        first_token_time = time.perf_counter()
                    buffer_chunks.append(token)
                    safe_token = stream_sanitizer.feed(token)
                    if safe_token:
                        display_text.append(safe_token)
                        live.update(display_text)
                final_token = stream_sanitizer.flush()
                if final_token:
                    display_text.append(final_token)
                    live.update(display_text)
        else:
            async for token in token_stream:
                if first_token_time is None and token.strip():
                    first_token_time = time.perf_counter()
                buffer_chunks.append(token)

        end_time = time.perf_counter()

        # Compute latency metrics
        if first_token_time is not None:
            ttft_ms = (first_token_time - start_time) * 1000.0
        else:
            ttft_ms = (end_time - start_time) * 1000.0
        total_duration_ms = (end_time - start_time) * 1000.0

        latency_metrics = LatencyMetrics(
            ttft_ms=round(ttft_ms, 2),
            total_duration_ms=round(total_duration_ms, 2),
        )

        raw_text = "".join(buffer_chunks)

        # Comprehensive Output Guardrails (Secret Redaction + Grounding Check)
        output_guard_decision = None
        context_str = (
            context.assembled_context_text
            if isinstance(context, AssembledContextPayload)
            else str(context)
        )
        try:
            guard_service = GuardrailsService.from_config()
            output_guard_decision = guard_service.validate_output(
                text=raw_text,
                context=context_str,
            )
            sanitized_text = output_guard_decision.redacted_text
        except Exception:
            sanitized_text = sanitize_generated_text(raw_text)

        # Citation Resolution & Ghost Citation Elimination
        valid_citations, ghost_citations = resolve_citations(
            text=sanitized_text, provenance_map=provenance
        )

        response = GenerationResponse(
            query=query,
            answer=sanitized_text,
            model_used=model_used,
            fallback_triggered=fallback_triggered,
            citations=valid_citations,
            ghost_citations=ghost_citations,
            latency_metrics=latency_metrics,
            token_usage=TokenUsage(),
            output_guard_decision=output_guard_decision,
        )

        logger.info(
            "Generation completed",
            model=model_used,
            fallback=fallback_triggered,
            ttft_ms=latency_metrics.ttft_ms,
            total_duration_ms=latency_metrics.total_duration_ms,
            citations_count=len(valid_citations),
            ghost_citations_count=len(ghost_citations),
        )
        return response

    def generate_sync(
        self,
        query: str,
        context: str | AssembledContextPayload,
        provenance_map: dict[str, Any] | None = None,
        render_live: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> GenerationResponse:
        """Synchronous wrapper for generate."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            if nest_asyncio is not None:
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.generate(
                        query=query,
                        context=context,
                        provenance_map=provenance_map,
                        render_live=render_live,
                        history=history,
                    )
                )
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(
                    asyncio.run,
                    self.generate(
                        query=query,
                        context=context,
                        provenance_map=provenance_map,
                        render_live=render_live,
                        history=history,
                    ),
                ).result()
        return asyncio.run(
            self.generate(
                query=query,
                context=context,
                provenance_map=provenance_map,
                render_live=render_live,
                history=history,
            )
        )
