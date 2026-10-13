"""
Multi-Model Fallback Chain Subsystem.

Provides fault tolerance against LLM provider outages, socket timeouts, and HTTP 429 rate limits.
Sequentially traverses primary and fallback models configured in config.toml with explicit
developer notifications and transparent stream recovery.
Adheres strictly to dev/workspace/generation_citations.md and dev/arc/generation.md.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, cast

import structlog

logger = structlog.get_logger(__name__)


class StreamingLLMClientProtocol(Protocol):
    """Structural protocol for asynchronous token streaming LLM clients."""

    def stream_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1500,
        timeout_seconds: float = 15.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronously streams string tokens for a completion request."""
        ...


class DefaultLiteLLMStreamingClient:
    """
    Default adapter providing asynchronous token streaming via LiteLLM.
    Supports local runners (Ollama, vLLM) and cloud providers (Groq, OpenAI, Anthropic).
    """

    async def stream_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1500,
        timeout_seconds: float = 15.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        import litellm

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
            **kwargs,
        )

        async for chunk in cast(Any, response):
            if hasattr(chunk, "choices") and chunk.choices:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta:
                    content = getattr(delta, "content", None)
                    if content:
                        yield content


async def _empty_stream() -> AsyncIterator[str]:
    return
    yield ""


async def _chain_first_token(
    first_token: str, stream_iter: AsyncIterator[str]
) -> AsyncIterator[str]:
    yield first_token
    async for token in stream_iter:
        yield token


async def stream_with_fallback(
    models: list[str],
    messages: list[dict[str, str]],
    client: StreamingLLMClientProtocol,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    timeout_seconds: float = 15.0,
    on_fallback_notice: Callable[[str, str, Exception], None] | None = None,
    **kwargs: Any,
) -> tuple[AsyncIterator[str], str, bool]:
    """
    Executes an asynchronous streaming request across a prioritized list of models.
    If the primary model encounters rate limits (429), timeouts, or outages, automatically
    switches to the next available fallback candidate.

    Parameters:
        models: Ordered list of models to try [primary_model, fallback_1, fallback_2...].
        messages: Formatted message payload.
        client: Streaming LLM provider adapter.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in completion.
        timeout_seconds: Timeout per model invocation attempt.
        on_fallback_notice: Optional callback when a fallback model is triggered.

    Returns:
        tuple of (token_stream_generator, model_used, fallback_triggered).

    Raises:
        RuntimeError: If all candidate models in the fallback chain fail.
    """
    if not models:
        raise ValueError("Model candidate list cannot be empty")

    errors: list[tuple[str, Exception]] = []

    for i, current_model in enumerate(models):
        fallback_triggered = i > 0
        try:
            logger.info("Dispatching generation request", model=current_model, attempt=i + 1)
            raw_stream = client.stream_completion(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )

            # Probe the first token to verify provider connectivity before committing
            stream_iter = aiter(raw_stream)
            try:
                first_token = await anext(stream_iter)
            except StopAsyncIteration:
                # Provider responded immediately with zero tokens
                return _empty_stream(), current_model, fallback_triggered

            return (
                _chain_first_token(first_token, stream_iter),
                current_model,
                fallback_triggered,
            )

        except Exception as exc:
            errors.append((current_model, exc))
            next_model = models[i + 1] if i + 1 < len(models) else None

            logger.warning(
                "Generation model failed; evaluating fallback chain",
                failed_model=current_model,
                next_model=next_model,
                error=str(exc),
            )

            if next_model and on_fallback_notice is not None:
                on_fallback_notice(current_model, next_model, exc)

    error_summary = "; ".join(f"{m}: {type(e).__name__}({e})" for m, e in errors)
    raise RuntimeError(f"All generation models in fallback chain failed: {error_summary}")
