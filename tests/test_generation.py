"""
Comprehensive unit and integration tests for Phase 6: Answer Generation & Citations.
(Testing Point 9 in dev/workspace/tests_plan.md).

Verifies:
1. Grounded prompt formulation enforcing strict grounding and refusal directives.
2. Real-time asynchronous token streaming with TTFT measurement (< 500 ms).
3. Fallback chain: Simulated HTTP 429 rate limit or timeout triggers fallback model cleanly.
4. Inline citations resolve 1:1 to valid source documents, headings, and parent chunk IDs.
5. Ghost citations (referencing nonexistent sources) are detected and omitted from footnotes.
6. Output sanitization: API keys and connection strings are redacted before rendering.
7. Full generation service integration with AssembledContextPayload.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest

from ragnarok.context.contract import AssembledContextPayload
from ragnarok.core.config import GenerationConfig, GenerationFallbacksConfig
from ragnarok.generation.citations import (
    format_footnotes_table,
    resolve_citations,
)
from ragnarok.generation.fallback import stream_with_fallback
from ragnarok.generation.generator import AnswerGenerator
from ragnarok.generation.prompts import (
    REFUSAL_MESSAGE,
    build_generation_messages,
    format_user_prompt,
)
from ragnarok.generation.sanitizer import sanitize_generated_text
from ragnarok.generation.service import GenerationService


class MockStreamingClient:
    """Mock streaming LLM client for deterministic testing."""

    def __init__(
        self,
        token_streams: dict[str, list[str]] | None = None,
        failing_models: set[str] | None = None,
        token_delay_seconds: float = 0.005,
    ) -> None:
        self.token_streams = token_streams or {}
        self.failing_models = failing_models or set()
        self.token_delay_seconds = token_delay_seconds
        self.call_history: list[str] = []

    async def stream_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1500,
        timeout_seconds: float = 15.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.call_history.append(model)
        if model in self.failing_models:
            raise RuntimeError(f"Simulated HTTP 429 rate limit or timeout on model: {model}")

        tokens = self.token_streams.get(model, ["Default", " generated", " answer."])
        for token in tokens:
            if self.token_delay_seconds > 0:
                await asyncio.sleep(self.token_delay_seconds)
            yield token


def test_prompt_formulation_enforces_grounding_and_refusal() -> None:
    """
    Verifies that system prompt incorporates grounding rules, explicit citation syntax,
    and the standardized refusal directive.
    """
    query = "Where are parent chunks stored?"
    context_text = "[Source 1] (Document: storage.md, Section: Architecture)\nParent chunks reside in PostgreSQL."

    messages = build_generation_messages(query=query, context=context_text)

    assert len(messages) == 2
    system_msg = messages[0]["content"]
    user_msg = messages[1]["content"]

    # Grounding directives present
    assert "ONLY the factual information provided in the Context sources" in system_msg
    assert "Never infer, hypothesize, or extrapolate" in system_msg

    # Explicit refusal directive present
    assert REFUSAL_MESSAGE in system_msg
    assert "I cannot find sufficient information in the provided documents." in system_msg

    # Citation rules present
    assert "[1] or [2]" in system_msg

    # User message contains context and query
    assert "[Source 1] (Document: storage.md, Section: Architecture)" in user_msg
    assert "User Question:\nWhere are parent chunks stored?" in user_msg


def test_prompt_formulation_with_empty_context() -> None:
    """Verifies that empty context text produces a valid user prompt indicating no documents."""
    query = "What is the capital of France?"
    user_prompt = format_user_prompt(query=query, context_text="")
    assert "[No context documents provided]" in user_prompt
    assert "What is the capital of France?" in user_prompt


@pytest.mark.asyncio
async def test_token_streaming_and_ttft_measurement() -> None:
    """
    Verifies that tokens stream asynchronously and that Time-to-First-Token (TTFT)
    is recorded accurately (< 500 ms).
    """
    mock_tokens = ["Parent-", "child", " retrieval", " indexes", " small", " chunks", " [1]."]
    mock_client = MockStreamingClient(
        token_streams={"groq/llama-3.3-70b-versatile": mock_tokens},
        token_delay_seconds=0.01,
    )

    config = GenerationConfig(model="groq/llama-3.3-70b-versatile")
    generator = AnswerGenerator(config=config, client=mock_client)

    response = await generator.generate(
        query="Explain parent-child retrieval",
        context="[Source 1] (Document: arch.md, Section: Chunks)\nSmall child chunks are indexed.",
        provenance_map={
            "[Source 1]": {
                "document_id": str(uuid4()),
                "filename": "arch.md",
                "parent_chunk_id": str(uuid4()),
                "heading_path": ["Chunks"],
            }
        },
    )

    # Asserts streamed text matches concatenation
    assert response.answer == "Parent-child retrieval indexes small chunks [1]."
    assert response.model_used == "groq/llama-3.3-70b-versatile"
    assert response.fallback_triggered is False

    # Asserts TTFT is measured and well below 500ms
    assert response.latency_metrics.ttft_ms > 0
    assert response.latency_metrics.ttft_ms < 500.0
    assert response.latency_metrics.total_duration_ms >= response.latency_metrics.ttft_ms


@pytest.mark.asyncio
async def test_fallback_chain_on_provider_rate_limit() -> None:
    """
    Simulates an HTTP 429 rate limit or outage on the primary model;
    verifies that the fallback chain transparently invokes the secondary model.
    """
    primary = "openai/gpt-4o"
    secondary = "groq/llama-3.3-70b-versatile"

    mock_client = MockStreamingClient(
        token_streams={secondary: ["Fallback", " answer", " generated", " cleanly", " [1]."]},
        failing_models={primary},
    )

    fallback_notices: list[str] = []

    def on_fallback(failed: str, next_m: str, exc: Exception) -> None:
        fallback_notices.append(f"{failed}->{next_m}")

    stream, model_used, fallback_triggered = await stream_with_fallback(
        models=[primary, secondary],
        messages=[{"role": "user", "content": "test"}],
        client=mock_client,
        on_fallback_notice=on_fallback,
    )

    tokens = [t async for t in stream]
    assembled = "".join(tokens)

    # Verification
    assert assembled == "Fallback answer generated cleanly [1]."
    assert model_used == secondary
    assert fallback_triggered is True
    assert primary in mock_client.call_history
    assert secondary in mock_client.call_history
    assert len(fallback_notices) == 1
    assert fallback_notices[0] == f"{primary}->{secondary}"


@pytest.mark.asyncio
async def test_fallback_chain_all_failing_raises_runtime_error() -> None:
    """Verifies that if every model in the fallback chain fails, a RuntimeError is raised."""
    mock_client = MockStreamingClient(
        failing_models={"model_1", "model_2"},
    )

    with pytest.raises(RuntimeError, match="All generation models in fallback chain failed"):
        await stream_with_fallback(
            models=["model_1", "model_2"],
            messages=[{"role": "user", "content": "test"}],
            client=mock_client,
        )


def test_citation_resolution_and_footnotes_table() -> None:
    """
    Verifies that inline citations ([1], [2]) resolve 1:1 to source document,
    heading hierarchy, and parent chunk ID in the footnotes table.
    """
    doc1_id = uuid4()
    doc2_id = uuid4()
    p1_id = uuid4()
    p2_id = uuid4()

    provenance_map = {
        "[Source 1]": {
            "document_id": str(doc1_id),
            "filename": "database_spec.pdf",
            "parent_chunk_id": str(p1_id),
            "heading_path": ["Database", "PostgreSQL"],
        },
        "[Source 2]": {
            "document_id": str(doc2_id),
            "filename": "retrieval_spec.md",
            "parent_chunk_id": str(p2_id),
            "heading_path": ["Retrieval", "TurboVec"],
        },
    }

    generated_text = (
        "PostgreSQL 16 persists parent chunks [1]. TurboVec indexes child embeddings [2]."
    )

    valid_citations, ghost_citations = resolve_citations(
        text=generated_text, provenance_map=provenance_map
    )

    assert len(valid_citations) == 2
    assert len(ghost_citations) == 0

    assert valid_citations[0].citation_index == 1
    assert valid_citations[0].filename == "database_spec.pdf"
    assert valid_citations[0].parent_chunk_id == p1_id
    assert valid_citations[0].heading_path == ["Database", "PostgreSQL"]

    assert valid_citations[1].citation_index == 2
    assert valid_citations[1].filename == "retrieval_spec.md"
    assert valid_citations[1].parent_chunk_id == p2_id
    assert valid_citations[1].heading_path == ["Retrieval", "TurboVec"]

    # Footnotes table rendering check
    table_text = format_footnotes_table(valid_citations)
    assert "[1] database_spec.pdf" in table_text
    assert "Section: Database > PostgreSQL" in table_text
    assert f"Parent Chunk: {p1_id}" in table_text
    assert "[2] retrieval_spec.md" in table_text
    assert "Section: Retrieval > TurboVec" in table_text


def test_ghost_citation_elimination() -> None:
    """
    Verifies that if the model emits a citation referencing a nonexistent source (e.g., [99]),
    it is classified as a ghost citation and excluded from the footnotes table.
    """
    doc_id = uuid4()
    p_id = uuid4()

    provenance_map = {
        "[Source 1]": {
            "document_id": str(doc_id),
            "filename": "valid_doc.md",
            "parent_chunk_id": str(p_id),
            "heading_path": ["Introduction"],
        }
    }

    text_with_ghost = "Valid claim [1], but here is an ungrounded hallucination [99]."

    valid_citations, ghost_citations = resolve_citations(
        text=text_with_ghost, provenance_map=provenance_map
    )

    assert len(valid_citations) == 1
    assert valid_citations[0].citation_index == 1
    assert len(ghost_citations) == 1
    assert ghost_citations[0] == 99

    table = format_footnotes_table(valid_citations)
    assert "[1] valid_doc.md" in table
    assert "[99]" not in table


def test_minimal_output_sanitization_redacts_secrets() -> None:
    """
    Verifies that exposed API keys, database connection strings, passwords,
    and email addresses are redacted with [REDACTED <TYPE>] before rendering.
    """
    raw_output = (
        "Here is the database URI: postgresql://admin:secret123@localhost:5432/ragnarok "
        "and OpenAI key: sk-abcdef1234567890abcdef1234567890 and email: dev@example.com."
    )

    clean_output = sanitize_generated_text(raw_output)

    assert "postgresql://admin" not in clean_output
    assert "[REDACTED DATABASE_URL]" in clean_output

    assert "sk-abcdef1234567890" not in clean_output
    assert "[REDACTED OPENAI_KEY]" in clean_output

    assert "dev@example.com" not in clean_output
    assert "[REDACTED EMAIL]" in clean_output


def test_generation_service_end_to_end_with_assembled_payload() -> None:
    """
    Verifies that GenerationService coordinates prompt formulation,
    streaming, sanitization, and citation resolution with an AssembledContextPayload.
    """
    doc_id = uuid4()
    p_id = uuid4()

    assembled_payload = AssembledContextPayload(
        total_tokens=150,
        max_tokens_budget=4000,
        selected_parent_count=1,
        omitted_parent_count=0,
        assembled_context_text="[Source 1] (Document: test.md, Section: Test)\nRelevant evidence text.",
        provenance_map={
            "[Source 1]": {
                "source_index": 1,
                "document_id": str(doc_id),
                "filename": "test.md",
                "parent_chunk_id": str(p_id),
                "heading_path": ["Test"],
                "best_child_rank": 1,
                "matched_children": [],
            }
        },
    )

    mock_client = MockStreamingClient(
        token_streams={
            "mock-model": ["Synthesized", " answer", " based", " on", " evidence", " [1]."]
        }
    )

    cfg = GenerationConfig(
        model="mock-model",
        fallbacks=GenerationFallbacksConfig(enabled=False, models=[]),
    )

    service = GenerationService.from_config(config=cfg, client=mock_client)
    response = service.generate_sync(
        query="What is the evidence?",
        context=assembled_payload,
    )

    assert response.answer == "Synthesized answer based on evidence [1]."
    assert response.model_used == "mock-model"
    assert len(response.citations) == 1
    assert response.citations[0].citation_index == 1
    assert response.citations[0].filename == "test.md"
    assert response.latency_metrics.ttft_ms >= 0

    # Test formatted terminal layout output
    cli_output = service.format_terminal_output(response)
    assert "RAGnarok Synthesis Output" in cli_output
    assert "Model: mock-model" in cli_output
    assert "[1] test.md" in cli_output
