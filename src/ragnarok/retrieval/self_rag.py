"""
Self-RAG Decision Gate Subsystem.

Evaluates whether incoming user queries require retrieval from the indexed
knowledge corpus or can be answered reliably using internal world knowledge.
Provides fast classification returning structured routing decisions:
RETRIEVE vs. DIRECT_GENERATE.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import structlog

from ragnarok.core.config import SelfRagConfig

logger = structlog.get_logger(__name__)


class SelfRagDecision(StrEnum):
    """Routing decisions emitted by the Self-RAG decision gate."""

    RETRIEVE = "RETRIEVE"
    DIRECT_GENERATE = "DIRECT_GENERATE"


@dataclass(frozen=True)
class SelfRagResult:
    """Structured result returned by the Self-RAG decision gate."""

    decision: SelfRagDecision
    reasoning: str = ""


class LLMClientProtocol(Protocol):
    """Structural typing contract for LLM completion providers."""

    def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        ...


class DefaultLiteLLMClient:
    """Default adapter wrapping LiteLLM completion calls."""

    def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        import litellm

        return litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )


_SELF_RAG_SYSTEM_PROMPT = """You are a retrieval decision gate for a technical documentation search and question answering system (RAGnarok).
Evaluate if the user's query requires retrieving document evidence from an indexed corpus or can be answered directly using general internal knowledge.

Criteria:
- DIRECT_GENERATE: General knowledge questions (common science, history, geography, basic math, broad programming concepts without proprietary contexts), greetings ("hello"), or conversational pleasantries.
- RETRIEVE: Questions asking about specific technical architectures, configuration keys, database schemas, code mechanisms, error messages, domain-specific systems, or documentation details. When in doubt, prefer RETRIEVE.

You MUST respond strictly with valid JSON conforming to:
{
  "decision": "RETRIEVE" | "DIRECT_GENERATE",
  "reasoning": "<concise explanation>"
}"""


class SelfRagGate:
    """
    Self-RAG Decision Gate evaluating query retrieval requirements.
    """

    def __init__(
        self,
        config: SelfRagConfig | None = None,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        self.config = config if config is not None else SelfRagConfig()
        self.llm_client = llm_client if llm_client is not None else DefaultLiteLLMClient()

    def evaluate(self, query: str) -> SelfRagResult:
        """
        Evaluates whether the incoming query requires corpus retrieval or direct answer generation.
        """
        if not self.config.enabled:
            return SelfRagResult(
                decision=SelfRagDecision.RETRIEVE,
                reasoning="Self-RAG gate disabled by configuration",
            )

        cleaned_query = query.strip()
        if not cleaned_query:
            return SelfRagResult(
                decision=SelfRagDecision.DIRECT_GENERATE,
                reasoning="Empty query requires no retrieval",
            )

        messages = [
            {"role": "system", "content": _SELF_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"User Query: {cleaned_query}"},
        ]

        try:
            response = self.llm_client.completion(
                model=self.config.model,
                messages=messages,
                temperature=0.0,
            )

            raw_content = ""
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                    raw_content = first_choice.message.content or ""
                elif isinstance(first_choice, dict):
                    raw_content = first_choice.get("message", {}).get("content", "")
            elif isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    raw_content = choices[0].get("message", {}).get("content", "")
            elif isinstance(response, str):
                raw_content = response

            decision, reasoning = self._parse_response(raw_content)
            logger.info(
                "Self-RAG evaluation complete",
                query=cleaned_query,
                decision=decision.value,
                reasoning=reasoning,
            )
            return SelfRagResult(decision=decision, reasoning=reasoning)

        except Exception as exc:
            logger.warning(
                "Self-RAG gate evaluation encountered an error; falling back to RETRIEVE",
                query=cleaned_query,
                error=str(exc),
            )
            return SelfRagResult(
                decision=SelfRagDecision.RETRIEVE,
                reasoning=f"Fail-safe fallback due to evaluation error: {exc}",
            )

    @staticmethod
    def _parse_response(raw_text: str) -> tuple[SelfRagDecision, str]:
        """Parses structured JSON decision from LLM output with regex fallback."""
        cleaned = raw_text.strip()
        # Attempt direct JSON parse or extraction from markdown json fences
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                dec_str = str(payload.get("decision", "")).strip().upper()
                reasoning = str(payload.get("reasoning", "")).strip()
                if dec_str == "DIRECT_GENERATE":
                    return SelfRagDecision.DIRECT_GENERATE, reasoning
                return SelfRagDecision.RETRIEVE, reasoning
            except Exception:
                pass

        # Text keyword fallback
        upper = cleaned.upper()
        if "DIRECT_GENERATE" in upper and "RETRIEVE" not in upper:
            return SelfRagDecision.DIRECT_GENERATE, cleaned
        return SelfRagDecision.RETRIEVE, cleaned
