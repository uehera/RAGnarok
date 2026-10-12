"""
Query Rewriter Subsystem.

LLM-based single-query reformulation clarifying ambiguous pronouns,
resolving conversational references, and making implicit retrieval goals explicit.
Strictly preserves technical identifiers (symbols, error codes, API names)
and produces exactly one reformulated query (multi-query expansion is prohibited).
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from ragnarok.core.config import QueryRewritingConfig
from ragnarok.retrieval.self_rag import DefaultLiteLLMClient, LLMClientProtocol

logger = structlog.get_logger(__name__)

_REWRITER_SYSTEM_PROMPT = """You are a technical search query rewriter for a documentation retrieval engine (RAGnarok).
Your objective is to clarify vague references, expand ambiguous pronouns (e.g., 'it', 'they', 'this', 'that'), and state implicit retrieval goals clearly.

MANDATORY RULES:
1. Output EXACTLY ONE rewritten search query. Multi-query branching, lists, numbered alternatives, and bullet points are STRICTLY PROHIBITED.
2. PRESERVE all exact technical terms, symbols, error codes, class/function names, and file paths VERBATIM (e.g., IdMapIndex, bm25s, 404, docling.md, parent_chunks).
3. Do NOT answer the question, add preamble, or explain your edit. Output ONLY the single rewritten query text.
"""


class QueryRewriter:
    """
    Stateful service executing LLM-based single-query reformulation.
    """

    def __init__(
        self,
        config: QueryRewritingConfig | None = None,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        self.config = config if config is not None else QueryRewritingConfig()
        self.llm_client = llm_client if llm_client is not None else DefaultLiteLLMClient()

    def rewrite(self, query: str) -> str:
        """
        Reformulates an incoming query into a single clarified technical retrieval query.
        Returns the original query unchanged if rewriting is disabled or encounters an error.
        """
        cleaned_query = query.strip()
        if not self.config.enabled or not cleaned_query:
            return cleaned_query

        messages = [
            {"role": "system", "content": _REWRITER_SYSTEM_PROMPT},
            {"role": "user", "content": f"User Query: {cleaned_query}"},
        ]

        try:
            response = self.llm_client.completion(
                model=self.config.model,
                messages=messages,
                temperature=0.0,
            )

            raw_text = self._extract_content(response)
            rewritten = self._sanitize_single_query(raw_text, fallback=cleaned_query)

            logger.info(
                "Query rewritten successfully",
                original=cleaned_query,
                rewritten=rewritten,
            )
            return rewritten

        except Exception as exc:
            logger.warning(
                "Query rewriting failed; falling back to original query",
                original=cleaned_query,
                error=str(exc),
            )
            return cleaned_query

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extracts text content from various LLM response formats."""
        if hasattr(response, "choices") and response.choices:
            first = response.choices[0]
            if hasattr(first, "message") and hasattr(first.message, "content"):
                return first.message.content or ""
            if isinstance(first, dict):
                return first.get("message", {}).get("content", "")
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        if isinstance(response, str):
            return response
        return ""

    @staticmethod
    def _sanitize_single_query(text: str, fallback: str) -> str:
        """
        Enforces single-query output: strips quotes, preambles, and enforces
        single-line string even if the model attempts to generate multiple items.
        """
        stripped = text.strip()
        if not stripped:
            return fallback

        # Strip surrounding markdown code blocks or quotes
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()

        # Extract lines, ignoring empty lines or explanations
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return fallback

        first_line = lines[0]
        # Strip list markers like "1. ", "- ", "* "
        clean_first = re.sub(r"^(\d+[\.\)]|\*|\-)\s*", "", first_line).strip()
        # Strip surrounding quotes
        if (clean_first.startswith('"') and clean_first.endswith('"')) or (
            clean_first.startswith("'") and clean_first.endswith("'")
        ):
            clean_first = clean_first[1:-1].strip()

        # If model outputs "Rewritten Query: ...", strip that prefix
        clean_first = re.sub(
            r"^(Rewritten Query|Rewritten|Search Query):\s*",
            "",
            clean_first,
            flags=re.IGNORECASE,
        ).strip()

        return clean_first if clean_first else fallback
