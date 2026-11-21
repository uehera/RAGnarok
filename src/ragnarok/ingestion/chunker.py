"""
Two-Stage Parent-Child Chunking Subsystem.

Level 1: MarkdownHeaderTextSplitter splits along structural heading boundaries (#, ##, ###)
into bounded parent blocks adhering to parent_target_size.
Level 2: RecursiveCharacterTextSplitter subdivides parents into child chunks
adhering to child_target_size and child_overlap.
Counts tokens via Hugging Face AutoTokenizer with tiktoken fallback.
Generates zero-based sequential indices (parent_index, chunk_index) and parent_chunk_id foreign keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import structlog
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ragnarok.core.config import ChunkingConfig
from ragnarok.core.models import ChildChunk, ParentChunk

logger = structlog.get_logger(__name__)


class TokenCounterProtocol(Protocol):
    """Structural protocol for text token counters."""

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens in the given text."""
        ...


class TiktokenCounter:
    """Token counter using OpenAI's tiktoken library with offline heuristic fallback."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding: Any = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        # Robust offline fallback: approximate English token count (~4 chars / token)
        words = len(text.split())
        chars = len(text)
        return max(1, max(words, chars // 4))


class AutoTokenizerCounter:
    """Token counter using Hugging Face AutoTokenizer aligned with the embedding model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        fallback_counter: TokenCounterProtocol | None = None,
    ) -> None:
        self.model_name = model_name
        self.fallback = fallback_counter if fallback_counter is not None else TiktokenCounter()
        self._tokenizer: Any = None
        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        try:
            from transformers import AutoTokenizer

            # 1. Check local FastEmbed / models cache first
            cache_dirs = [
                Path("models/fastembed_cache/models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots"),
                Path(__file__).resolve().parents[3] / "models" / "fastembed_cache" / "models--Qdrant--bge-small-en-v1.5-onnx-Q" / "snapshots",
                Path("models/ms-marco-MiniLM-L-12-v2"),
            ]
            for c_dir in cache_dirs:
                if c_dir.exists():
                    if c_dir.name == "snapshots":
                        subdirs = list(c_dir.glob("*"))
                        if subdirs and (subdirs[0] / "tokenizer.json").exists():
                            self._tokenizer = AutoTokenizer.from_pretrained(str(subdirs[0]), local_files_only=True)
                            return
                    elif (c_dir / "tokenizer.json").exists():
                        self._tokenizer = AutoTokenizer.from_pretrained(str(c_dir), local_files_only=True)
                        return

            # 2. Attempt local HF cache loading
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
                return
            except Exception:
                pass

            # 3. Attempt network loading if online
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load Hugging Face AutoTokenizer; falling back to offline counter",
                model=self.model_name,
                error=str(exc),
            )
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                tokens = self._tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception:
                return self.fallback.count_tokens(text)
        return self.fallback.count_tokens(text)


def resolve_token_counter(
    size_counter: str = "transformers.AutoTokenizer",
    model_name: str = "BAAI/bge-small-en-v1.5",
    fallback_counter: str = "tiktoken",
) -> TokenCounterProtocol:
    """
    Pure factory function resolving the active token counter backend.
    Enforces graceful fallback if AutoTokenizer cannot be initialized.
    """
    fallback: TokenCounterProtocol = TiktokenCounter()

    if size_counter in ("transformers.AutoTokenizer", "AutoTokenizer"):
        return AutoTokenizerCounter(model_name=model_name, fallback_counter=fallback)
    elif size_counter == "tiktoken":
        return fallback
    else:
        # Custom or named model
        return AutoTokenizerCounter(model_name=size_counter, fallback_counter=fallback)


class TwoStageChunker:
    """
    Stateful service coordinating two-stage parent-child chunking.
    Adheres strictly to heading boundaries, token budgets, and foreign key relations.
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        token_counter: TokenCounterProtocol | None = None,
        document_guard: Any | None = None,
    ) -> None:
        self.config = config if config is not None else ChunkingConfig()
        self.model_name = model_name
        self.document_guard = document_guard
        self.token_counter = (
            token_counter
            if token_counter is not None
            else resolve_token_counter(
                size_counter=self.config.size_counter,
                model_name=self.model_name,
                fallback_counter=self.config.fallback_counter,
            )
        )

        # Build headers configuration for MarkdownHeaderTextSplitter
        # Maps '#' -> 'Header 1', '##' -> 'Header 2', '###' -> 'Header 3'
        self.headers_to_split_on: list[tuple[str, str]] = [
            (level, f"Header {len(level)}") for level in self.config.heading_levels
        ]

    def chunk_markdown(
        self,
        markdown_content: str,
        document_id: UUID,
        source_name: str | None = None,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Executes two-stage chunking on normalized Markdown content:

        Level 1 (Parent):
        - Splits text on structural heading boundaries using MarkdownHeaderTextSplitter.
        - Preserves breadcrumb heading hierarchy in heading_path.
        - Subdivides oversized sections exceeding parent_target_size.
        - Assigns sequential zero-based parent_index.

        Level 2 (Child):
        - Subdivides each parent chunk using RecursiveCharacterTextSplitter.
        - Bounded by child_target_size and child_overlap.
        - Preserves parent_chunk_id foreign key link and heading_path.
        - Assigns sequential zero-based chunk_index across document.

        Returns:
            tuple[list[ParentChunk], list[ChildChunk]]
        """
        if not markdown_content.strip():
            return [], []

        # Ingestion Document Guard: Screen, defang or quarantine indirect prompt injections
        if self.document_guard is not None:
            guard_decision = self.document_guard.validate(
                content=markdown_content,
                source_name=source_name,
            )
            if guard_decision.quarantined:
                logger.warning(
                    "TwoStageChunker: Document quarantined; skipping chunk generation",
                    document_id=str(document_id),
                    quarantine_path=guard_decision.quarantine_path,
                )
                return [], []
            markdown_content = guard_decision.defanged_content

        # -------------------------------------------------------------
        # Stage 1: Level 1 Markdown Header Splitting
        # -------------------------------------------------------------
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        raw_sections = header_splitter.split_text(markdown_content)

        # Splitter for oversized parent sections (> parent_target_size)
        parent_recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.parent_target_size,
            chunk_overlap=self.config.parent_overlap,
            length_function=self.token_counter.count_tokens,
            separators=self.config.separators,
        )

        parent_chunks: list[ParentChunk] = []
        parent_index_counter = 0

        for section in raw_sections:
            content = section.page_content.strip()
            if not content:
                continue

            # Extract heading breadcrumbs in hierarchical order: Header 1, Header 2, Header 3...
            metadata = section.metadata
            heading_keys = sorted(
                [k for k in metadata if k.startswith("Header ")],
                key=lambda k: int(k.split(" ")[1]) if k.split(" ")[1].isdigit() else 99,
            )
            heading_path = [str(metadata[k]) for k in heading_keys]

            token_count = self.token_counter.count_tokens(content)
            if token_count > self.config.parent_target_size:
                # Subdivide oversized parent section into bounded parent units
                sub_parent_texts = parent_recursive_splitter.split_text(content)
                for sub_text in sub_parent_texts:
                    sub_text_clean = sub_text.strip()
                    if not sub_text_clean:
                        continue
                    parent_chunks.append(
                        ParentChunk(
                            parent_chunk_id=uuid4(),
                            document_id=document_id,
                            content=sub_text_clean,
                            parent_index=parent_index_counter,
                            heading_path=heading_path,
                        )
                    )
                    parent_index_counter += 1
            else:
                parent_chunks.append(
                    ParentChunk(
                        parent_chunk_id=uuid4(),
                        document_id=document_id,
                        content=content,
                        parent_index=parent_index_counter,
                        heading_path=heading_path,
                    )
                )
                parent_index_counter += 1

        # -------------------------------------------------------------
        # Stage 2: Level 2 Recursive Character Splitting (Children)
        # -------------------------------------------------------------
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.child_target_size,
            chunk_overlap=self.config.child_overlap,
            length_function=self.token_counter.count_tokens,
            separators=self.config.separators,
        )

        child_chunks: list[ChildChunk] = []
        chunk_index_counter = 0

        for parent in parent_chunks:
            child_texts = child_splitter.split_text(parent.content)
            for child_text in child_texts:
                child_clean = child_text.strip()
                if not child_clean:
                    continue
                child_chunks.append(
                    ChildChunk(
                        chunk_id=uuid4(),
                        document_id=document_id,
                        parent_chunk_id=parent.parent_chunk_id,
                        content=child_clean,
                        chunk_index=chunk_index_counter,
                        heading_path=list(parent.heading_path),
                    )
                )
                chunk_index_counter += 1

        logger.info(
            "Completed two-stage chunking",
            document_id=str(document_id),
            parent_count=len(parent_chunks),
            child_count=len(child_chunks),
        )

        return parent_chunks, child_chunks


def chunk_document(
    document_id: UUID,
    markdown_text: str,
    config: ChunkingConfig | None = None,
    token_counter: TokenCounterProtocol | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Pure procedural function that runs two-stage parent-child chunking on markdown text.
    """
    chunker = TwoStageChunker(config=config, token_counter=token_counter)
    return chunker.chunk_markdown(markdown_content=markdown_text, document_id=document_id)
