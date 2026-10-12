"""
Lexical Indexing Subsystem (BM25s).

Builds, persists, and queries inverted BM25 indexes using bm25s[core].
Maintains 1:1 bijective mappings between internal document positions and PostgreSQL chunk UUIDs.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import bm25s
import structlog

from ragnarok.core.config import LexicalRetrievalConfig
from ragnarok.core.models import ChildChunk

logger = structlog.get_logger(__name__)


def _load_default_stemmer() -> Any:
    try:
        stemmer_mod = importlib.import_module("Stemmer")
        return stemmer_mod.Stemmer("english")
    except Exception:
        return None


_DEFAULT_STEMMER: Any = _load_default_stemmer()


def tokenize_corpus(
    texts: list[str],
    config: LexicalRetrievalConfig | None = None,
    return_ids: bool = False,
) -> Any:
    """
    Pure procedural tokenization function applying configured BM25s tokenizer settings:
    case normalization, stop words removal, and English stemming.
    """
    cfg = config if config is not None else LexicalRetrievalConfig()
    lower = cfg.case.lower() == "lowercase"
    stopwords: list[str] | str = "en_plus" if cfg.stop_words_removed else []

    tokenize_kwargs: dict[str, Any] = {
        "stopwords": stopwords,
        "lower": lower,
        "show_progress": False,
        "return_ids": return_ids,
    }
    if cfg.stemming_enabled and _DEFAULT_STEMMER is not None:
        tokenize_kwargs["stemmer"] = _DEFAULT_STEMMER

    return bm25s.tokenize(texts, **tokenize_kwargs)


def tokenize_query(
    query: str,
    config: LexicalRetrievalConfig | None = None,
) -> Any:
    """
    Tokenizes a single query using settings identical to the corpus tokenizer.
    """
    return tokenize_corpus([query], config=config)


class LexicalIndex:
    """
    Stateful service managing the BM25s inverted index lifecycle,
    persistence to data/indexes/bm25/, and ordered UUID mappings.
    """

    def __init__(
        self,
        config: LexicalRetrievalConfig | None = None,
        retriever: bm25s.BM25 | None = None,
        chunk_ids: list[UUID] | None = None,
    ) -> None:
        self.config = config if config is not None else LexicalRetrievalConfig()
        self.retriever = retriever
        self.chunk_ids: list[UUID] = chunk_ids if chunk_ids is not None else []

    @property
    def is_empty(self) -> bool:
        """Returns True if the index contains zero documents."""
        return len(self.chunk_ids) == 0 or self.retriever is None

    def build(self, chunks: list[ChildChunk]) -> None:
        """
        Builds a fresh BM25 index from finalized child chunk records.
        Preserves an ordered array of chunk UUIDs matching BM25 internal document positions.
        """
        if not chunks:
            self.retriever = None
            self.chunk_ids = []
            return

        texts = [chunk.content for chunk in chunks]
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]

        tokens = tokenize_corpus(texts, config=self.config)
        retriever = bm25s.BM25(method="lucene")
        retriever.index(tokens, show_progress=False)
        self.retriever = retriever

        logger.info(
            "Built BM25s lexical index",
            chunk_count=len(self.chunk_ids),
        )

    def save(self, index_dir: Path | str) -> None:
        """
        Persists BM25s index artifacts and ordered UUID mapping table to disk.
        """
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.retriever is not None and len(self.chunk_ids) > 0:
            self.retriever.save(str(dir_path))
            mapping_path = dir_path / "chunk_ids.json"
            mapping_path.write_text(
                json.dumps([str(uid) for uid in self.chunk_ids]),
                encoding="utf-8",
            )
            logger.info(
                "Persisted BM25s lexical index",
                directory=str(dir_path),
                chunk_count=len(self.chunk_ids),
            )

    @classmethod
    def load(
        cls,
        index_dir: Path | str,
        config: LexicalRetrievalConfig | None = None,
    ) -> LexicalIndex:
        """
        Loads a persisted BM25s index and its ordered UUID mapping from disk.
        """
        dir_path = Path(index_dir)
        mapping_path = dir_path / "chunk_ids.json"

        if not dir_path.exists() or not mapping_path.exists():
            return cls(config=config)

        chunk_ids = [UUID(uid) for uid in json.loads(mapping_path.read_text(encoding="utf-8"))]
        retriever = bm25s.BM25.load(str(dir_path), load_corpus=False)

        logger.info(
            "Loaded BM25s lexical index from disk",
            directory=str(dir_path),
            chunk_count=len(chunk_ids),
        )

        return cls(config=config, retriever=retriever, chunk_ids=chunk_ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[UUID, float]]:
        """
        Executes BM25 lexical search against the indexed child chunks.
        Returns ordered list of (chunk_id, score) pairs.
        """
        if self.is_empty or self.retriever is None:
            return []

        query_tokens = tokenize_query(query, config=self.config)
        k = min(top_k, len(self.chunk_ids))
        raw_results = self.retriever.retrieve(query_tokens, k=k, show_progress=False)
        if not isinstance(raw_results, tuple):
            return []

        doc_matrix, score_matrix = raw_results
        doc_indices = doc_matrix[0]
        scores = score_matrix[0]

        output: list[tuple[UUID, float]] = []
        for doc_idx, score in zip(doc_indices, scores, strict=True):
            chunk_uuid = self.chunk_ids[int(doc_idx)]
            output.append((chunk_uuid, float(score)))

        return output
