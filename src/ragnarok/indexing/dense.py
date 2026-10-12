"""
Dense Vector Indexing Subsystem (TurboVec).

Generates semantic embeddings via sentence-transformers, enforces dimension validation,
maintains an injective bidirectional UUID-to-uint64 ID translation table, and interfaces
with TurboVec's IdMapIndex for similarity search and metadata allowlists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import numpy as np
import structlog
from turbovec import IdMapIndex

from ragnarok.core.config import DenseRetrievalConfig
from ragnarok.core.models import ChildChunk

logger = structlog.get_logger(__name__)


class VectorIdMap:
    """
    Injective bidirectional mapping between PostgreSQL UUIDs and 64-bit unsigned integer IDs.
    Guarantees loss-free translation between relational primary keys and TurboVec internal handles.
    """

    def __init__(
        self,
        uuid_to_id: dict[UUID, int] | None = None,
        next_id: int = 1,
    ) -> None:
        self.uuid_to_id: dict[UUID, int] = uuid_to_id if uuid_to_id is not None else {}
        self.id_to_uuid: dict[int, UUID] = {v: k for k, v in self.uuid_to_id.items()}
        self.next_id: int = next_id

    def __len__(self) -> int:
        return len(self.uuid_to_id)

    def register(self, chunk_uuid: UUID) -> int:
        """Assigns or retrieves an existing uint64 handle for a chunk UUID."""
        if chunk_uuid in self.uuid_to_id:
            return self.uuid_to_id[chunk_uuid]

        assigned_id = self.next_id
        self.next_id += 1
        self.uuid_to_id[chunk_uuid] = assigned_id
        self.id_to_uuid[assigned_id] = chunk_uuid
        return assigned_id

    def get_uint64(self, chunk_uuid: UUID) -> int | None:
        """Translates a UUID to its uint64 handle."""
        return self.uuid_to_id.get(chunk_uuid)

    def get_handle(self, chunk_uuid: UUID) -> int | None:
        """Translates a UUID to its uint64 handle (alias for get_uint64)."""
        return self.get_uint64(chunk_uuid)

    def add(self, chunk_uuid: UUID) -> int:
        """Assigns or retrieves an existing uint64 handle for a chunk UUID (alias for register)."""
        return self.register(chunk_uuid)

    def get_uuid(self, uint64_id: int) -> UUID | None:
        """Translates a uint64 handle back to its PostgreSQL UUID."""
        return self.id_to_uuid.get(uint64_id)

    def save(self, path: Path | str) -> None:
        """Persists mapping table to a JSON file."""
        file_path = Path(path)
        payload = {
            "uuid_to_id": {str(k): v for k, v in self.uuid_to_id.items()},
            "next_id": self.next_id,
        }
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> VectorIdMap:
        """Loads mapping table from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            return cls()

        data = json.loads(file_path.read_text(encoding="utf-8"))
        uuid_to_id = {UUID(k): int(v) for k, v in data.get("uuid_to_id", {}).items()}
        next_id = int(data.get("next_id", 1))
        return cls(uuid_to_id=uuid_to_id, next_id=next_id)


class EmbedderProtocol(Protocol):
    """Protocol for embedding generators."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generates 2D float32 embeddings for a batch of strings."""
        ...


class SentenceTransformerEmbedder:
    """Default embedding generator using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(vectors, dtype=np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalizes vectors along axis 1 for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class DenseIndex:
    """
    Stateful service managing the TurboVec dense index lifecycle,
    embedding generation, dimension validation, and persistence.
    """

    def __init__(
        self,
        config: DenseRetrievalConfig | None = None,
        index: IdMapIndex | None = None,
        id_map: VectorIdMap | None = None,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self.config = config if config is not None else DenseRetrievalConfig()
        self.embedder = (
            embedder
            if embedder is not None
            else SentenceTransformerEmbedder(model_name=self.config.model)
        )
        self.id_map = id_map if id_map is not None else VectorIdMap()
        self.index = index if index is not None else IdMapIndex(bit_width=4)

    @property
    def is_empty(self) -> bool:
        """Returns True if the index contains zero vectors."""
        return len(self.index) == 0

    def build(self, chunks: list[ChildChunk]) -> None:
        """
        Builds or rebuilds the dense vector index from child chunk records:
        1. Embeds child texts in batches.
        2. Validates vector dimensions against config.
        3. Normalizes vectors if similarity is cosine.
        4. Maps chunk UUIDs to 64-bit numeric handles.
        5. Inserts into TurboVec IdMapIndex.
        """
        if not chunks:
            self.index = IdMapIndex(bit_width=4)
            self.id_map = VectorIdMap()
            return

        texts = [chunk.content for chunk in chunks]
        batch_size = self.config.batch_size

        all_vectors_list: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_vecs = self.embedder.embed_texts(batch_texts)

            # Validate vector dimensions
            if batch_vecs.shape[1] != self.config.vector_dimension:
                raise ValueError(
                    f"Generated embedding dimension {batch_vecs.shape[1]} does not match "
                    f"configured vector_dimension {self.config.vector_dimension}"
                )
            all_vectors_list.append(batch_vecs)

        vectors = np.concatenate(all_vectors_list, axis=0)

        # Apply cosine normalization if configured
        if self.config.similarity_metric.lower() == "cosine":
            vectors = l2_normalize(vectors)

        # Fresh index and handle mapping
        self.index = IdMapIndex(bit_width=4)
        self.id_map = VectorIdMap()

        handles = np.array(
            [self.id_map.register(chunk.chunk_id) for chunk in chunks],
            dtype=np.uint64,
        )

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        handles = np.ascontiguousarray(handles, dtype=np.uint64)

        self.index.add_with_ids(vectors, handles)

        logger.info(
            "Built TurboVec dense vector index",
            chunk_count=len(chunks),
            dimension=self.config.vector_dimension,
        )

    def save(self, index_dir: Path | str) -> None:
        """
        Persists TurboVec index file and ID mapping table to data/indexes/dense/.
        """
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "index.tvim"
        id_map_file = dir_path / "id_map.json"

        # Write TurboVec binary index
        self.index.write(str(index_file))
        # Write JSON sidecar ID mapping
        self.id_map.save(id_map_file)

        logger.info(
            "Persisted TurboVec dense index",
            directory=str(dir_path),
            vector_count=len(self.index),
        )

    @classmethod
    def load(
        cls,
        index_dir: Path | str,
        config: DenseRetrievalConfig | None = None,
        embedder: EmbedderProtocol | None = None,
    ) -> DenseIndex:
        """
        Loads a persisted TurboVec index and ID mapping from disk.
        """
        dir_path = Path(index_dir)
        index_file = dir_path / "index.tvim"
        id_map_file = dir_path / "id_map.json"

        if not index_file.exists() or not id_map_file.exists():
            return cls(config=config, embedder=embedder)

        index = IdMapIndex.load(str(index_file))
        id_map = VectorIdMap.load(id_map_file)

        logger.info(
            "Loaded TurboVec dense index from disk",
            directory=str(dir_path),
            vector_count=len(index),
        )

        return cls(config=config, index=index, id_map=id_map, embedder=embedder)

    def search(
        self,
        query: str,
        top_k: int = 10,
        allowlist: list[UUID] | None = None,
    ) -> list[tuple[UUID, float]]:
        """
        Executes dense vector similarity search:
        - Embeds and normalizes query text.
        - Translates allowlist UUIDs to 64-bit integer IDs if filtering is applied.
        - If allowlist is empty or zero candidates match, returns [] without vector search.
        - Performs TurboVec search and translates returned handles back to UUIDs.
        """
        if self.is_empty:
            return []

        qvec = self.embedder.embed_texts([query])
        if qvec.shape[1] != self.config.vector_dimension:
            raise ValueError(
                f"Query vector dimension {qvec.shape[1]} does not match "
                f"configured vector_dimension {self.config.vector_dimension}"
            )

        if self.config.similarity_metric.lower() == "cosine":
            qvec = l2_normalize(qvec)

        qvec = np.ascontiguousarray(qvec, dtype=np.float32)

        if allowlist is not None:
            # Resolve candidate handles
            valid_handles = [
                self.id_map.get_uint64(u)
                for u in allowlist
                if self.id_map.get_uint64(u) is not None
            ]
            if not valid_handles:
                # No chunks match metadata criteria: return empty list immediately
                return []

            allowlist_handles = np.ascontiguousarray(
                np.array(valid_handles, dtype=np.uint64)
            )
            search_k = min(top_k, len(valid_handles))
            scores, handles = self.index.search(
                qvec,
                search_k,
                allowlist=allowlist_handles,
            )
        else:
            search_k = min(top_k, len(self.index))
            scores, handles = self.index.search(qvec, search_k)

        output: list[tuple[UUID, float]] = []
        for score, handle in zip(scores[0], handles[0], strict=True):
            chunk_uuid = self.id_map.get_uuid(int(handle))
            if chunk_uuid is not None:
                output.append((chunk_uuid, float(score)))

        return output
