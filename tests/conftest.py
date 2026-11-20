"""
Shared Pytest Fixtures and Test Harness Setup for RAGnarok.

Provides isolated database repository connections, cleanup tracking,
and temporary directory management.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path
from uuid import UUID

import pytest

from ragnarok.core.config import load_config
from ragnarok.storage.database import DatabasePool
from ragnarok.storage.repository import RelationalRepository


def is_db_reachable() -> bool:
    """Verifies that the DuckDB database is reachable (always True for embedded engine)."""
    return True


is_postgres_reachable = is_db_reachable


@pytest.fixture(scope="session")
def live_db_pool(tmp_path_factory: pytest.TempPathFactory) -> Generator[DatabasePool, None, None]:
    """Provides a shared DatabasePool for integration tests using an isolated DuckDB file."""
    test_db_dir = tmp_path_factory.mktemp("db")
    test_db_path = str(test_db_dir / "test_ragnarok.duckdb")
    with DatabasePool(db_path=test_db_path) as pool:
        pool.init_schema(Path("scripts/init_schema.sql"))
        yield pool


@pytest.fixture
def repo(live_db_pool: DatabasePool) -> Generator[RelationalRepository, None, None]:
    """
    Yields a RelationalRepository instance and cleans up any documents
    registered during the test execution.
    """
    created_doc_ids: list[UUID] = []
    repository = RelationalRepository(live_db_pool)

    # Wrap insert_document to track document IDs for automatic teardown
    original_insert = repository.insert_document

    def tracking_insert(doc):
        inserted = original_insert(doc)
        created_doc_ids.append(inserted.document_id)
        return inserted

    repository.insert_document = tracking_insert  # type: ignore

    try:
        yield repository
    finally:
        for doc_id in created_doc_ids:
            with contextlib.suppress(Exception):
                repository.delete_document(doc_id)


@pytest.fixture
def isolated_paths(tmp_path: Path) -> dict[str, Path]:
    """Creates isolated staging, raw, and parsed directories for test runs."""
    staging = tmp_path / "staging"
    raw = tmp_path / "raw"
    parsed = tmp_path / "parsed"
    staging.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    parsed.mkdir(parents=True, exist_ok=True)
    return {
        "staging": staging,
        "raw": raw,
        "parsed": parsed,
    }
