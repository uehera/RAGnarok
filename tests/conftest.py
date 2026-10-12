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


def is_postgres_reachable() -> bool:
    """Verifies that the live PostgreSQL container is reachable."""
    try:
        cfg = load_config()
        with (
            DatabasePool(database_url=cfg.secrets.database_url, timeout_seconds=2.0) as pool,
            pool.connection() as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1;")
            row = cur.fetchone()
            return row is not None and row[0] == 1
    except Exception:
        return False


@pytest.fixture(scope="session")
def live_db_pool() -> Generator[DatabasePool, None, None]:
    """Provides a shared DatabasePool for integration tests if PostgreSQL is online."""
    if not is_postgres_reachable():
        pytest.skip("PostgreSQL container is not reachable at localhost:5432")

    cfg = load_config()
    with DatabasePool(database_url=cfg.secrets.database_url) as pool:
        # Ensure schema is applied
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
