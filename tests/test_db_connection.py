"""
Unit and integration tests for PostgreSQL database connection and lifecycle.
Verifies connection lifecycle, resource cleanup, schema initialization, and health checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragnarok.core.config import load_config
from ragnarok.storage.database import DatabasePool


def test_database_pool_lifecycle() -> None:
    """Verifies that DatabasePool handles context management and clean closure."""
    pool = DatabasePool(
        database_url="postgresql://postgres:postgres@localhost:5432/ragnarok",
        min_connections=1,
        max_connections=5,
    )
    assert not pool.is_closed
    pool.close()
    assert pool.is_closed

    # Trying to acquire a connection from closed pool raises RuntimeError
    with pytest.raises(RuntimeError, match="DatabasePool is closed"), pool.connection():
        pass

def test_database_pool_context_manager() -> None:
    """Verifies with-statement lifecycle of DatabasePool."""
    with DatabasePool(
        database_url="postgresql://postgres:postgres@localhost:5432/ragnarok"
    ) as pool:
        assert not pool.is_closed
    assert pool.is_closed

def test_unreachable_database_health_check() -> None:
    """Verifies that an unreachable database returns False without raising unhandled exceptions."""
    # Use invalid port where no DB is running
    pool = DatabasePool(
        database_url="postgresql://postgres:postgres@localhost:59999/nonexistent",
        timeout_seconds=1.0,
    )
    assert pool.health_check() is False

def test_init_schema_missing_file_raises_error() -> None:
    """Verifies that init_schema with a non-existent file path raises FileNotFoundError."""
    pool = DatabasePool(
        database_url="postgresql://postgres:postgres@localhost:5432/ragnarok"
    )
    with pytest.raises(FileNotFoundError):
        pool.init_schema("nonexistent_schema.sql")

def _is_postgres_reachable() -> bool:
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


@pytest.mark.skipif(
    not _is_postgres_reachable(),
    reason="PostgreSQL container is not reachable",
)
def test_live_database_connection_and_schema() -> None:
    """Tests live PostgreSQL connection, table bootstrap, and health check."""
    config = load_config()
    schema_path = Path("scripts/init_schema.sql")

    with DatabasePool(database_url=config.secrets.database_url) as pool:
        # Initialize schema
        pool.init_schema(schema_path)
        # Verify health check finds all 4 tables
        assert pool.health_check() is True
