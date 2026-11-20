"""
Unit and integration tests for DuckDB database connection and lifecycle.
Verifies connection lifecycle, resource cleanup, schema initialization, and health checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragnarok.core.config import load_config
from ragnarok.storage.database import DatabasePool


def test_database_pool_lifecycle(tmp_path: Path) -> None:
    """Verifies that DatabasePool handles context management and clean closure."""
    db_file = tmp_path / "lifecycle.duckdb"
    pool = DatabasePool(
        db_path=str(db_file),
        min_connections=1,
        max_connections=5,
    )
    assert not pool.is_closed
    pool.close()
    assert pool.is_closed

    # Trying to acquire a connection from closed pool raises RuntimeError
    with pytest.raises(RuntimeError, match="DatabasePool is closed"), pool.connection():
        pass


def test_database_pool_context_manager(tmp_path: Path) -> None:
    """Verifies with-statement lifecycle of DatabasePool."""
    db_file = tmp_path / "ctx.duckdb"
    with DatabasePool(db_path=str(db_file)) as pool:
        assert not pool.is_closed
    assert pool.is_closed


def test_unreachable_database_health_check() -> None:
    """Verifies that an unopenable database returns False without raising unhandled exceptions."""
    pool = DatabasePool(
        db_path="/proc/sys/nonexistent_dir_forbidden/ragnarok.duckdb",
        timeout_seconds=1.0,
    )
    assert pool.health_check() is False


def test_init_schema_missing_file_raises_error(tmp_path: Path) -> None:
    """Verifies that init_schema with a non-existent file path raises FileNotFoundError."""
    db_file = tmp_path / "missing.duckdb"
    pool = DatabasePool(db_path=str(db_file))
    with pytest.raises(FileNotFoundError):
        pool.init_schema("nonexistent_schema.sql")


def test_live_database_connection_and_schema(tmp_path: Path) -> None:
    """Tests DuckDB connection, table bootstrap, and health check."""
    db_file = tmp_path / "test_live.duckdb"
    schema_path = Path("scripts/init_schema.sql")

    with DatabasePool(db_path=str(db_file)) as pool:
        # Initialize schema
        pool.init_schema(schema_path)
        # Verify health check finds all 4 tables
        assert pool.health_check() is True
