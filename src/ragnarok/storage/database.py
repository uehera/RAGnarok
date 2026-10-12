"""
PostgreSQL Storage Infrastructure.

Provides connection management, schema initialization, and health diagnostics
using Psycopg 3 with strict resource management and explicit dependency injection.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, LiteralString, cast

import psycopg


class DatabasePool:
    """
    Stateful service managing PostgreSQL connection lifecycle.
    Encapsulates connection parameters and provides safe context-managed sessions.
    """

    def __init__(
        self,
        database_url: str,
        min_connections: int = 1,
        max_connections: int = 10,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.timeout_seconds = timeout_seconds
        self._is_closed: bool = False

    @property
    def is_closed(self) -> bool:
        """Returns True if the connection pool has been closed."""
        return self._is_closed

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection[Any], None, None]:
        """
        Context manager yielding an active psycopg connection.
        Ensures proper resource cleanup and automatic transaction management.
        """
        if self._is_closed:
            raise RuntimeError("DatabasePool is closed")

        with psycopg.connect(
            self.database_url, connect_timeout=int(self.timeout_seconds)
        ) as conn:
            yield conn

    def health_check(self) -> bool:
        """
        Pings PostgreSQL and verifies that all 4 required metadata tables exist.
        """
        try:
            with self.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1;")
                row = cur.fetchone()
                if not row or row[0] != 1:
                    return False

                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('documents', 'parent_chunks', 'chunks', 'processing_jobs');
                    """
                )
                tables = [r[0] for r in cur.fetchall()]
                return len(tables) == 4
        except Exception:
            return False

    def init_schema(self, schema_sql_path: Path | str) -> None:
        """
        Applies SQL DDL to initialize schema tables, unique constraints, and indexes.
        """
        path = Path(schema_sql_path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found at: {path}")

        sql_content = path.read_text(encoding="utf-8")
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(cast(LiteralString, sql_content))
            conn.commit()

    def close(self) -> None:
        """Closes the connection manager."""
        self._is_closed = True

    def __enter__(self) -> DatabasePool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()
