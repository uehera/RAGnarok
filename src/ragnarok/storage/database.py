"""
DuckDB Embedded Storage Infrastructure.

Provides connection management, schema initialization, and health diagnostics
using embedded DuckDB with strict resource management and thread safety.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb


class DuckDBCursorWrapper:
    """
    Wrapper around DuckDB connection maintaining DBAPI cursor compatibility
    on the exact same connection to preserve transaction boundaries.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, lock: threading.RLock) -> None:
        self._conn = conn
        self._lock = lock
        self._res: duckdb.DuckDBPyConnection | None = None

    def execute(self, query: Any, parameters: Any = None) -> DuckDBCursorWrapper:
        with self._lock:
            q_str = query.as_string(None) if hasattr(query, "as_string") else str(query)
            clean_query = q_str.replace("%s", "?")
            if parameters is not None:
                self._res = self._conn.execute(clean_query, parameters)
            else:
                self._res = self._conn.execute(clean_query)
            return self

    def fetchone(self) -> Any:
        with self._lock:
            if self._res is not None:
                return self._res.fetchone()
            return self._conn.fetchone()

    def fetchall(self) -> list[Any]:
        with self._lock:
            if self._res is not None:
                return self._res.fetchall()
            return self._conn.fetchall()

    @property
    def rowcount(self) -> int:
        return self._conn.rowcount

    def close(self) -> None:
        pass


class DuckDBConnectionWrapper:
    """Wrapper around DuckDB connection providing context-managed transactions."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, lock: threading.RLock) -> None:
        self._conn = conn
        self._lock = lock

    @contextmanager
    def transaction(self) -> Generator[DuckDBConnectionWrapper, None, None]:
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION;")
            try:
                yield self
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise

    @contextmanager
    def cursor(self) -> Generator[DuckDBCursorWrapper, None, None]:
        with self._lock:
            wrapper = DuckDBCursorWrapper(self._conn, self._lock)
            try:
                yield wrapper
            finally:
                wrapper.close()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._conn.execute(*args, **kwargs)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        pass


class DatabasePool:
    """
    Stateful service managing DuckDB connection lifecycle.
    Encapsulates connection parameters and provides safe context-managed sessions.
    """

    def __init__(
        self,
        database_url: str = "data/db/ragnarok.duckdb",
        min_connections: int = 1,
        max_connections: int = 10,
        timeout_seconds: float = 30.0,
        db_path: str | None = None,
    ) -> None:
        raw = db_path or database_url
        if raw.startswith("postgresql://") or raw.startswith("postgres://"):
            self.db_path = "data/db/ragnarok.duckdb"
        else:
            self.db_path = raw

        self.database_url = self.db_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.timeout_seconds = timeout_seconds
        self._is_closed: bool = False
        self._lock = threading.RLock()
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def is_closed(self) -> bool:
        """Returns True if the connection pool has been closed."""
        return self._is_closed

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        with self._lock:
            if self._conn is None:
                if self.db_path != ":memory:":
                    Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                self._conn = duckdb.connect(self.db_path)
            return self._conn

    @contextmanager
    def connection(self) -> Generator[DuckDBConnectionWrapper, None, None]:
        """
        Context manager yielding an active connection wrapper.
        Ensures thread-safe access and resource cleanup.
        """
        if self._is_closed:
            raise RuntimeError("DatabasePool is closed")

        raw_conn = self._get_connection()
        wrapper = DuckDBConnectionWrapper(raw_conn, self._lock)
        yield wrapper

    def health_check(self) -> bool:
        """
        Verifies DuckDB storage and checks that all 4 required metadata tables exist.
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
                    WHERE table_name IN ('documents', 'parent_chunks', 'chunks', 'processing_jobs');
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
            cur.execute(sql_content)

    def close(self) -> None:
        """Closes the connection manager."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
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
