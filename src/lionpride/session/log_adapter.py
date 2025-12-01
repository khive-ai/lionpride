# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Async log adapters for SQLite WAL and PostgreSQL backends."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import BaseModel, Field

from lionpride.ln import json_dumps

if TYPE_CHECKING:
    from .logs import Log

__all__ = (
    "LogAdapter",
    "LogAdapterConfig",
    "PostgresLogAdapter",
    "SQLiteWALLogAdapter",
)


# SQL identifier validation pattern
# Valid: alphanumeric, underscore, must start with letter or underscore
# Max 63 chars (PostgreSQL limit)
_SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


def _validate_sql_identifier(name: str, param_name: str = "identifier") -> str:
    """Validate SQL identifier to prevent SQL injection.

    Args:
        name: The identifier to validate (table name, column name, etc.)
        param_name: Name of the parameter for error messages

    Returns:
        The validated identifier

    Raises:
        ValueError: If identifier is invalid or potentially dangerous
    """
    if not name:
        raise ValueError(f"{param_name} cannot be empty")

    if not _SQL_IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"Invalid {param_name} '{name}': must start with letter or underscore, "
            "contain only alphanumeric/underscore characters, max 63 chars"
        )

    # Additional safety: reject SQL keywords (common ones)
    sql_keywords = {
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "truncate",
        "grant",
        "revoke",
        "union",
        "where",
        "from",
        "join",
        "table",
        "index",
        "database",
        "schema",
        "exec",
        "execute",
    }
    if name.lower() in sql_keywords:
        raise ValueError(f"Invalid {param_name} '{name}': cannot use SQL keyword")

    return name


logger = logging.getLogger(__name__)


class LogAdapterConfig(BaseModel):
    """Configuration for log adapters."""

    # SQLite settings
    sqlite_path: str | Path = "./data/logs.db"
    sqlite_wal_mode: bool = True

    # PostgreSQL settings
    postgres_dsn: str | None = None
    postgres_table: str = "logs"

    # Common settings
    batch_size: int = Field(default=100, ge=1, le=1000)
    auto_create_table: bool = True


class LogAdapter(ABC):
    """Abstract base class for log storage adapters.

    Adapters provide async read/write interface for log persistence.
    """

    @abstractmethod
    async def write(self, logs: list[Log]) -> int:
        """Write logs to storage.

        Args:
            logs: List of Log objects to persist

        Returns:
            Number of logs written
        """
        pass

    @abstractmethod
    async def read(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        log_type: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read logs from storage.

        Args:
            limit: Maximum logs to return
            offset: Skip this many logs
            log_type: Filter by log type
            since: Filter by created_at >= since (ISO format)

        Returns:
            List of log dicts
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the adapter and release resources."""
        pass


class SQLiteWALLogAdapter(LogAdapter):
    """SQLite WAL-mode adapter for local log storage.

    Uses Write-Ahead Logging for better concurrent access.
    Requires aiosqlite (optional dependency).

    Schema is general-purpose: stores full log as JSON content
    with indexed fields for common queries.
    """

    def __init__(
        self,
        db_path: str | Path = "./data/logs.db",
        wal_mode: bool = True,
        auto_create: bool = True,
    ):
        self.db_path = Path(db_path)
        self.wal_mode = wal_mode
        self.auto_create = auto_create
        self._connection = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure database and table exist."""
        if self._initialized:
            return

        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite is required for SQLiteWALLogAdapter. Install with: pip install aiosqlite"
            )

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self._connection = await aiosqlite.connect(str(self.db_path))

        # Enable WAL mode
        if self.wal_mode:
            await self._connection.execute("PRAGMA journal_mode=WAL")

        # Create table if needed - general schema with JSON content
        if self.auto_create:
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    log_type TEXT NOT NULL,
                    source TEXT,
                    content TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at)"
            )
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_type ON logs(log_type)"
            )
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_source ON logs(source)"
            )
            await self._connection.commit()

        self._initialized = True

    async def write(self, logs: list[Log]) -> int:
        """Write logs to SQLite."""
        await self._ensure_initialized()

        if not logs:
            return 0

        count = 0
        for log in logs:
            log_dict = log.to_dict(mode="json")
            # Store full log as JSON content
            content = json_dumps(log_dict)
            metadata = json_dumps(log_dict.get("metadata", {}))

            await self._connection.execute(
                """
                INSERT OR REPLACE INTO logs
                (id, created_at, log_type, source, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(log.id),
                    log_dict.get("created_at"),
                    log.log_type.value if hasattr(log.log_type, "value") else log.log_type,
                    log.source,
                    content,
                    metadata,
                ),
            )
            count += 1

        await self._connection.commit()
        logger.debug(f"Wrote {count} logs to SQLite WAL")
        return count

    async def read(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        log_type: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read logs from SQLite."""
        await self._ensure_initialized()

        query = "SELECT content FROM logs WHERE 1=1"
        params: list[Any] = []

        if log_type:
            query += " AND log_type = ?"
            params.append(log_type)

        if since:
            query += " AND created_at >= ?"
            params.append(since)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            try:
                log_dict = orjson.loads(row[0])
                results.append(log_dict)
            except (orjson.JSONDecodeError, TypeError):
                pass

        return results

    async def close(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False


class PostgresLogAdapter(LogAdapter):
    """PostgreSQL adapter for production log storage.

    Uses JSONB content column for flexible log storage.
    Indexed fields for common queries (created_at, log_type, source).
    """

    def __init__(
        self,
        dsn: str,
        table: str = "logs",
        auto_create: bool = True,
    ):
        self.dsn = dsn
        # Validate table name to prevent SQL injection
        self.table = _validate_sql_identifier(table, "table name")
        self.auto_create = auto_create
        self._initialized = False
        self._engine = None

    async def _ensure_initialized(self) -> None:
        """Ensure table exists."""
        if self._initialized:
            return

        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine
        except ImportError:
            raise ImportError(
                "sqlalchemy[asyncio] and asyncpg are required for PostgresLogAdapter. "
                "Install with: pip install 'sqlalchemy[asyncio]' asyncpg"
            )

        # Convert DSN to asyncpg format if needed
        dsn = self.dsn
        if dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://")

        self._engine = create_async_engine(dsn)

        if self.auto_create:
            async with self._engine.begin() as conn:
                # General schema with JSONB content
                await conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        id UUID PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        log_type VARCHAR(50) NOT NULL,
                        source TEXT,
                        content JSONB NOT NULL,
                        metadata JSONB
                    )
                    """)
                )
                await conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.table}_created_at ON {self.table}(created_at)"
                    )
                )
                await conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.table}_type ON {self.table}(log_type)"
                    )
                )
                await conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.table}_source ON {self.table}(source)"
                    )
                )
                # GIN index for JSONB content queries
                await conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.table}_content ON {self.table} USING GIN (content)"
                    )
                )

        self._initialized = True

    async def write(self, logs: list[Log]) -> int:
        """Write logs to PostgreSQL."""
        await self._ensure_initialized()

        if not logs:
            return 0

        try:
            from sqlalchemy import text

            async with self._engine.begin() as conn:
                for log in logs:
                    log_dict = log.to_dict(mode="json")
                    content = json_dumps(log_dict)
                    metadata = json_dumps(log_dict.get("metadata", {}))

                    await conn.execute(
                        text(f"""
                            INSERT INTO {self.table} (id, created_at, log_type, source, content, metadata)
                            VALUES (:id, :created_at, :log_type, :source, CAST(:content AS jsonb), CAST(:metadata AS jsonb))
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata
                        """),
                        {
                            "id": str(log.id),
                            "created_at": log.created_at,  # Use actual datetime, not string
                            "log_type": log.log_type.value
                            if hasattr(log.log_type, "value")
                            else log.log_type,
                            "source": log.source,
                            "content": content,
                            "metadata": metadata,
                        },
                    )

            logger.debug(f"Wrote {len(logs)} logs to PostgreSQL")
            return len(logs)

        except Exception as e:
            logger.error(f"Failed to write logs to PostgreSQL: {e}")
            raise

    async def read(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        log_type: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read logs from PostgreSQL."""
        await self._ensure_initialized()

        try:
            from sqlalchemy import text

            # Build query with parameters
            conditions = ["1=1"]
            params: dict[str, Any] = {"limit": limit, "offset": offset}

            if log_type:
                conditions.append("log_type = :log_type")
                params["log_type"] = log_type

            if since:
                conditions.append("created_at >= :since")
                # asyncpg requires datetime objects, not ISO strings
                if isinstance(since, str):
                    params["since"] = datetime.fromisoformat(since)
                else:
                    params["since"] = since

            where_clause = " AND ".join(conditions)
            query = text(f"""
                SELECT content FROM {self.table}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)

            async with self._engine.connect() as conn:
                result = await conn.execute(query, params)
                rows = result.fetchall()

            results = []
            for row in rows:
                content = row[0]
                if isinstance(content, str):
                    content = orjson.loads(content)
                results.append(content)

            return results

        except Exception as e:
            logger.error(f"Failed to read logs from PostgreSQL: {e}")
            raise

    async def close(self) -> None:
        """Close PostgreSQL engine."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._initialized = False
