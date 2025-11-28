# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for PostgresLogAdapter.

Unit tests (mocked):
    pytest tests/session/test_log_adapter_postgres.py -v -k "not integration"

Integration tests (require Docker + testcontainers):
    pip install 'sqlalchemy[asyncio]' asyncpg 'testcontainers[postgres]'
    pytest tests/session/test_log_adapter_postgres.py -v -m integration
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lionpride.session import Log, LogType
from lionpride.session.log_adapter import PostgresLogAdapter

# Check if sqlalchemy is available for unit tests
try:
    import sqlalchemy

    del sqlalchemy  # Only checking availability
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# -----------------------------------------------------------------------------
# Table name validation tests (don't need testcontainers)
# -----------------------------------------------------------------------------


class TestPostgresLogAdapterValidation:
    """Tests for PostgresLogAdapter table name validation."""

    def test_valid_table_name(self):
        """Valid table names should work."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test", table="my_logs")
        assert adapter.table == "my_logs"

    def test_default_table_name(self):
        """Default table name should be 'logs'."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        assert adapter.table == "logs"

    def test_invalid_table_name_sql_injection(self):
        """SQL injection attempts should be rejected."""
        with pytest.raises(ValueError, match="must start with letter"):
            PostgresLogAdapter(dsn="postgresql://localhost/test", table="logs; DROP TABLE users;--")

    def test_invalid_table_name_keyword(self):
        """SQL keywords should be rejected."""
        with pytest.raises(ValueError, match="cannot use SQL keyword"):
            PostgresLogAdapter(dsn="postgresql://localhost/test", table="select")

    def test_invalid_table_name_special_chars(self):
        """Special characters should be rejected."""
        with pytest.raises(ValueError, match="must start with letter"):
            PostgresLogAdapter(dsn="postgresql://localhost/test", table="logs-table")


# -----------------------------------------------------------------------------
# Mock-based unit tests (no Docker required)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="sqlalchemy not available")
class TestPostgresLogAdapterMocked:
    """Unit tests for PostgresLogAdapter using mocks (requires sqlalchemy)."""

    def test_dsn_conversion_postgresql_to_asyncpg(self):
        """Should convert postgresql:// to postgresql+asyncpg://."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        assert adapter.dsn == "postgresql://localhost/test"

    def test_auto_create_default_true(self):
        """auto_create should default to True."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        assert adapter.auto_create is True

    def test_auto_create_false(self):
        """auto_create=False should be respected."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test", auto_create=False)
        assert adapter.auto_create is False

    @pytest.mark.asyncio
    async def test_write_empty_logs(self):
        """Writing empty list should return 0 without touching DB."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        # Should not require initialization for empty list
        adapter._initialized = True
        adapter._engine = None

        count = await adapter.write([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_close_without_engine(self):
        """Close should handle case when engine is None."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        assert adapter._engine is None

        # Should not raise
        await adapter.close()
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        """Close should reset initialized state."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True
        adapter._engine = MagicMock()
        adapter._engine.dispose = AsyncMock()

        await adapter.close()

        assert adapter._engine is None
        assert adapter._initialized is False

    def test_table_with_underscore_valid(self):
        """Table names with underscores should be valid."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test", table="my_app_logs")
        assert adapter.table == "my_app_logs"

    def test_table_starting_with_underscore_valid(self):
        """Table names starting with underscore should be valid."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test", table="_internal_logs")
        assert adapter.table == "_internal_logs"

    @pytest.mark.asyncio
    async def test_read_empty_database_mocked(self):
        """Read should return empty list when no logs exist."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        # Mock engine and connection
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        result = await adapter.read(limit=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_read_with_log_type_filter(self):
        """Read should filter by log_type."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        # Mock return data
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [({"log_type": "info", "message": "test"},)]

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        result = await adapter.read(log_type="info", limit=10)

        assert len(result) == 1
        assert result[0]["log_type"] == "info"

    @pytest.mark.asyncio
    async def test_read_with_since_filter_iso_string(self):
        """Read should convert ISO string to datetime for since filter."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        # Use ISO string
        since_str = "2025-01-01T00:00:00+00:00"

        await adapter.read(since=since_str, limit=10)

        # Verify the call was made (datetime conversion happens internally)
        assert mock_conn.execute.called
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        # The since param should be a datetime, not string
        assert isinstance(params["since"], datetime)


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="sqlalchemy not available")
class TestPostgresLogAdapterEnsureInitialized:
    """Tests for PostgresLogAdapter._ensure_initialized with mocks."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_skips_if_already_initialized(self):
        """Should skip initialization if already initialized."""
        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True
        adapter._engine = "fake_engine"

        # Should not raise even without real engine
        await adapter._ensure_initialized()

        # State unchanged
        assert adapter._initialized is True
        assert adapter._engine == "fake_engine"

    @pytest.mark.asyncio
    async def test_ensure_initialized_dsn_conversion(self):
        """Should convert postgresql:// to postgresql+asyncpg://."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        # Patch where it's imported from, not where it's used
        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ) as mock_create_engine:
            await adapter._ensure_initialized()

            # Should have converted DSN
            mock_create_engine.assert_called_once_with("postgresql+asyncpg://localhost/test")

        assert adapter._initialized is True
        assert adapter._engine is mock_engine

    @pytest.mark.asyncio
    async def test_ensure_initialized_dsn_already_asyncpg(self):
        """Should not double-convert if DSN already has asyncpg."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = PostgresLogAdapter(dsn="postgresql+asyncpg://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ) as mock_create_engine:
            await adapter._ensure_initialized()

            # Should use DSN as-is (no double conversion)
            mock_create_engine.assert_called_once_with("postgresql+asyncpg://localhost/test")

    @pytest.mark.asyncio
    async def test_ensure_initialized_auto_create_true_creates_table(self):
        """Should create table and indexes when auto_create=True."""
        from unittest.mock import AsyncMock, MagicMock, call, patch

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test", auto_create=True)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ):
            await adapter._ensure_initialized()

        # Should have called execute 5 times (table + 4 indexes)
        assert mock_conn.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_ensure_initialized_auto_create_false_skips_table(self):
        """Should skip table creation when auto_create=False."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test", auto_create=False)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock()

        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ):
            await adapter._ensure_initialized()

        # Should NOT have called begin (no table creation)
        mock_engine.begin.assert_not_called()
        assert adapter._initialized is True

    @pytest.mark.asyncio
    async def test_ensure_initialized_import_error(self):
        """Should raise ImportError with helpful message when sqlalchemy not available."""
        import builtins
        from unittest.mock import patch

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        # Store original import
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sqlalchemy" or name.startswith("sqlalchemy."):
                raise ImportError("No module named 'sqlalchemy'")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="sqlalchemy"),
        ):
            await adapter._ensure_initialized()


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="sqlalchemy not available")
class TestPostgresLogAdapterWriteMocked:
    """Tests for PostgresLogAdapter.write with mocks."""

    @pytest.mark.asyncio
    async def test_write_single_log_success(self):
        """Should write a single log and return count."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        # Create mock for write transaction
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        # Pre-initialize to skip _ensure_initialized
        adapter._initialized = True
        adapter._engine = mock_engine

        log = Log(log_type=LogType.INFO, source="test", message="test message")

        count = await adapter.write([log])

        assert count == 1
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_multiple_logs_success(self):
        """Should write multiple logs and return total count."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        adapter._initialized = True
        adapter._engine = mock_engine

        logs = [Log(log_type=LogType.INFO, message=f"msg_{i}") for i in range(5)]

        count = await adapter.write(logs)

        assert count == 5
        assert mock_conn.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_write_calls_ensure_initialized(self):
        """Write should call _ensure_initialized before processing."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        # Mock _ensure_initialized
        init_called = False

        async def mock_ensure():
            nonlocal init_called
            init_called = True
            adapter._initialized = True

            # Setup mock engine
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock()
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=None)

            mock_engine = MagicMock()
            mock_engine.begin = MagicMock(return_value=mock_conn)
            adapter._engine = mock_engine

        adapter._ensure_initialized = mock_ensure

        log = Log(log_type=LogType.INFO, message="test")
        await adapter.write([log])

        assert init_called is True

    @pytest.mark.asyncio
    async def test_write_with_log_type_enum(self):
        """Should handle log_type as enum (with .value)."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        adapter._initialized = True
        adapter._engine = mock_engine

        log = Log(log_type=LogType.API_CALL, model="gpt-4")

        await adapter.write([log])

        # Verify execute was called with correct params
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["log_type"] == "api_call"  # Enum converted to string

    @pytest.mark.asyncio
    async def test_write_exception_handling(self):
        """Should log error and re-raise on exception."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB connection failed"))
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)

        adapter._initialized = True
        adapter._engine = mock_engine

        log = Log(log_type=LogType.INFO, message="test")

        with pytest.raises(Exception, match="DB connection failed"):
            await adapter.write([log])


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="sqlalchemy not available")
class TestPostgresLogAdapterReadMocked:
    """Tests for PostgresLogAdapter.read with comprehensive mocks."""

    @pytest.mark.asyncio
    async def test_read_calls_ensure_initialized(self):
        """Read should call _ensure_initialized before processing."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")

        init_called = False

        async def mock_ensure():
            nonlocal init_called
            init_called = True
            adapter._initialized = True

            mock_result = MagicMock()
            mock_result.fetchall.return_value = []

            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock(return_value=mock_result)
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=None)

            mock_engine = MagicMock()
            mock_engine.connect = MagicMock(return_value=mock_conn)
            adapter._engine = mock_engine

        adapter._ensure_initialized = mock_ensure

        await adapter.read(limit=10)

        assert init_called is True

    @pytest.mark.asyncio
    async def test_read_with_string_content(self):
        """Read should parse JSON string content."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        # Return content as JSON string (not dict)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ('{"log_type": "info", "message": "test from string"}',),
        ]

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        result = await adapter.read(limit=10)

        assert len(result) == 1
        assert result[0]["log_type"] == "info"
        assert result[0]["message"] == "test from string"

    @pytest.mark.asyncio
    async def test_read_with_dict_content(self):
        """Read should handle dict content directly (no JSON parsing)."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        # Return content as dict (JSONB returns dict in asyncpg)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ({"log_type": "info", "message": "test from dict"},),
        ]

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        result = await adapter.read(limit=10)

        assert len(result) == 1
        assert result[0]["log_type"] == "info"
        assert result[0]["message"] == "test from dict"

    @pytest.mark.asyncio
    async def test_read_with_since_as_datetime(self):
        """Read should handle since as datetime (not string)."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        # Pass datetime directly (not string)
        since_dt = datetime(2025, 1, 1, tzinfo=UTC)

        await adapter.read(since=since_dt, limit=10)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        # Should pass datetime as-is
        assert params["since"] == since_dt

    @pytest.mark.asyncio
    async def test_read_exception_handling(self):
        """Should log error and re-raise on exception."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("Query failed"))
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        with pytest.raises(Exception, match="Query failed"):
            await adapter.read(limit=10)

    @pytest.mark.asyncio
    async def test_read_combined_log_type_and_since(self):
        """Should build correct query with both log_type and since filters."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ({"log_type": "api_call", "model": "gpt-4"},),
        ]

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        since_str = "2025-01-01T00:00:00+00:00"
        result = await adapter.read(log_type="api_call", since=since_str, limit=5, offset=2)

        assert len(result) == 1
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["log_type"] == "api_call"
        assert params["limit"] == 5
        assert params["offset"] == 2
        assert isinstance(params["since"], datetime)

    @pytest.mark.asyncio
    async def test_read_multiple_rows(self):
        """Should return multiple logs correctly."""
        from unittest.mock import AsyncMock, MagicMock

        adapter = PostgresLogAdapter(dsn="postgresql://localhost/test")
        adapter._initialized = True

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ({"log_type": "info", "message": "msg1"},),
            ({"log_type": "error", "error": "err1"},),
            ('{"log_type": "warning", "message": "warn1"}',),  # String content
        ]

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_result)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_conn)

        adapter._engine = mock_engine

        result = await adapter.read(limit=10)

        assert len(result) == 3
        assert result[0]["log_type"] == "info"
        assert result[1]["log_type"] == "error"
        assert result[2]["log_type"] == "warning"


# -----------------------------------------------------------------------------
# Integration tests (require testcontainers)
# -----------------------------------------------------------------------------

# Check if integration test dependencies are available
try:
    import asyncpg
    import sqlalchemy
    from testcontainers.postgres import PostgresContainer

    del asyncpg, sqlalchemy  # Only checking availability
    INTEGRATION_DEPS_AVAILABLE = True
except ImportError:
    INTEGRATION_DEPS_AVAILABLE = False
    PostgresContainer = None  # type: ignore


@pytest.fixture(scope="module")
def postgres_container():
    """Start a PostgreSQL container for the test module."""
    if not INTEGRATION_DEPS_AVAILABLE:
        pytest.skip("testcontainers, sqlalchemy, or asyncpg not available")
    with PostgresContainer("postgres:15-alpine") as postgres:
        yield postgres


@pytest.fixture
def postgres_dsn(postgres_container):
    """Get PostgreSQL DSN from container - convert to asyncpg format."""
    url = postgres_container.get_connection_url()
    # testcontainers returns psycopg2 URL, convert to asyncpg
    if "+psycopg2" in url:
        url = url.replace("+psycopg2", "+asyncpg")
    elif "postgresql://" in url and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    return url


@pytest.fixture
async def adapter(postgres_dsn):
    """Create a PostgresLogAdapter with unique table per test."""
    import uuid

    # Use unique table name per test to avoid conflicts
    table_name = f"logs_{uuid.uuid4().hex[:8]}"
    adapter = PostgresLogAdapter(dsn=postgres_dsn, table=table_name, auto_create=True)
    yield adapter
    await adapter.close()


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION_DEPS_AVAILABLE, reason="Integration deps not available")
class TestPostgresLogAdapterBasics:
    """Basic functionality tests for PostgresLogAdapter."""

    @pytest.mark.asyncio
    async def test_adapter_creates_table(self, postgres_dsn):
        """Adapter should create logs table if auto_create=True."""
        import uuid

        table_name = f"test_create_{uuid.uuid4().hex[:8]}"
        adapter = PostgresLogAdapter(dsn=postgres_dsn, table=table_name, auto_create=True)

        # Write to trigger initialization
        log = Log(log_type=LogType.INFO, source="test", message="init test")
        count = await adapter.write([log])
        assert count == 1

        # Verify table was created by reading
        logs = await adapter.read(limit=10)
        assert len(logs) == 1

        await adapter.close()

    @pytest.mark.asyncio
    async def test_dsn_format_conversion(self, postgres_dsn):
        """Should work with asyncpg DSN format."""
        import uuid

        table_name = f"test_dsn_{uuid.uuid4().hex[:8]}"

        # DSN fixture converts to asyncpg format
        assert "asyncpg" in postgres_dsn or postgres_dsn.startswith("postgresql://")

        adapter = PostgresLogAdapter(dsn=postgres_dsn, table=table_name)
        log = Log(log_type=LogType.INFO, message="dsn test")
        count = await adapter.write([log])

        assert count == 1

        await adapter.close()


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION_DEPS_AVAILABLE, reason="Integration deps not available")
class TestPostgresLogAdapterWrite:
    """Write operation tests for PostgresLogAdapter."""

    @pytest.mark.asyncio
    async def test_write_single_log(self, adapter):
        """Should write a single log to database."""
        log = Log(
            log_type=LogType.API_CALL,
            source="test_source",
            model="gpt-4",
            provider="openai",
            duration_ms=150.5,
            total_tokens=100,
        )

        count = await adapter.write([log])
        assert count == 1

        # Verify in database
        logs = await adapter.read(limit=10)
        assert len(logs) == 1
        assert logs[0]["model"] == "gpt-4"
        assert logs[0]["provider"] == "openai"
        assert logs[0]["duration_ms"] == 150.5

    @pytest.mark.asyncio
    async def test_write_multiple_logs(self, adapter):
        """Should write multiple logs in a single batch."""
        logs = [
            Log(log_type=LogType.INFO, source="src1", message=f"message_{i}") for i in range(10)
        ]

        count = await adapter.write(logs)
        assert count == 10

        # Verify all logs were written
        result = await adapter.read(limit=20)
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_write_empty_list(self, adapter):
        """Writing empty list should return 0."""
        count = await adapter.write([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_write_preserves_all_fields(self, adapter):
        """All log fields should be preserved when written and read back."""
        log = Log(
            log_type=LogType.API_CALL,
            source="session_123",
            model="claude-3",
            provider="anthropic",
            request={"messages": [{"role": "user", "content": "hello"}]},
            response={"choices": [{"message": {"content": "hi"}}]},
            duration_ms=200.5,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            message="API call completed",
            data={"custom_key": "custom_value"},
            error=None,
        )

        await adapter.write([log])
        result = await adapter.read(limit=1)

        assert len(result) == 1
        retrieved = result[0]

        assert retrieved["id"] == str(log.id)
        assert retrieved["log_type"] == "api_call"
        assert retrieved["source"] == "session_123"
        assert retrieved["model"] == "claude-3"
        assert retrieved["provider"] == "anthropic"
        assert retrieved["request"] == {"messages": [{"role": "user", "content": "hello"}]}
        assert retrieved["response"] == {"choices": [{"message": {"content": "hi"}}]}
        assert retrieved["duration_ms"] == 200.5
        assert retrieved["input_tokens"] == 10
        assert retrieved["output_tokens"] == 5
        assert retrieved["total_tokens"] == 15
        assert retrieved["message"] == "API call completed"
        assert retrieved["data"] == {"custom_key": "custom_value"}

    @pytest.mark.asyncio
    async def test_write_upsert_behavior(self, adapter):
        """Writing log with same ID should update (upsert)."""
        log = Log(log_type=LogType.INFO, source="test", message="original")
        await adapter.write([log])

        # Write same log again (upsert)
        await adapter.write([log])

        result = await adapter.read(limit=10)
        # Should have only 1 entry (upsert)
        assert len(result) == 1


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION_DEPS_AVAILABLE, reason="Integration deps not available")
class TestPostgresLogAdapterRead:
    """Read operation tests for PostgresLogAdapter."""

    @pytest.mark.asyncio
    async def test_read_with_limit(self, adapter):
        """Should respect limit parameter."""
        # Write 20 logs
        logs = [Log(log_type=LogType.INFO, message=f"msg_{i}") for i in range(20)]
        await adapter.write(logs)

        # Read with limit
        result = await adapter.read(limit=5)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_read_with_offset(self, adapter):
        """Should respect offset parameter."""
        # Write logs
        logs = [Log(log_type=LogType.INFO, message=f"msg_{i:02d}") for i in range(10)]
        await adapter.write(logs)

        # Read with offset
        result = await adapter.read(limit=3, offset=2)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_read_filter_by_log_type(self, adapter):
        """Should filter by log_type."""
        # Write mixed log types
        logs = [
            Log(log_type=LogType.INFO, message="info1"),
            Log(log_type=LogType.ERROR, error="error1"),
            Log(log_type=LogType.API_CALL, model="gpt-4"),
            Log(log_type=LogType.INFO, message="info2"),
            Log(log_type=LogType.ERROR, error="error2"),
        ]
        await adapter.write(logs)

        # Filter by INFO
        result = await adapter.read(log_type="info")
        assert len(result) == 2
        assert all(r["log_type"] == "info" for r in result)

        # Filter by ERROR
        result = await adapter.read(log_type="error")
        assert len(result) == 2
        assert all(r["log_type"] == "error" for r in result)

        # Filter by API_CALL
        result = await adapter.read(log_type="api_call")
        assert len(result) == 1
        assert result[0]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_read_filter_by_since(self, adapter):
        """Should filter by since (created_at >= since)."""
        # Write a log
        log = Log(log_type=LogType.INFO, message="recent")
        await adapter.write([log])

        # Filter with since in the past - should find log
        past = datetime.now(UTC) - timedelta(hours=1)
        result = await adapter.read(since=past.isoformat())
        assert len(result) == 1

        # Filter with since in the future - should find nothing
        future = datetime.now(UTC) + timedelta(hours=1)
        result = await adapter.read(since=future.isoformat())
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_read_combined_filters(self, adapter):
        """Should combine multiple filters."""
        # Write various logs
        logs = [
            Log(log_type=LogType.INFO, message="info1"),
            Log(log_type=LogType.ERROR, error="error1"),
            Log(log_type=LogType.INFO, message="info2"),
        ]
        await adapter.write(logs)

        past = datetime.now(UTC) - timedelta(hours=1)

        # Combine log_type and since
        result = await adapter.read(log_type="info", since=past.isoformat())
        assert len(result) == 2

        # With limit
        result = await adapter.read(log_type="info", since=past.isoformat(), limit=1)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_read_empty_database(self, adapter):
        """Reading from empty database should return empty list."""
        result = await adapter.read(limit=100)
        assert result == []


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION_DEPS_AVAILABLE, reason="Integration deps not available")
class TestPostgresLogAdapterLifecycle:
    """Lifecycle and resource management tests."""

    @pytest.mark.asyncio
    async def test_close_releases_engine(self, postgres_dsn):
        """Close should dispose engine."""
        import uuid

        table_name = f"test_close_{uuid.uuid4().hex[:8]}"
        adapter = PostgresLogAdapter(dsn=postgres_dsn, table=table_name)

        # Initialize
        await adapter.write([Log(log_type=LogType.INFO, message="test")])
        assert adapter._engine is not None
        assert adapter._initialized is True

        # Close
        await adapter.close()
        assert adapter._engine is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_reconnect_after_close(self, postgres_dsn):
        """Should be able to reconnect after close."""
        import uuid

        table_name = f"test_reconnect_{uuid.uuid4().hex[:8]}"
        adapter = PostgresLogAdapter(dsn=postgres_dsn, table=table_name)

        # First use
        await adapter.write([Log(log_type=LogType.INFO, message="first")])
        await adapter.close()

        # Reconnect and use again
        await adapter.write([Log(log_type=LogType.INFO, message="second")])
        result = await adapter.read(limit=10)

        assert len(result) == 2

        await adapter.close()


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION_DEPS_AVAILABLE, reason="Integration deps not available")
class TestPostgresLogAdapterEdgeCases:
    """Edge cases and error handling tests."""

    @pytest.mark.asyncio
    async def test_write_log_with_special_characters(self, adapter):
        """Should handle special characters in log content."""
        log = Log(
            log_type=LogType.INFO,
            source="test",
            message="Special: 'quotes', \"double\", \\backslash, \nnewline, \ttab, unicode: éàñ",
            data={"emoji": "😀", "chinese": "中文"},
        )

        count = await adapter.write([log])
        assert count == 1

        result = await adapter.read(limit=1)
        assert len(result) == 1
        assert "newline" in result[0]["message"]
        assert result[0]["data"]["emoji"] == "😀"
        assert result[0]["data"]["chinese"] == "中文"

    @pytest.mark.asyncio
    async def test_write_log_with_large_content(self, adapter):
        """Should handle large content in logs."""
        large_text = "x" * 10_000  # 10KB text (reduced for speed)
        log = Log(log_type=LogType.INFO, message=large_text)

        count = await adapter.write([log])
        assert count == 1

        result = await adapter.read(limit=1)
        assert len(result) == 1
        assert len(result[0]["message"]) == 10_000

    @pytest.mark.asyncio
    async def test_write_all_log_types(self, adapter):
        """Should handle all log types correctly."""
        logs = [
            Log(log_type=LogType.API_CALL, model="test"),
            Log(log_type=LogType.MESSAGE, message="msg"),
            Log(log_type=LogType.OPERATION, message="op"),
            Log(log_type=LogType.ERROR, error="err"),
            Log(log_type=LogType.WARNING, message="warn"),
            Log(log_type=LogType.INFO, message="info"),
        ]

        count = await adapter.write(logs)
        assert count == 6

        # Verify each type can be filtered
        for log_type in LogType:
            result = await adapter.read(log_type=log_type.value)
            assert len(result) == 1, f"Expected 1 log of type {log_type.value}"

    @pytest.mark.asyncio
    async def test_jsonb_content_query(self, adapter):
        """JSONB content should be queryable (basic verification)."""
        # Write log with complex nested data
        log = Log(
            log_type=LogType.API_CALL,
            model="gpt-4",
            request={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            response={
                "choices": [{"message": {"content": "Hi there!", "role": "assistant"}}],
                "usage": {"total_tokens": 50},
            },
        )

        await adapter.write([log])

        result = await adapter.read(limit=1)
        assert len(result) == 1

        # Verify nested structure preserved
        assert result[0]["request"]["messages"][0]["role"] == "system"
        assert result[0]["response"]["usage"]["total_tokens"] == 50
