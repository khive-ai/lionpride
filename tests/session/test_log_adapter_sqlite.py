# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for SQLiteWALLogAdapter with real aiosqlite.

Requires: pip install aiosqlite
Run with: pytest tests/session/test_log_adapter_sqlite.py -v
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lionpride.session import Log, LogType
from lionpride.session.log_adapter import _validate_sql_identifier

# Skip all tests if aiosqlite is not installed
pytest.importorskip("aiosqlite")

from lionpride.session.log_adapter import SQLiteWALLogAdapter

# -----------------------------------------------------------------------------
# SQL Identifier Validation Tests
# -----------------------------------------------------------------------------


class TestSQLIdentifierValidation:
    """Tests for SQL identifier validation (security)."""

    def test_valid_simple_name(self):
        """Simple alphanumeric names should be valid."""
        assert _validate_sql_identifier("logs") == "logs"
        assert _validate_sql_identifier("my_logs") == "my_logs"
        assert _validate_sql_identifier("logs_2024") == "logs_2024"
        assert _validate_sql_identifier("_private") == "_private"

    def test_valid_mixed_case(self):
        """Mixed case names should be valid."""
        assert _validate_sql_identifier("MyLogs") == "MyLogs"
        assert _validate_sql_identifier("API_Logs") == "API_Logs"

    def test_invalid_empty(self):
        """Empty name should be rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_sql_identifier("")

    def test_invalid_starts_with_number(self):
        """Names starting with numbers should be rejected."""
        with pytest.raises(ValueError, match="must start with letter"):
            _validate_sql_identifier("123logs")

    def test_invalid_special_characters(self):
        """Names with special characters should be rejected."""
        with pytest.raises(ValueError, match="must start with letter"):
            _validate_sql_identifier("logs; DROP TABLE users;--")

        with pytest.raises(ValueError, match="must start with letter"):
            _validate_sql_identifier("logs-table")

        with pytest.raises(ValueError, match="must start with letter"):
            _validate_sql_identifier("logs.table")

    def test_invalid_sql_keywords(self):
        """SQL keywords should be rejected."""
        with pytest.raises(ValueError, match="cannot use SQL keyword"):
            _validate_sql_identifier("select")

        with pytest.raises(ValueError, match="cannot use SQL keyword"):
            _validate_sql_identifier("DROP")

        with pytest.raises(ValueError, match="cannot use SQL keyword"):
            _validate_sql_identifier("TABLE")

    def test_invalid_too_long(self):
        """Names over 63 chars should be rejected."""
        long_name = "a" * 64
        with pytest.raises(ValueError, match="max 63 chars"):
            _validate_sql_identifier(long_name)

    def test_valid_max_length(self):
        """Names of exactly 63 chars should be valid."""
        name = "a" * 63
        assert _validate_sql_identifier(name) == name


@pytest.mark.integration
class TestSQLiteWALLogAdapterBasics:
    """Basic functionality tests for SQLiteWALLogAdapter (requires aiosqlite)."""

    @pytest.mark.asyncio
    async def test_adapter_creates_database_and_table(self, tmp_path: Path):
        """Adapter should create database file and logs table."""
        db_path = tmp_path / "test_logs.db"

        adapter = SQLiteWALLogAdapter(db_path=db_path, wal_mode=True, auto_create=True)

        # Write a log to trigger initialization
        log = Log(log_type=LogType.INFO, source="test", message="init test")
        count = await adapter.write([log])

        assert count == 1
        assert db_path.exists()

        # Verify table exists by reading
        logs = await adapter.read(limit=10)
        assert len(logs) == 1

        await adapter.close()

    @pytest.mark.asyncio
    async def test_adapter_creates_parent_directories(self, tmp_path: Path):
        """Adapter should create parent directories if they don't exist."""
        db_path = tmp_path / "subdir" / "nested" / "logs.db"

        adapter = SQLiteWALLogAdapter(db_path=db_path)
        log = Log(log_type=LogType.INFO, message="test")
        await adapter.write([log])

        assert db_path.exists()
        assert db_path.parent.exists()

        await adapter.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_path: Path):
        """WAL mode should be enabled when wal_mode=True."""
        import aiosqlite

        db_path = tmp_path / "wal_test.db"

        adapter = SQLiteWALLogAdapter(db_path=db_path, wal_mode=True)
        log = Log(log_type=LogType.INFO, message="wal test")
        await adapter.write([log])

        # Connect directly and check journal mode
        async with aiosqlite.connect(str(db_path)) as conn:
            cursor = await conn.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0].lower() == "wal"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_wal_mode_disabled(self, tmp_path: Path):
        """WAL mode should not be enabled when wal_mode=False."""
        import aiosqlite

        db_path = tmp_path / "no_wal_test.db"

        adapter = SQLiteWALLogAdapter(db_path=db_path, wal_mode=False)
        log = Log(log_type=LogType.INFO, message="no wal test")
        await adapter.write([log])

        # Connect directly and check journal mode (should be delete or another default)
        async with aiosqlite.connect(str(db_path)) as conn:
            cursor = await conn.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            # Without explicit WAL pragma, SQLite defaults to "delete" mode
            assert row[0].lower() != "wal"

        await adapter.close()


@pytest.mark.integration
class TestSQLiteWALLogAdapterWrite:
    """Write operation tests for SQLiteWALLogAdapter (requires aiosqlite)."""

    @pytest.mark.asyncio
    async def test_write_single_log(self, tmp_path: Path):
        """Should write a single log to database."""
        db_path = tmp_path / "write_single.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_write_multiple_logs(self, tmp_path: Path):
        """Should write multiple logs in a single batch."""
        db_path = tmp_path / "write_batch.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        logs = [
            Log(log_type=LogType.INFO, source="src1", message=f"message_{i}") for i in range(10)
        ]

        count = await adapter.write(logs)
        assert count == 10

        # Verify all logs were written
        result = await adapter.read(limit=20)
        assert len(result) == 10

        await adapter.close()

    @pytest.mark.asyncio
    async def test_write_empty_list(self, tmp_path: Path):
        """Writing empty list should return 0."""
        db_path = tmp_path / "write_empty.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        count = await adapter.write([])
        assert count == 0

        await adapter.close()

    @pytest.mark.asyncio
    async def test_write_preserves_all_fields(self, tmp_path: Path):
        """All log fields should be preserved when written and read back."""
        db_path = tmp_path / "write_fields.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_write_upsert_behavior(self, tmp_path: Path):
        """Writing log with same ID should update (upsert)."""
        db_path = tmp_path / "write_upsert.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        log = Log(log_type=LogType.INFO, source="test", message="original")
        await adapter.write([log])

        # Create new log with same ID but different content
        # (In real usage, we'd modify the log but Log is immutable after from_dict)
        # Instead, test that writing same log twice doesn't duplicate
        await adapter.write([log])

        result = await adapter.read(limit=10)
        # Should have only 1 entry (upsert)
        assert len(result) == 1

        await adapter.close()


@pytest.mark.integration
class TestSQLiteWALLogAdapterRead:
    """Read operation tests for SQLiteWALLogAdapter (requires aiosqlite)."""

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tmp_path: Path):
        """Should respect limit parameter."""
        db_path = tmp_path / "read_limit.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        # Write 20 logs
        logs = [Log(log_type=LogType.INFO, message=f"msg_{i}") for i in range(20)]
        await adapter.write(logs)

        # Read with limit
        result = await adapter.read(limit=5)
        assert len(result) == 5

        await adapter.close()

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tmp_path: Path):
        """Should respect offset parameter."""
        db_path = tmp_path / "read_offset.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        # Write logs with identifiable messages
        logs = [Log(log_type=LogType.INFO, message=f"msg_{i:02d}") for i in range(10)]
        await adapter.write(logs)

        # Read all first to see the order
        all_logs = await adapter.read(limit=100)
        assert len(all_logs) == 10

        # Read with offset (results are ordered DESC by created_at)
        result = await adapter.read(limit=3, offset=2)
        assert len(result) == 3

        await adapter.close()

    @pytest.mark.asyncio
    async def test_read_filter_by_log_type(self, tmp_path: Path):
        """Should filter by log_type."""
        db_path = tmp_path / "read_type.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_read_filter_by_since(self, tmp_path: Path):
        """Should filter by since (created_at >= since)."""
        db_path = tmp_path / "read_since.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_read_combined_filters(self, tmp_path: Path):
        """Should combine multiple filters."""
        db_path = tmp_path / "read_combined.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_read_empty_database(self, tmp_path: Path):
        """Reading from empty database should return empty list."""
        db_path = tmp_path / "read_empty.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        result = await adapter.read(limit=100)
        assert result == []

        await adapter.close()


@pytest.mark.integration
class TestSQLiteWALLogAdapterLifecycle:
    """Lifecycle and resource management tests (requires aiosqlite)."""

    @pytest.mark.asyncio
    async def test_close_releases_connection(self, tmp_path: Path):
        """Close should release database connection."""
        db_path = tmp_path / "close_test.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        # Initialize
        await adapter.write([Log(log_type=LogType.INFO, message="test")])
        assert adapter._connection is not None
        assert adapter._initialized is True

        # Close
        await adapter.close()
        assert adapter._connection is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_reconnect_after_close(self, tmp_path: Path):
        """Should be able to reconnect after close."""
        db_path = tmp_path / "reconnect.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        # First use
        await adapter.write([Log(log_type=LogType.INFO, message="first")])
        await adapter.close()

        # Reconnect and use again
        await adapter.write([Log(log_type=LogType.INFO, message="second")])
        result = await adapter.read(limit=10)

        assert len(result) == 2

        await adapter.close()

    @pytest.mark.asyncio
    async def test_multiple_adapters_same_database(self, tmp_path: Path):
        """Multiple adapters can share same database (WAL mode)."""
        db_path = tmp_path / "shared.db"

        adapter1 = SQLiteWALLogAdapter(db_path=db_path, wal_mode=True)
        adapter2 = SQLiteWALLogAdapter(db_path=db_path, wal_mode=True)

        # Both write
        await adapter1.write([Log(log_type=LogType.INFO, message="from_1")])
        await adapter2.write([Log(log_type=LogType.INFO, message="from_2")])

        # Both should see all logs
        result1 = await adapter1.read(limit=10)
        result2 = await adapter2.read(limit=10)

        assert len(result1) == 2
        assert len(result2) == 2

        await adapter1.close()
        await adapter2.close()


@pytest.mark.integration
class TestSQLiteWALLogAdapterEdgeCases:
    """Edge cases and error handling tests (requires aiosqlite)."""

    @pytest.mark.asyncio
    async def test_write_log_with_special_characters(self, tmp_path: Path):
        """Should handle special characters in log content."""
        db_path = tmp_path / "special_chars.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_write_log_with_large_content(self, tmp_path: Path):
        """Should handle large content in logs."""
        db_path = tmp_path / "large_content.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        large_text = "x" * 10_000  # 10KB text (reduced for speed)
        log = Log(log_type=LogType.INFO, message=large_text)

        count = await adapter.write([log])
        assert count == 1

        result = await adapter.read(limit=1)
        assert len(result) == 1
        assert len(result[0]["message"]) == 10_000

        await adapter.close()

    @pytest.mark.asyncio
    async def test_auto_create_false_no_table(self, tmp_path: Path):
        """When auto_create=False and table doesn't exist, should fail."""
        db_path = tmp_path / "no_auto.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path, auto_create=False)

        log = Log(log_type=LogType.INFO, message="test")

        # Should fail because table doesn't exist
        import sqlite3

        with pytest.raises(sqlite3.OperationalError):
            await adapter.write([log])

        await adapter.close()

    @pytest.mark.asyncio
    async def test_write_all_log_types(self, tmp_path: Path):
        """Should handle all log types correctly."""
        db_path = tmp_path / "all_types.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

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

        await adapter.close()


@pytest.mark.integration
class TestSQLiteLogAdapterCorruptData:
    """Tests for SQLite adapter handling of corrupt/invalid JSON data.

    Covers log_adapter.py lines 288-289 (json decode error handling).
    """

    @pytest.mark.asyncio
    async def test_read_handles_corrupt_json_silently(self, tmp_path: Path):
        """Corrupt JSON rows should be skipped without raising errors."""
        import aiosqlite

        db_path = tmp_path / "corrupt_data.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        # First write a valid log to create the table
        valid_log = Log(log_type=LogType.INFO, source="valid", message="valid message")
        await adapter.write([valid_log])

        # Directly insert corrupt JSON into the database
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute(
                """
                INSERT INTO logs (id, created_at, log_type, source, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "corrupt-id-123",
                    datetime.now(UTC).isoformat(),
                    "info",
                    "corrupt_source",
                    "not valid json {{{",  # Invalid JSON
                    "{}",
                ),
            )
            await conn.commit()

        # Read should succeed but skip the corrupt row
        result = await adapter.read(limit=10)

        # Should only return the valid log (corrupt one is skipped)
        assert len(result) == 1
        assert result[0]["message"] == "valid message"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_read_handles_empty_json_object(self, tmp_path: Path):
        """Empty JSON object rows should be returned as empty dicts."""
        import aiosqlite

        db_path = tmp_path / "empty_json.db"
        adapter = SQLiteWALLogAdapter(db_path=db_path)

        # First write a valid log to create the table
        valid_log = Log(log_type=LogType.INFO, source="valid", message="valid message")
        await adapter.write([valid_log])

        # Directly insert empty JSON object
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute(
                """
                INSERT INTO logs (id, created_at, log_type, source, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "empty-id-456",
                    datetime.now(UTC).isoformat(),
                    "info",
                    "empty_source",
                    "{}",  # Valid empty JSON object
                    "{}",
                ),
            )
            await conn.commit()

        # Read should succeed and include both logs
        result = await adapter.read(limit=10)

        # Should return both logs
        assert len(result) == 2

        await adapter.close()
