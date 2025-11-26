# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for logging system."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from lionpride.core import Element, Pile
from lionpride.session import Log, LogStore, LogStoreConfig, LogType


class TestLogType:
    """Tests for LogType enum."""

    def test_log_types_exist(self):
        """All expected log types should exist."""
        assert LogType.API_CALL.value == "api_call"
        assert LogType.MESSAGE.value == "message"
        assert LogType.OPERATION.value == "operation"
        assert LogType.ERROR.value == "error"
        assert LogType.WARNING.value == "warning"
        assert LogType.INFO.value == "info"


class TestLogStoreConfig:
    """Tests for LogStoreConfig."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = LogStoreConfig()
        assert config.persist_dir == "./data/logs"
        assert config.file_prefix is None
        assert config.capacity is None
        assert config.extension == ".json"
        assert config.use_timestamp is True
        assert config.auto_save_on_exit is True
        assert config.clear_after_dump is True

    def test_extension_validation(self):
        """Extension should be .json or .jsonl."""
        # Valid extensions
        config = LogStoreConfig(extension=".json")
        assert config.extension == ".json"

        config = LogStoreConfig(extension=".jsonl")
        assert config.extension == ".jsonl"

        config = LogStoreConfig(extension="json")  # auto-adds dot
        assert config.extension == ".json"

        # Invalid extension
        with pytest.raises(ValueError, match="Extension must be"):
            LogStoreConfig(extension=".csv")

    def test_capacity_validation(self):
        """Capacity must be non-negative."""
        config = LogStoreConfig(capacity=100)
        assert config.capacity == 100

        config = LogStoreConfig(capacity=0)
        assert config.capacity == 0

        with pytest.raises(ValueError, match="non-negative"):
            LogStoreConfig(capacity=-1)


class TestLog:
    """Tests for Log Element."""

    def test_log_extends_element(self):
        """Log should extend Element."""
        log = Log(log_type=LogType.INFO)
        assert isinstance(log, Element)
        assert isinstance(log.id, UUID)
        assert isinstance(log.created_at, datetime)

    def test_log_creation(self):
        """Test basic log creation."""
        log = Log(
            log_type=LogType.API_CALL,
            source="test",
            model="gpt-4",
            duration_ms=100.5,
            total_tokens=150,
        )
        assert log.log_type == LogType.API_CALL
        assert log.source == "test"
        assert log.model == "gpt-4"
        assert log.duration_ms == 100.5
        assert log.total_tokens == 150

    def test_log_serialization(self):
        """Log should serialize to dict and back."""
        log = Log(
            log_type=LogType.ERROR,
            source="test_source",
            error="Something went wrong",
            data={"key": "value"},
        )

        # Serialize
        data = log.to_dict(mode="json")
        assert "id" in data
        assert "created_at" in data
        assert data["log_type"] == "error"
        assert data["source"] == "test_source"
        assert data["error"] == "Something went wrong"
        assert data["data"] == {"key": "value"}

        # Deserialize
        restored = Log.from_dict(data)
        assert restored.id == log.id
        assert restored.log_type == log.log_type
        assert restored.source == log.source
        assert restored.error == log.error

    def test_log_immutability_after_from_dict(self):
        """Log should be immutable after from_dict."""
        log = Log(log_type=LogType.INFO, message="test")
        data = log.to_dict(mode="json")

        restored = Log.from_dict(data)
        with pytest.raises(AttributeError, match="immutable"):
            restored.message = "changed"

    def test_log_create_from_element(self):
        """Log.create should wrap Element in a log."""
        element = Element()
        log = Log.create(element, log_type=LogType.MESSAGE)

        assert log.log_type == LogType.MESSAGE
        assert log.data is not None
        assert "id" in log.data

    def test_log_create_from_dict(self):
        """Log.create should wrap dict in a log."""
        data = {"key": "value", "nested": {"a": 1}}
        log = Log.create(data, log_type=LogType.INFO)

        assert log.log_type == LogType.INFO
        assert log.data == data

    def test_log_legacy_timestamp_field(self):
        """from_dict should handle legacy timestamp field."""
        data = {
            "id": "12345678-1234-5678-1234-567812345678",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "log_type": "info",
            "source": "",
        }
        log = Log.from_dict(data)
        assert log.created_at.year == 2025


class TestLogStore:
    """Tests for LogStore."""

    def test_logstore_uses_pile(self):
        """LogStore should use Pile internally."""
        store = LogStore(auto_save_on_exit=False)
        assert isinstance(store.logs, Pile)

    def test_logstore_add_and_iterate(self):
        """Test adding logs and iterating."""
        store = LogStore(auto_save_on_exit=False)
        log1 = store.log_info(source="test", message="msg1")
        log2 = store.log_info(source="test", message="msg2")

        assert len(store) == 2

        logs = list(store)
        assert logs[0].id == log1.id
        assert logs[1].id == log2.id

    def test_logstore_uuid_lookup(self):
        """Test O(1) UUID lookup via Pile."""
        store = LogStore(auto_save_on_exit=False)
        log = store.log_info(source="test", message="lookup test")

        # Get by UUID
        retrieved = store[log.id]
        assert retrieved.id == log.id
        assert retrieved.message == "lookup test"

        # Get by string UUID
        retrieved = store[str(log.id)]
        assert retrieved.id == log.id

    def test_logstore_index_lookup(self):
        """Test index-based lookup."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(source="test", message="msg0")
        store.log_info(source="test", message="msg1")
        store.log_info(source="test", message="msg2")

        assert store[0].message == "msg0"
        assert store[1].message == "msg1"
        assert store[-1].message == "msg2"

    def test_log_api_call(self):
        """Test log_api_call convenience method."""
        store = LogStore(auto_save_on_exit=False)
        log = store.log_api_call(
            source="session_1",
            model="gpt-4",
            provider="openai",
            request={"messages": []},
            response={"choices": []},
            duration_ms=150.5,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        assert log.log_type == LogType.API_CALL
        assert log.model == "gpt-4"
        assert log.provider == "openai"
        assert log.duration_ms == 150.5
        assert log.total_tokens == 150

    def test_log_operation(self):
        """Test log_operation convenience method."""
        store = LogStore(auto_save_on_exit=False)
        log = store.log_operation(
            source="branch_1", message="Operation completed", data={"items": 5}
        )

        assert log.log_type == LogType.OPERATION
        assert log.message == "Operation completed"
        assert log.data == {"items": 5}

    def test_log_error(self):
        """Test log_error convenience method."""
        store = LogStore(auto_save_on_exit=False)
        log = store.log_error(
            source="validator", error="Validation failed", data={"field": "email"}
        )

        assert log.log_type == LogType.ERROR
        assert log.error == "Validation failed"

    def test_log_info(self):
        """Test log_info convenience method."""
        store = LogStore(auto_save_on_exit=False)
        log = store.log_info(source="system", message="System initialized")

        assert log.log_type == LogType.INFO
        assert log.message == "System initialized"

    def test_filter_by_type(self):
        """Test filtering logs by type."""
        store = LogStore(auto_save_on_exit=False)
        store.log_api_call(model="gpt-4")
        store.log_api_call(model="claude")
        store.log_error(error="error1")
        store.log_info(message="info1")

        api_calls = store.filter(log_type=LogType.API_CALL)
        assert len(api_calls) == 2

        errors = store.filter(log_type=LogType.ERROR)
        assert len(errors) == 1

    def test_filter_by_source(self):
        """Test filtering logs by source."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(source="branch_1", message="msg1")
        store.log_info(source="branch_2", message="msg2")
        store.log_info(source="branch_1", message="msg3")

        branch_1_logs = store.filter(source="branch_1")
        assert len(branch_1_logs) == 2

    def test_filter_by_time_range(self):
        """Test filtering logs by time range."""
        store = LogStore(auto_save_on_exit=False)
        now = datetime.now(UTC)

        store.log_info(message="msg1")
        store.log_info(message="msg2")

        # Filter since slightly before now
        logs = store.filter(since=now - timedelta(seconds=1))
        assert len(logs) == 2

        # Filter with future since - should get nothing
        logs = store.filter(since=now + timedelta(hours=1))
        assert len(logs) == 0

    def test_filter_by_model(self):
        """Test filtering logs by model."""
        store = LogStore(auto_save_on_exit=False)
        store.log_api_call(model="gpt-4")
        store.log_api_call(model="gpt-4-turbo")
        store.log_api_call(model="claude-3")

        gpt_logs = store.filter(model="gpt-4")
        assert len(gpt_logs) == 2  # matches gpt-4 and gpt-4-turbo

    def test_get_api_calls(self):
        """Test get_api_calls convenience method."""
        store = LogStore(auto_save_on_exit=False)
        store.log_api_call(model="gpt-4")
        store.log_info(message="not api call")

        api_calls = store.get_api_calls()
        assert len(api_calls) == 1
        assert api_calls[0].model == "gpt-4"

    def test_get_errors(self):
        """Test get_errors convenience method."""
        store = LogStore(auto_save_on_exit=False)
        store.log_error(error="error1")
        store.log_error(error="error2")
        store.log_info(message="not error")

        errors = store.get_errors()
        assert len(errors) == 2

    def test_to_list(self):
        """Test exporting logs to list of dicts."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(message="msg1")
        store.log_error(error="error1")

        logs_list = store.to_list()
        assert len(logs_list) == 2
        assert isinstance(logs_list[0], dict)
        assert "id" in logs_list[0]

    def test_clear(self):
        """Test clearing logs."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(message="msg1")
        store.log_info(message="msg2")
        assert len(store) == 2

        count = store.clear()
        assert count == 2
        assert len(store) == 0

    def test_summary(self):
        """Test summary statistics."""
        store = LogStore(auto_save_on_exit=False)
        store.log_api_call(model="gpt-4", total_tokens=100, duration_ms=50)
        store.log_api_call(model="gpt-4", total_tokens=200, duration_ms=100)
        store.log_api_call(model="claude", total_tokens=150, duration_ms=75)
        store.log_error(error="error1")
        store.log_info(message="info1")

        summary = store.summary()
        assert summary["total_logs"] == 5
        assert summary["api_calls"] == 3
        assert summary["errors"] == 1
        assert summary["total_tokens"] == 450
        assert summary["total_duration_ms"] == 225
        assert set(summary["models_used"]) == {"gpt-4", "claude"}

    def test_dump_to_file(self):
        """Test dumping logs to file."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(source="test", message="dump test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "logs.json"
            count = store.dump(path=path, clear=False)

            assert count == 1
            assert path.exists()
            assert len(store) == 1  # Not cleared

            # Dump with clear
            count = store.dump(path=path, clear=True)
            assert len(store) == 0  # Cleared

    def test_dump_creates_directory(self):
        """Test that dump creates parent directory if needed."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(message="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "logs.json"
            count = store.dump(path=path)

            assert count == 1
            assert path.exists()

    def test_config_with_max_logs_override(self):
        """Test that max_logs parameter overrides config capacity."""
        config = LogStoreConfig(capacity=1000, auto_save_on_exit=False)
        store = LogStore(max_logs=50, config=config)

        assert store._config.capacity == 50

    def test_repr(self):
        """Test string representation."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(message="test")
        assert "LogStore(logs=1)" in repr(store)


class TestLogStoreAsync:
    """Async tests for LogStore."""

    @pytest.mark.asyncio
    async def test_alog(self):
        """Test async logging."""
        store = LogStore(auto_save_on_exit=False)
        log = Log(log_type=LogType.INFO, message="async test")

        await store.alog(log)

        assert len(store) == 1
        assert store[0].message == "async test"

    @pytest.mark.asyncio
    async def test_alog_with_content(self):
        """Test async logging with arbitrary content."""
        store = LogStore(auto_save_on_exit=False)

        await store.alog({"key": "value"})

        assert len(store) == 1
        assert store[0].data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_adump(self):
        """Test async dump."""
        store = LogStore(auto_save_on_exit=False)
        store.log_info(message="async dump test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "logs.json"
            count = await store.adump(path=path, clear=True)

            assert count == 1
            assert path.exists()
            assert len(store) == 0

    @pytest.mark.asyncio
    async def test_concurrent_alog(self):
        """Test concurrent async logging is safe."""
        store = LogStore(auto_save_on_exit=False)

        async def log_many(prefix: str, count: int):
            for i in range(count):
                log = Log(log_type=LogType.INFO, message=f"{prefix}_{i}")
                await store.alog(log)

        # Run concurrent logging
        await asyncio.gather(
            log_many("task_a", 10),
            log_many("task_b", 10),
            log_many("task_c", 10),
        )

        # All logs should be captured
        assert len(store) == 30


class TestLogStoreCapacity:
    """Tests for LogStore capacity handling."""

    def test_auto_dump_at_capacity(self):
        """Test that logs auto-dump when capacity is reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogStoreConfig(
                capacity=5,
                persist_dir=tmpdir,
                auto_save_on_exit=False,
                clear_after_dump=True,
            )
            store = LogStore(config=config)

            # Add logs up to capacity
            for i in range(5):
                store.log_info(message=f"msg_{i}")

            assert len(store) == 5

            # Adding one more should trigger dump
            store.log_info(message="overflow")

            # After dump and clear, only the new log should remain
            assert len(store) == 1
            assert store[0].message == "overflow"

            # Check that dump file was created
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1
