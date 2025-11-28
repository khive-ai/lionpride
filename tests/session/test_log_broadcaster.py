# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for LogBroadcaster and LogSubscribers.

Unit tests run without external dependencies.
Integration tests (marked @pytest.mark.integration) require Docker and optional deps.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lionpride.session import Log, LogType
from lionpride.session.log_broadcaster import (
    LogBroadcaster,
    LogBroadcasterConfig,
    LogSubscriber,
    PostgresLogSubscriber,
    S3LogSubscriber,
    WebhookLogSubscriber,
)

# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


class MockSubscriber(LogSubscriber):
    """Mock subscriber for testing."""

    def __init__(
        self,
        name: str = "mock",
        fail: bool = False,
        delay: float = 0.0,
    ):
        self._name = name
        self.fail = fail
        self.delay = delay
        self.received_logs: list[Log] = []
        self.receive_count = 0
        self.close_called = False

    @property
    def name(self) -> str:
        return self._name

    async def receive(self, logs: list[Log]) -> int:
        if self.delay > 0:
            import asyncio

            await asyncio.sleep(self.delay)

        self.receive_count += 1

        if self.fail:
            raise RuntimeError(f"Subscriber {self._name} failed")

        self.received_logs.extend(logs)
        return len(logs)

    async def close(self) -> None:
        self.close_called = True


# -----------------------------------------------------------------------------
# LogBroadcasterConfig tests
# -----------------------------------------------------------------------------


class TestLogBroadcasterConfig:
    """Tests for LogBroadcasterConfig."""

    def test_default_config(self):
        """Default config values."""
        config = LogBroadcasterConfig()
        assert config.fail_fast is False
        assert config.parallel is True


# -----------------------------------------------------------------------------
# LogBroadcaster tests
# -----------------------------------------------------------------------------


class TestLogBroadcaster:
    """Tests for LogBroadcaster."""

    def test_add_subscriber(self):
        """Should add subscriber."""
        broadcaster = LogBroadcaster()
        sub = MockSubscriber(name="test_sub")

        broadcaster.add_subscriber(sub)

        assert "test_sub" in broadcaster.list_subscribers()

    def test_add_duplicate_subscriber_replaces(self):
        """Adding subscriber with same name should replace."""
        broadcaster = LogBroadcaster()
        sub1 = MockSubscriber(name="test_sub")
        sub2 = MockSubscriber(name="test_sub")

        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        # Should only have one subscriber
        assert len(broadcaster.list_subscribers()) == 1

    def test_remove_subscriber(self):
        """Should remove subscriber."""
        broadcaster = LogBroadcaster()
        sub = MockSubscriber(name="test_sub")

        broadcaster.add_subscriber(sub)
        result = broadcaster.remove_subscriber("test_sub")

        assert result is True
        assert "test_sub" not in broadcaster.list_subscribers()

    def test_remove_nonexistent_subscriber(self):
        """Removing nonexistent subscriber returns False."""
        broadcaster = LogBroadcaster()
        result = broadcaster.remove_subscriber("nonexistent")
        assert result is False

    def test_list_subscribers(self):
        """Should list all subscriber names."""
        broadcaster = LogBroadcaster()
        broadcaster.add_subscriber(MockSubscriber(name="sub1"))
        broadcaster.add_subscriber(MockSubscriber(name="sub2"))
        broadcaster.add_subscriber(MockSubscriber(name="sub3"))

        names = broadcaster.list_subscribers()
        assert set(names) == {"sub1", "sub2", "sub3"}

    @pytest.mark.asyncio
    async def test_broadcast_to_single_subscriber(self):
        """Should broadcast to single subscriber."""
        broadcaster = LogBroadcaster()
        sub = MockSubscriber(name="test_sub")
        broadcaster.add_subscriber(sub)

        logs = [
            Log(log_type=LogType.INFO, message="msg1"),
            Log(log_type=LogType.INFO, message="msg2"),
        ]

        results = await broadcaster.broadcast(logs)

        assert results == {"test_sub": 2}
        assert len(sub.received_logs) == 2

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self):
        """Should broadcast to all subscribers."""
        broadcaster = LogBroadcaster()
        sub1 = MockSubscriber(name="sub1")
        sub2 = MockSubscriber(name="sub2")
        sub3 = MockSubscriber(name="sub3")

        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)
        broadcaster.add_subscriber(sub3)

        logs = [Log(log_type=LogType.INFO, message="test")]

        results = await broadcaster.broadcast(logs)

        assert results == {"sub1": 1, "sub2": 1, "sub3": 1}
        assert len(sub1.received_logs) == 1
        assert len(sub2.received_logs) == 1
        assert len(sub3.received_logs) == 1

    @pytest.mark.asyncio
    async def test_broadcast_empty_list(self):
        """Broadcasting empty list should return empty results."""
        broadcaster = LogBroadcaster()
        sub = MockSubscriber(name="test_sub")
        broadcaster.add_subscriber(sub)

        results = await broadcaster.broadcast([])

        assert results == {}
        assert sub.receive_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers(self):
        """Broadcasting with no subscribers should return empty results."""
        broadcaster = LogBroadcaster()
        logs = [Log(log_type=LogType.INFO, message="test")]

        results = await broadcaster.broadcast(logs)

        assert results == {}

    @pytest.mark.asyncio
    async def test_broadcast_parallel(self):
        """Should broadcast in parallel by default."""
        config = LogBroadcasterConfig(parallel=True)
        broadcaster = LogBroadcaster(config=config)

        sub1 = MockSubscriber(name="sub1", delay=0.05)
        sub2 = MockSubscriber(name="sub2", delay=0.05)
        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        logs = [Log(log_type=LogType.INFO, message="test")]

        import time

        start = time.time()
        await broadcaster.broadcast(logs)
        elapsed = time.time() - start

        # Parallel should be faster than sequential (0.1s)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_broadcast_sequential(self):
        """Should broadcast sequentially when parallel=False."""
        config = LogBroadcasterConfig(parallel=False)
        broadcaster = LogBroadcaster(config=config)

        sub1 = MockSubscriber(name="sub1", delay=0.02)
        sub2 = MockSubscriber(name="sub2", delay=0.02)
        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        logs = [Log(log_type=LogType.INFO, message="test")]

        import time

        start = time.time()
        await broadcaster.broadcast(logs)
        elapsed = time.time() - start

        # Sequential should take at least 0.04s (2 x 0.02s)
        assert elapsed >= 0.03

    @pytest.mark.asyncio
    async def test_broadcast_failure_continues(self):
        """Failed subscriber should not stop other subscribers."""
        config = LogBroadcasterConfig(fail_fast=False)
        broadcaster = LogBroadcaster(config=config)

        sub1 = MockSubscriber(name="sub1")
        sub2 = MockSubscriber(name="sub2", fail=True)
        sub3 = MockSubscriber(name="sub3")

        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)
        broadcaster.add_subscriber(sub3)

        logs = [Log(log_type=LogType.INFO, message="test")]

        results = await broadcaster.broadcast(logs)

        # sub1 and sub3 should succeed, sub2 should fail (return 0)
        assert results["sub1"] == 1
        assert results["sub2"] == 0
        assert results["sub3"] == 1

    @pytest.mark.asyncio
    async def test_broadcast_fail_fast(self):
        """fail_fast=True should stop on first failure (sequential only)."""
        config = LogBroadcasterConfig(fail_fast=True, parallel=False)
        broadcaster = LogBroadcaster(config=config)

        sub1 = MockSubscriber(name="sub1")
        sub2 = MockSubscriber(name="sub2", fail=True)
        sub3 = MockSubscriber(name="sub3")

        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)
        broadcaster.add_subscriber(sub3)

        logs = [Log(log_type=LogType.INFO, message="test")]

        results = await broadcaster.broadcast(logs)

        # Should stop after sub2 fails
        assert results["sub1"] == 1
        assert results["sub2"] == 0
        # sub3 may or may not have been processed depending on dict order
        # In Python 3.7+, dict maintains insertion order

    @pytest.mark.asyncio
    async def test_close_all_subscribers(self):
        """close should close all subscribers."""
        broadcaster = LogBroadcaster()
        sub1 = MockSubscriber(name="sub1")
        sub2 = MockSubscriber(name="sub2")

        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        await broadcaster.close()

        assert sub1.close_called is True
        assert sub2.close_called is True
        assert len(broadcaster.list_subscribers()) == 0

    def test_repr(self):
        """repr should show subscribers."""
        broadcaster = LogBroadcaster()
        broadcaster.add_subscriber(MockSubscriber(name="sub1"))
        broadcaster.add_subscriber(MockSubscriber(name="sub2"))

        repr_str = repr(broadcaster)
        assert "sub1" in repr_str or "LogBroadcaster" in repr_str


# -----------------------------------------------------------------------------
# WebhookLogSubscriber tests
# -----------------------------------------------------------------------------


class TestWebhookLogSubscriber:
    """Tests for WebhookLogSubscriber."""

    def test_webhook_subscriber_name(self):
        """Name should include URL."""
        sub = WebhookLogSubscriber(url="https://example.com/logs")
        assert sub.name == "webhook:https://example.com/logs"

    def test_webhook_default_headers(self):
        """Default headers should include Content-Type."""
        sub = WebhookLogSubscriber(url="https://example.com/logs")
        assert sub.headers["Content-Type"] == "application/json"

    def test_webhook_custom_headers(self):
        """Should accept custom headers."""
        headers = {"Authorization": "Bearer token", "X-Custom": "value"}
        sub = WebhookLogSubscriber(url="https://example.com/logs", headers=headers)
        assert sub.headers["Authorization"] == "Bearer token"

    @pytest.mark.asyncio
    async def test_webhook_receive_success(self):
        """Should send logs to webhook endpoint."""
        import httpx

        sub = WebhookLogSubscriber(url="https://example.com/logs")

        # Patch httpx.AsyncClient to use mock transport
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=httpx.Response(200))
            mock_client_class.return_value = mock_client

            logs = [
                Log(log_type=LogType.INFO, message="msg1"),
                Log(log_type=LogType.INFO, message="msg2"),
            ]

            count = await sub.receive(logs)

            assert count == 2
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_receive_empty(self):
        """Empty logs should return 0 without calling endpoint."""
        sub = WebhookLogSubscriber(url="https://example.com/logs")

        count = await sub.receive([])

        assert count == 0

    @pytest.mark.asyncio
    async def test_webhook_receive_failure(self):
        """Failed request should return 0, not raise."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            import httpx

            mock_client.post = AsyncMock(return_value=httpx.Response(500))
            mock_client_class.return_value = mock_client

            sub = WebhookLogSubscriber(url="https://example.com/logs")
            logs = [Log(log_type=LogType.INFO, message="test")]

            count = await sub.receive(logs)

            assert count == 0

    @pytest.mark.asyncio
    async def test_webhook_receive_batch_size(self):
        """Should send in batches."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            import httpx

            mock_client.post = AsyncMock(return_value=httpx.Response(200))
            mock_client_class.return_value = mock_client

            sub = WebhookLogSubscriber(url="https://example.com/logs", batch_size=2)

            # 5 logs should result in 3 batches (2, 2, 1)
            logs = [Log(log_type=LogType.INFO, message=f"msg{i}") for i in range(5)]

            count = await sub.receive(logs)

            assert count == 5
            assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_webhook_close_no_op(self):
        """close should be a no-op (no persistent resources)."""
        sub = WebhookLogSubscriber(url="https://example.com/logs")
        await sub.close()  # Should not raise


# -----------------------------------------------------------------------------
# S3LogSubscriber tests (mocked)
# -----------------------------------------------------------------------------


class TestS3LogSubscriber:
    """Tests for S3LogSubscriber (mocked)."""

    def test_s3_subscriber_name(self):
        """Name should include bucket."""
        sub = S3LogSubscriber(bucket="my-logs")
        assert sub.name == "s3:my-logs"

    def test_s3_prefix_normalization(self):
        """Prefix should be normalized with trailing slash."""
        sub1 = S3LogSubscriber(bucket="test", prefix="logs")
        assert sub1.prefix == "logs/"

        sub2 = S3LogSubscriber(bucket="test", prefix="logs/")
        assert sub2.prefix == "logs/"

    @pytest.mark.asyncio
    async def test_s3_receive_empty(self):
        """Empty logs should return 0."""
        sub = S3LogSubscriber(bucket="test")
        count = await sub.receive([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_s3_receive_with_mocked_session(self):
        """Test S3 subscriber with mocked boto session."""
        sub = S3LogSubscriber(bucket="test", prefix="logs/")

        # Track what was uploaded
        uploaded = {}

        async def mock_put_object(**kwargs):
            uploaded["bucket"] = kwargs.get("Bucket")
            uploaded["key"] = kwargs.get("Key")
            uploaded["body"] = kwargs.get("Body")
            return {}

        # Create mock client
        mock_client = AsyncMock()
        mock_client.put_object = mock_put_object

        # Create async context manager
        class MockClientCM:
            async def __aenter__(self):
                return mock_client

            async def __aexit__(self, *args):
                pass

        # Mock session
        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=MockClientCM())

        # Patch _ensure_client to set our mock client instead of real one
        async def mock_ensure_client():
            sub._client = mock_session

        with patch.object(sub, "_ensure_client", mock_ensure_client):
            logs = [Log(log_type=LogType.INFO, message="test")]
            count = await sub.receive(logs)

        assert count == 1
        assert uploaded["bucket"] == "test"
        assert uploaded["key"].startswith("logs/")
        assert b"test" in uploaded["body"]

    @pytest.mark.asyncio
    async def test_s3_close_no_op(self):
        """close should clear client reference."""
        sub = S3LogSubscriber(bucket="test")
        sub._client = MagicMock()

        await sub.close()

        assert sub._client is None


# -----------------------------------------------------------------------------
# PostgresLogSubscriber tests
# -----------------------------------------------------------------------------


class TestPostgresLogSubscriber:
    """Tests for PostgresLogSubscriber."""

    def test_postgres_subscriber_name(self):
        """Name should include table."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test", table="my_logs")
        assert sub.name == "postgres:my_logs"

    def test_postgres_default_table(self):
        """Default table should be 'logs'."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test")
        assert sub.table == "logs"
        assert sub.name == "postgres:logs"

    @pytest.mark.asyncio
    async def test_postgres_receive_empty(self):
        """Empty logs should return 0."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test")
        count = await sub.receive([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_postgres_receive_delegates_to_adapter(self):
        """receive should delegate to PostgresLogAdapter."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test")

        # Mock the adapter
        mock_adapter = AsyncMock()
        mock_adapter.write = AsyncMock(return_value=2)
        sub._adapter = mock_adapter

        logs = [
            Log(log_type=LogType.INFO, message="msg1"),
            Log(log_type=LogType.INFO, message="msg2"),
        ]

        count = await sub.receive(logs)

        assert count == 2
        mock_adapter.write.assert_called_once_with(logs)

    @pytest.mark.asyncio
    async def test_postgres_close_closes_adapter(self):
        """close should close the adapter."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test")

        mock_adapter = AsyncMock()
        mock_adapter.close = AsyncMock()
        sub._adapter = mock_adapter

        await sub.close()

        mock_adapter.close.assert_called_once()
        assert sub._adapter is None


# -----------------------------------------------------------------------------
# Integration tests for LogStore + LogBroadcaster
# -----------------------------------------------------------------------------


class TestLogStoreBroadcasterIntegration:
    """Integration tests for LogStore with LogBroadcaster."""

    @pytest.mark.asyncio
    async def test_logstore_aflush_broadcasts_to_subscribers(self):
        """aflush should broadcast to all registered subscribers."""
        from lionpride.session import LogStore

        store = LogStore(auto_save_on_exit=False)

        # Create broadcaster with mock subscribers
        broadcaster = LogBroadcaster()
        sub1 = MockSubscriber(name="sub1")
        sub2 = MockSubscriber(name="sub2")
        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        store.set_broadcaster(broadcaster)

        # Add logs
        store.log_info(message="test1")
        store.log_info(message="test2")
        store.log_api_call(model="gpt-4")

        # Flush
        results = await store.aflush(clear=True)

        # Verify results
        assert "broadcaster" in results
        assert results["broadcaster"]["sub1"] == 3
        assert results["broadcaster"]["sub2"] == 3

        # Verify subscribers received logs
        assert len(sub1.received_logs) == 3
        assert len(sub2.received_logs) == 3

        # Logs should be cleared
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_logstore_with_failing_subscriber(self):
        """LogStore should handle failing subscribers gracefully."""
        from lionpride.session import LogStore

        store = LogStore(auto_save_on_exit=False)

        broadcaster = LogBroadcaster()
        sub1 = MockSubscriber(name="sub1")
        sub2 = MockSubscriber(name="sub2", fail=True)
        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        store.set_broadcaster(broadcaster)

        store.log_info(message="test")

        results = await store.aflush(clear=True)

        # sub1 should succeed, sub2 should fail
        assert results["broadcaster"]["sub1"] == 1
        assert results["broadcaster"]["sub2"] == 0

        # Logs should NOT be cleared on partial failure (data loss prevention)
        assert len(store) == 1


# -----------------------------------------------------------------------------
# Real Integration tests with MinIO (requires Docker)
# -----------------------------------------------------------------------------

# Try to import testcontainers for MinIO tests
try:
    from testcontainers.minio import MinioContainer

    HAS_MINIO_TESTCONTAINER = True
except ImportError:
    HAS_MINIO_TESTCONTAINER = False

# Try to import aioboto3 for S3 tests
try:
    import aioboto3

    HAS_AIOBOTO3 = True
except ImportError:
    HAS_AIOBOTO3 = False


# -----------------------------------------------------------------------------
# Additional unit tests for coverage
# -----------------------------------------------------------------------------


class TestS3LogSubscriberEnsureClient:
    """Tests for S3LogSubscriber._ensure_client edge cases."""

    @pytest.mark.asyncio
    async def test_ensure_client_already_initialized(self):
        """_ensure_client should return early when client is already set."""
        sub = S3LogSubscriber(bucket="test")
        mock_client = MagicMock()
        sub._client = mock_client

        await sub._ensure_client()

        # Client should not be changed
        assert sub._client is mock_client

    @pytest.mark.asyncio
    async def test_ensure_client_import_error(self):
        """_ensure_client should raise ImportError when aioboto3 not installed."""
        sub = S3LogSubscriber(bucket="test")

        with (
            patch.dict("sys.modules", {"aioboto3": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'aioboto3'")),
            pytest.raises(ImportError, match="aioboto3 is required"),
        ):
            await sub._ensure_client()

    @pytest.mark.asyncio
    async def test_ensure_client_creates_session(self):
        """_ensure_client should create aioboto3 session."""
        sub = S3LogSubscriber(bucket="test")
        assert sub._client is None

        mock_session = MagicMock()
        with patch.dict("sys.modules", {"aioboto3": MagicMock(Session=lambda: mock_session)}):
            await sub._ensure_client()

        assert sub._client is mock_session


class TestS3LogSubscriberReceiveException:
    """Tests for S3LogSubscriber exception handling."""

    @pytest.mark.asyncio
    async def test_receive_s3_exception_returns_zero(self):
        """S3 errors should be logged and return 0."""
        sub = S3LogSubscriber(bucket="test")

        # Create mock client that raises exception
        class MockClientCM:
            async def __aenter__(self):
                raise RuntimeError("S3 connection failed")

            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.client = MagicMock(return_value=MockClientCM())
        sub._client = mock_session

        logs = [Log(log_type=LogType.INFO, message="test")]
        count = await sub.receive(logs)

        assert count == 0


class TestPostgresLogSubscriberInit:
    """Tests for PostgresLogSubscriber initialization."""

    def test_init_stores_parameters(self):
        """__init__ should store all parameters."""
        sub = PostgresLogSubscriber(
            dsn="postgresql://user:pass@localhost/db",
            table="custom_logs",
            auto_create=False,
        )

        assert sub.dsn == "postgresql://user:pass@localhost/db"
        assert sub.table == "custom_logs"
        assert sub.auto_create is False
        assert sub._adapter is None


class TestPostgresLogSubscriberEnsureAdapter:
    """Tests for PostgresLogSubscriber._ensure_adapter."""

    @pytest.mark.asyncio
    async def test_ensure_adapter_already_initialized(self):
        """_ensure_adapter should return early when adapter is already set."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test")
        mock_adapter = MagicMock()
        sub._adapter = mock_adapter

        await sub._ensure_adapter()

        assert sub._adapter is mock_adapter

    @pytest.mark.asyncio
    async def test_ensure_adapter_creates_adapter(self):
        """_ensure_adapter should create PostgresLogAdapter."""
        sub = PostgresLogSubscriber(
            dsn="postgresql://localhost/test",
            table="my_logs",
            auto_create=True,
        )

        # Patch at the module where it's imported
        with patch("lionpride.session.log_adapter.PostgresLogAdapter") as mock_class:
            mock_adapter = MagicMock()
            mock_class.return_value = mock_adapter

            await sub._ensure_adapter()

            mock_class.assert_called_once_with(
                dsn="postgresql://localhost/test",
                table="my_logs",
                auto_create=True,
            )
            assert sub._adapter is mock_adapter


class TestPostgresLogSubscriberReceive:
    """Tests for PostgresLogSubscriber.receive with adapter initialization."""

    @pytest.mark.asyncio
    async def test_receive_initializes_adapter(self):
        """receive should call _ensure_adapter before writing."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test")

        mock_adapter = AsyncMock()
        mock_adapter.write = AsyncMock(return_value=3)

        # Patch at the module where it's imported
        with patch("lionpride.session.log_adapter.PostgresLogAdapter", return_value=mock_adapter):
            logs = [
                Log(log_type=LogType.INFO, message="msg1"),
                Log(log_type=LogType.INFO, message="msg2"),
                Log(log_type=LogType.INFO, message="msg3"),
            ]

            count = await sub.receive(logs)

            assert count == 3
            mock_adapter.write.assert_called_once_with(logs)


class TestPostgresLogSubscriberClose:
    """Tests for PostgresLogSubscriber.close."""

    @pytest.mark.asyncio
    async def test_close_when_adapter_none(self):
        """close should be no-op when adapter is None."""
        sub = PostgresLogSubscriber(dsn="postgresql://localhost/test")
        assert sub._adapter is None

        await sub.close()  # Should not raise

        assert sub._adapter is None


class TestWebhookLogSubscriberImportError:
    """Tests for WebhookLogSubscriber httpx import error."""

    @pytest.mark.asyncio
    async def test_receive_httpx_import_error(self):
        """receive should raise ImportError when httpx not installed."""
        sub = WebhookLogSubscriber(url="https://example.com/logs")
        logs = [Log(log_type=LogType.INFO, message="test")]

        with (
            patch.dict("sys.modules", {"httpx": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'httpx'")),
            pytest.raises(ImportError, match="httpx is required"),
        ):
            await sub.receive(logs)


class TestWebhookLogSubscriberException:
    """Tests for WebhookLogSubscriber exception handling."""

    @pytest.mark.asyncio
    async def test_receive_request_exception(self):
        """Exception during POST should log error and continue."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_class.return_value = mock_client

            sub = WebhookLogSubscriber(url="https://example.com/logs")
            logs = [Log(log_type=LogType.INFO, message="test")]

            count = await sub.receive(logs)

            assert count == 0


class TestLogBroadcasterParallelFailFast:
    """Tests for LogBroadcaster parallel broadcast with fail_fast."""

    @pytest.mark.asyncio
    async def test_parallel_fail_fast_re_raises_in_send(self):
        """fail_fast=True in parallel mode raises inside _send (caught by gather).

        When fail_fast=True and an exception occurs in parallel mode:
        1. The exception is re-raised in _send (line 346)
        2. gather(return_exceptions=True) catches it and returns it as outcome
        3. The exception is skipped in results (line 354)
        """
        config = LogBroadcasterConfig(fail_fast=True, parallel=True)
        broadcaster = LogBroadcaster(config=config)

        sub1 = MockSubscriber(name="sub1", fail=True)
        sub2 = MockSubscriber(name="sub2")
        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        logs = [Log(log_type=LogType.INFO, message="test")]

        # fail_fast in parallel: exception is raised in _send, caught by gather
        # The failing subscriber won't appear in results (exception is skipped)
        results = await broadcaster.broadcast(logs)

        # sub1 failed and was skipped (exception caught by gather)
        assert "sub1" not in results or results.get("sub1") == 0
        # sub2 should succeed
        assert results["sub2"] == 1

    @pytest.mark.asyncio
    async def test_parallel_exception_skipped_in_results(self):
        """Exceptions in parallel should be skipped in results (line 354)."""
        config = LogBroadcasterConfig(fail_fast=False, parallel=True)
        broadcaster = LogBroadcaster(config=config)

        # Create a subscriber that raises directly
        class DirectFailSubscriber(LogSubscriber):
            @property
            def name(self) -> str:
                return "direct_fail"

            async def receive(self, logs: list[Log]) -> int:
                raise ValueError("Direct failure")

            async def close(self) -> None:
                pass

        sub1 = MockSubscriber(name="sub1")
        sub2 = DirectFailSubscriber()
        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)

        logs = [Log(log_type=LogType.INFO, message="test")]

        results = await broadcaster.broadcast(logs)

        # sub1 should succeed, sub2 exception skipped (not in results)
        assert results["sub1"] == 1
        # direct_fail returns 0 (handled by except block)
        assert results["direct_fail"] == 0


class TestLogBroadcasterCloseException:
    """Tests for LogBroadcaster.close exception handling."""

    @pytest.mark.asyncio
    async def test_close_handles_subscriber_exception(self):
        """close should handle exceptions from subscriber.close()."""

        class FailingCloseSubscriber(LogSubscriber):
            @property
            def name(self) -> str:
                return "failing_close"

            async def receive(self, logs: list[Log]) -> int:
                return len(logs)

            async def close(self) -> None:
                raise RuntimeError("Close failed")

        broadcaster = LogBroadcaster()
        sub1 = MockSubscriber(name="sub1")
        sub2 = FailingCloseSubscriber()
        sub3 = MockSubscriber(name="sub3")

        broadcaster.add_subscriber(sub1)
        broadcaster.add_subscriber(sub2)
        broadcaster.add_subscriber(sub3)

        # Should not raise, should continue to close other subscribers
        await broadcaster.close()

        # All subscribers should be removed
        assert len(broadcaster.list_subscribers()) == 0
        # sub1 and sub3 should have close called
        assert sub1.close_called is True
        assert sub3.close_called is True


@pytest.mark.integration
@pytest.mark.skipif(
    not HAS_MINIO_TESTCONTAINER or not HAS_AIOBOTO3,
    reason="testcontainers[minio] or aioboto3 not installed",
)
class TestS3LogSubscriberWithMinIO:
    """Real S3 integration tests using MinIO testcontainer.

    Note: MinIO testcontainer API varies by version. These tests verify
    the S3LogSubscriber works with S3-compatible storage.
    """

    @pytest.fixture(scope="class")
    def minio_container(self):
        """Start MinIO container for tests."""
        with MinioContainer() as minio:
            yield minio

    @pytest.mark.asyncio
    async def test_s3_receive_real_minio(self, minio_container):
        """Test actual S3 upload to MinIO."""
        # Get container config
        config = minio_container.get_config()
        host = minio_container.get_container_host_ip()
        port = minio_container.get_exposed_port(9000)
        endpoint = f"http://{host}:{port}"

        # Create subscriber
        subscriber = S3LogSubscriber(
            bucket="test-logs",
            prefix="logs/",
            endpoint_url=endpoint,
            aws_access_key_id=config.get("access_key", "minioadmin"),
            aws_secret_access_key=config.get("secret_key", "minioadmin"),
            region_name="us-east-1",
        )

        # Create bucket using minio client
        # Note: minio_container.get_client() is broken with minio>=7.2 (keyword-only args)
        # Create client manually with keyword arguments
        from minio import Minio
        from minio.error import S3Error

        client = Minio(
            endpoint=f"{host}:{port}",
            access_key=config.get("access_key", "minioadmin"),
            secret_key=config.get("secret_key", "minioadmin"),
            secure=False,
        )
        try:
            if not client.bucket_exists(bucket_name="test-logs"):
                client.make_bucket(bucket_name="test-logs")
        except S3Error as e:
            if e.code != "BucketAlreadyOwnedByYou":
                raise

        logs = [
            Log(log_type=LogType.INFO, message="test1"),
            Log(log_type=LogType.API_CALL, model="gpt-4", total_tokens=100),
        ]

        count = await subscriber.receive(logs)
        assert count == 2

        await subscriber.close()
