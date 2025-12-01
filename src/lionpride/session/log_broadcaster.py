# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Log broadcasters for multi-destination log distribution.

Provides async broadcasters that fan-out logs to multiple destinations:
- S3 (AWS, MinIO)
- PostgreSQL
- Custom subscribers

Example:
    broadcaster = LogBroadcaster()
    broadcaster.add_subscriber(S3LogSubscriber(bucket="logs"))
    broadcaster.add_subscriber(PostgresLogSubscriber(dsn="..."))
    await broadcaster.broadcast(logs)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lionpride.ln import json_dumps

if TYPE_CHECKING:
    from .logs import Log

__all__ = (
    "LogBroadcaster",
    "LogSubscriber",
    "PostgresLogSubscriber",
    "S3LogSubscriber",
    "WebhookLogSubscriber",
)

logger = logging.getLogger(__name__)


class LogSubscriber(ABC):
    """Abstract base class for log subscribers.

    Subscribers receive logs from the broadcaster and handle them
    according to their specific destination (S3, DB, webhook, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Subscriber name for identification."""
        pass

    @abstractmethod
    async def receive(self, logs: list[Log]) -> int:
        """Receive and process logs.

        Args:
            logs: List of Log objects to process

        Returns:
            Number of logs successfully processed
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the subscriber and release resources."""
        pass


class S3LogSubscriber(LogSubscriber):
    """S3-compatible storage subscriber.

    Writes logs as JSON files to S3 or S3-compatible storage (MinIO, etc.).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "logs/",
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-1",
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self._client = None

    @property
    def name(self) -> str:
        return f"s3:{self.bucket}"

    async def _ensure_client(self) -> None:
        """Ensure S3 client is initialized."""
        if self._client is not None:
            return

        try:
            import aioboto3
        except ImportError:
            raise ImportError(
                "aioboto3 is required for S3LogSubscriber. Install with: pip install aioboto3"
            )

        session = aioboto3.Session()
        self._client = session

    async def receive(self, logs: list[Log]) -> int:
        """Write logs to S3 as JSON file."""
        if not logs:
            return 0

        await self._ensure_client()

        # Generate key with timestamp
        ts = datetime.now(UTC).strftime("%Y/%m/%d/%H%M%S")
        key = f"{self.prefix}{ts}.json"

        # Convert logs to JSON
        log_dicts = [log.to_dict(mode="json") for log in logs]
        content = json_dumps(log_dicts, pretty=True)

        try:
            async with self._client.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            ) as client:
                await client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content.encode("utf-8"),
                    ContentType="application/json",
                )
            logger.debug(f"Wrote {len(logs)} logs to s3://{self.bucket}/{key}")
            return len(logs)
        except Exception as e:
            logger.error(f"Failed to write logs to S3: {e}")
            return 0

    async def close(self) -> None:
        """S3 client is session-based, no explicit close needed."""
        self._client = None


class PostgresLogSubscriber(LogSubscriber):
    """PostgreSQL database subscriber.

    Uses the PostgresLogAdapter for actual write operations.
    """

    def __init__(
        self,
        dsn: str,
        table: str = "logs",
        auto_create: bool = True,
    ):
        self.dsn = dsn
        self.table = table
        self.auto_create = auto_create
        self._adapter = None

    @property
    def name(self) -> str:
        return f"postgres:{self.table}"

    async def _ensure_adapter(self) -> None:
        """Ensure adapter is initialized."""
        if self._adapter is not None:
            return

        from .log_adapter import PostgresLogAdapter

        self._adapter = PostgresLogAdapter(
            dsn=self.dsn,
            table=self.table,
            auto_create=self.auto_create,
        )

    async def receive(self, logs: list[Log]) -> int:
        """Write logs to PostgreSQL."""
        if not logs:
            return 0

        await self._ensure_adapter()
        return await self._adapter.write(logs)

    async def close(self) -> None:
        """Close the adapter."""
        if self._adapter:
            await self._adapter.close()
            self._adapter = None


class WebhookLogSubscriber(LogSubscriber):
    """Webhook subscriber for sending logs to external services.

    Sends logs as JSON POST requests to configured endpoint.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        batch_size: int = 100,
    ):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.batch_size = batch_size
        self._client = None

    @property
    def name(self) -> str:
        return f"webhook:{self.url}"

    async def receive(self, logs: list[Log]) -> int:
        """Send logs to webhook endpoint."""
        if not logs:
            return 0

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for WebhookLogSubscriber. Install with: pip install httpx"
            )

        # Convert logs to JSON
        log_dicts = [log.to_dict(mode="json") for log in logs]

        # Send in batches
        count = 0
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(log_dicts), self.batch_size):
                batch = log_dicts[i : i + self.batch_size]
                try:
                    response = await client.post(
                        self.url,
                        json=batch,
                        headers=self.headers,
                    )
                    if response.is_success:
                        count += len(batch)
                    else:
                        logger.warning(f"Webhook returned {response.status_code}: {response.text}")
                except Exception as e:
                    logger.error(f"Failed to send logs to webhook: {e}")

        logger.debug(f"Sent {count} logs to webhook {self.url}")
        return count

    async def close(self) -> None:
        """No persistent resources to close."""
        pass


class LogBroadcasterConfig(BaseModel):
    """Configuration for LogBroadcaster."""

    fail_fast: bool = Field(
        default=False,
        description="Stop on first subscriber failure",
    )
    parallel: bool = Field(
        default=True,
        description="Send to subscribers in parallel",
    )


class LogBroadcaster:
    """Fan-out log broadcaster to multiple subscribers.

    Distributes logs to all registered subscribers (S3, DB, webhooks, etc.).

    Example:
        broadcaster = LogBroadcaster()
        broadcaster.add_subscriber(S3LogSubscriber(bucket="my-logs"))
        broadcaster.add_subscriber(PostgresLogSubscriber(dsn="..."))

        # Broadcast logs to all subscribers
        results = await broadcaster.broadcast(logs)
        # → {"s3:my-logs": 100, "postgres:logs": 100}
    """

    def __init__(self, config: LogBroadcasterConfig | None = None):
        self.config = config or LogBroadcasterConfig()
        self._subscribers: dict[str, LogSubscriber] = {}

    def add_subscriber(self, subscriber: LogSubscriber) -> None:
        """Add a subscriber."""
        if subscriber.name in self._subscribers:
            logger.warning(f"Replacing existing subscriber: {subscriber.name}")
        self._subscribers[subscriber.name] = subscriber

    def remove_subscriber(self, name: str) -> bool:
        """Remove a subscriber by name. Returns True if removed."""
        if name in self._subscribers:
            del self._subscribers[name]
            return True
        return False

    def list_subscribers(self) -> list[str]:
        """List all subscriber names."""
        return list(self._subscribers.keys())

    async def broadcast(self, logs: list[Log]) -> dict[str, int]:
        """Broadcast logs to all subscribers.

        Args:
            logs: List of Log objects to broadcast

        Returns:
            Dict mapping subscriber name to count of logs written
        """
        if not logs:
            return {}

        if not self._subscribers:
            logger.warning("No subscribers registered, logs not broadcasted")
            return {}

        results: dict[str, int] = {}

        if self.config.parallel:
            # Parallel broadcast
            from lionpride.libs import concurrency

            async def _send(name: str, sub: LogSubscriber) -> tuple[str, int]:
                try:
                    count = await sub.receive(logs)
                    return name, count
                except Exception as e:
                    logger.error(f"Subscriber {name} failed: {e}")
                    if self.config.fail_fast:
                        raise
                    return name, 0

            tasks = [_send(name, sub) for name, sub in self._subscribers.items()]
            outcomes = await concurrency.gather(*tasks, return_exceptions=True)

            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    continue
                name, count = outcome
                results[name] = count
        else:
            # Sequential broadcast
            for name, subscriber in self._subscribers.items():
                try:
                    count = await subscriber.receive(logs)
                    results[name] = count
                except Exception as e:
                    logger.error(f"Subscriber {name} failed: {e}")
                    results[name] = 0
                    if self.config.fail_fast:
                        break

        return results

    async def close(self) -> None:
        """Close all subscribers."""
        for subscriber in self._subscribers.values():
            try:
                await subscriber.close()
            except Exception as e:
                logger.error(f"Error closing subscriber {subscriber.name}: {e}")
        self._subscribers.clear()

    def __repr__(self) -> str:
        return f"LogBroadcaster(subscribers={list(self._subscribers.keys())})"
