# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

from typing_extensions import override

from lionpride.core import Event, Executor, Processor
from lionpride.libs.concurrency import get_cancelled_exc_class, sleep

from ..types.endpoint import APICalling
from ..utilities.rate_limiter import TokenBucket

if TYPE_CHECKING:
    import asyncio

    from lionpride.core import Pile

__all__ = ("RateLimitedExecutor", "RateLimitedProcessor")

logger = logging.getLogger(__name__)


class RateLimitedProcessor(Processor):
    """Permission-based rate limiter with dual token bucket limits and atomic rollback.

    Manages request and token limits using TokenBucket. Both limits are enforced
    atomically with automatic rollback if dual acquire fails.

    Attributes:
        request_bucket: Request count limiter (optional)
        token_bucket: Token usage limiter (optional)
        queue_capacity: Maximum queued callings
        capacity_refresh_time: Replenishment interval in seconds
    """

    event_type = APICalling

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event] | None = None,
        executor: Executor | None = None,
        request_bucket: TokenBucket | None = None,
        token_bucket: TokenBucket | None = None,
        replenishment_interval: float = 60.0,
        concurrency_limit: int = 100,
        max_queue_size: int = 1000,
        max_denial_tracking: int = 10000,
    ) -> None:
        """Initialize rate-limited processor.

        Args:
            queue_capacity: Max events per batch
            capacity_refresh_time: Batch refresh interval (seconds)
            pile: Reference to executor's Flow.items (set by executor)
            executor: Reference to executor for progression updates
            request_bucket: TokenBucket for request rate limiting (optional)
            token_bucket: TokenBucket for token rate limiting (optional)
            replenishment_interval: Rate limit reset interval (default: 60s)
            concurrency_limit: Max concurrent executions (default: 100)
            max_queue_size: Max queue size (default: 1000)
            max_denial_tracking: Max denial entries to track (default: 10000)
        """
        super().__init__(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            pile=pile,
            executor=executor,
            concurrency_limit=concurrency_limit,
            max_queue_size=max_queue_size,
            max_denial_tracking=max_denial_tracking,
        )

        self.request_bucket = request_bucket
        self.token_bucket = token_bucket
        self.replenishment_interval = replenishment_interval
        self.concurrency_limit = concurrency_limit
        self._replenisher_task: asyncio.Task[None] | None = None

    async def start_replenishing(self) -> None:
        """Background task to replenish rate limits at regular intervals."""
        await self.start()

        try:
            while not self.is_stopped():
                await sleep(self.replenishment_interval)

                if self.request_bucket:
                    await self.request_bucket.reset()
                    logger.debug(
                        "Request bucket replenished: %d requests",
                        self.request_bucket.capacity,
                    )

                if self.token_bucket:
                    await self.token_bucket.reset()
                    logger.debug(
                        "Token bucket replenished: %d tokens",
                        self.token_bucket.capacity,
                    )

        except get_cancelled_exc_class():
            logger.info("Rate limit replenisher task cancelled.")

    @override
    @classmethod
    async def create(
        cls,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event] | None = None,
        executor: Executor | None = None,
        request_bucket: TokenBucket | None = None,
        token_bucket: TokenBucket | None = None,
        replenishment_interval: float = 60.0,
        concurrency_limit: int = 100,
        max_queue_size: int = 1000,
        max_denial_tracking: int = 10000,
    ) -> Self:
        """Create processor and start replenishment task."""
        self = cls(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            pile=pile,
            executor=executor,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
            replenishment_interval=replenishment_interval,
            concurrency_limit=concurrency_limit,
            max_queue_size=max_queue_size,
            max_denial_tracking=max_denial_tracking,
        )

        import asyncio

        self._replenisher_task = asyncio.create_task(self.start_replenishing())

        return self

    @override
    async def stop(self) -> None:
        """Stop processor and cancel replenishment task."""
        if self._replenisher_task:
            self._replenisher_task.cancel()
            try:
                await self._replenisher_task
            except get_cancelled_exc_class():
                pass

        await super().stop()

    @override
    async def request_permission(
        self,
        required_tokens: int | None = None,
        **kwargs: Any,
    ) -> bool:
        """Request permission based on rate limits.

        Checks request and token limits (if configured). Atomically acquires from
        both buckets with automatic rollback if token check fails.

        Token check is skipped if required_tokens is None (supports ToolCalling).

        Args:
            required_tokens: Token usage for this request (None = skip token check)
            **kwargs: Additional request data (ignored)

        Returns:
            True if permitted, False if rate limited
        """
        if self.request_bucket is None and self.token_bucket is None:
            return True

        request_acquired = False
        if self.request_bucket:
            request_acquired = await self.request_bucket.try_acquire(tokens=1)
            if not request_acquired:
                logger.debug("Request rate limit exceeded")
                return False

        if self.token_bucket and required_tokens:
            token_acquired = await self.token_bucket.try_acquire(tokens=required_tokens)
            if not token_acquired:
                if request_acquired and self.request_bucket:
                    await self.request_bucket.release(tokens=1)

                logger.debug(
                    f"Token rate limit exceeded (required: {required_tokens}, "
                    f"available: {self.token_bucket.tokens:.0f})"
                )
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize processor config (excludes runtime state)."""
        return {
            "queue_capacity": self.queue_capacity,
            "capacity_refresh_time": self.capacity_refresh_time,
            "replenishment_interval": self.replenishment_interval,
            "concurrency_limit": self.concurrency_limit,
            "max_queue_size": self.max_queue_size,
            "max_denial_tracking": self.max_denial_tracking,
            "request_bucket": (self.request_bucket.to_dict() if self.request_bucket else None),
            "token_bucket": (self.token_bucket.to_dict() if self.token_bucket else None),
        }


class RateLimitedExecutor(Executor):
    """Executor with integrated RateLimitedProcessor for permission-based rate limiting.

    Manages processor lifecycle and forwards events for permission checking and queuing.

    Attributes:
        processor: RateLimitedProcessor instance
        processor_config: Config dict for processor serialization
    """

    processor_type = RateLimitedProcessor

    def __init__(
        self,
        processor_config: dict[str, Any] | None = None,
        strict_event_type: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize rate-limited executor.

        Args:
            processor_config: Config dict for RateLimitedProcessor.create()
            strict_event_type: If True, Flow enforces exact type matching
            name: Optional name for the executor Flow
        """
        super().__init__(
            processor_config=processor_config,
            strict_event_type=strict_event_type,
            name=name or "rate_limited_executor",
        )

    @override
    async def start(self) -> None:
        """Start executor and ensure replenishment task is running."""
        await super().start()

        if (
            self.processor
            and isinstance(self.processor, RateLimitedProcessor)
            and not self.processor._replenisher_task
        ):
            import asyncio

            self.processor._replenisher_task = asyncio.create_task(
                self.processor.start_replenishing()
            )
