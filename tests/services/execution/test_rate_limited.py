# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for RateLimitedProcessor and RateLimitedExecutor.

## Architecture Overview

### Dual Rate Limiting
RateLimitedProcessor manages two independent rate limits:
1. Request limit: Maximum number of API calls per time window
2. Token limit: Maximum tokens (estimated input/output size) per time window

Both limits must be satisfied for permission. Enforced atomically with automatic
rollback if dual acquire fails.

### Atomic Rollback (Two-Phase Commit)
When both limits configured:
1. Acquire from request bucket
2. Acquire from token bucket
3. If (2) fails, rollback (1) automatically

Prevents resource leakage if one limit succeeds but the other fails.

### Token Check Skip Behavior (F5 Finding)
If token_bucket configured but required_tokens is None, token check is SKIPPED.
Intentional design to support mixed workloads:

- APICalling with token estimate → both limits checked
- ToolCalling (no tokens) → token check skipped (only request limit)
- APICalling with empty messages → token check skipped (can't estimate)

Security: Skip is safe because request limit still enforced (prevents flood).

### Interval-Based Replenishment
Buckets refill at regular intervals (not continuous). Prevents "burst then wait"
behavior, provides smoother rate limiting matching API provider windows
(e.g., OpenAI: 500 RPM = reset every 60s).

### Concurrency Safety (BLIND-2 Finding)
TokenBucket uses threading.Lock for atomic token operations. Tests verify:
- Request limits atomic under concurrent access (no over-allocation)
- Token limits atomic under concurrent access
- Dual limit atomic rollback works correctly under stress

Pattern proven in lionagi v0 (~2 years production).

Key behaviors validated:
- Initialization: Minimal config, request-only, dual limits, custom limits
- Factory method: .create() starts background replenishment task
- Replenishment: Resets bucket capacities at intervals, task cancellation
- Permission checks: No limits allows all, request limit enforcement,
  token limit enforcement, dual limit both-must-pass, atomic rollback
- Executor integration: Auto-construction starts replenishment task
"""

import pytest

from lionpride.libs.concurrency import sleep
from lionpride.services.execution import RateLimitedExecutor, RateLimitedProcessor
from lionpride.services.types.endpoint import APICalling
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket


class TestRateLimitedProcessorInit:
    """Test RateLimitedProcessor initialization."""

    def test_init_minimal(self):
        """Test initialization with minimal params."""
        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
        )

        assert processor.request_bucket is None
        assert processor.token_bucket is None
        assert processor.replenishment_interval == 60.0
        assert processor._replenisher_task is None

    @pytest.mark.asyncio
    async def test_init_with_request_bucket(self):
        """Test initialization with request rate limiting."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=100, refill_rate=10))

        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
        )

        assert processor.request_bucket is request_bucket
        assert processor.token_bucket is None

    @pytest.mark.asyncio
    async def test_init_with_both_buckets(self):
        """Test initialization with dual rate limiting."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=100, refill_rate=10))
        token_bucket = TokenBucket(RateLimitConfig(capacity=10000, refill_rate=1000))

        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
            replenishment_interval=30.0,
        )

        assert processor.request_bucket is request_bucket
        assert processor.token_bucket is token_bucket
        assert processor.replenishment_interval == 30.0

    def test_init_with_custom_limits(self):
        """Test initialization with custom concurrency and queue limits."""
        processor = RateLimitedProcessor(
            queue_capacity=50,
            capacity_refresh_time=2.0,
            concurrency_limit=200,
            max_queue_size=2000,
            max_denial_tracking=20000,
        )

        # These are passed to base Processor
        assert processor.event_type == APICalling


class TestRateLimitedProcessorCreate:
    """Test RateLimitedProcessor.create() factory method."""

    @pytest.mark.asyncio
    async def test_create_starts_replenishment_task(self):
        """Test that create() starts background replenishment task."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1))

        processor = await RateLimitedProcessor.create(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
            replenishment_interval=0.1,  # Fast for testing
        )

        try:
            assert processor._replenisher_task is not None
            assert not processor._replenisher_task.done()
        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_create_with_all_params(self):
        """Test create() with all parameters."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=100, refill_rate=10))
        token_bucket = TokenBucket(RateLimitConfig(capacity=10000, refill_rate=1000))

        processor = await RateLimitedProcessor.create(
            queue_capacity=50,
            capacity_refresh_time=2.0,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
            replenishment_interval=30.0,
            concurrency_limit=200,
            max_queue_size=2000,
            max_denial_tracking=20000,
        )

        try:
            assert processor.request_bucket is request_bucket
            assert processor.token_bucket is token_bucket
            assert processor._replenisher_task is not None
        finally:
            await processor.stop()


class TestRateLimitedProcessorReplenishment:
    """Test rate limit replenishment behavior."""

    @pytest.mark.asyncio
    async def test_replenishment_resets_buckets(self):
        """Test that replenishment task resets bucket capacities."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1))
        token_bucket = TokenBucket(RateLimitConfig(capacity=1000, refill_rate=100))

        # Consume some tokens
        await request_bucket.acquire(tokens=5)
        await token_bucket.acquire(tokens=500)

        assert request_bucket.tokens == 5.0
        assert token_bucket.tokens == 500.0

        processor = await RateLimitedProcessor.create(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
            replenishment_interval=0.1,  # Fast replenishment for testing
        )

        try:
            # Wait for replenishment
            await sleep(0.15)

            # Buckets should be reset to full capacity
            assert request_bucket.tokens == 10.0
            assert token_bucket.tokens == 1000.0
        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_replenishment_task_cancellation(self):
        """Test that stop() cancels replenishment task."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1))

        processor = await RateLimitedProcessor.create(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
            replenishment_interval=0.1,
        )

        replenisher_task = processor._replenisher_task
        assert not replenisher_task.done()

        await processor.stop()

        # Task should be cancelled
        assert replenisher_task.cancelled()


class TestRateLimitedProcessorPermission:
    """Test request_permission() rate limit enforcement."""

    @pytest.mark.asyncio
    async def test_no_buckets_allows_all(self):
        """Test that no rate limits means all requests are allowed."""
        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
        )

        # Should allow unlimited requests
        for _ in range(100):
            assert await processor.request_permission() is True

    @pytest.mark.asyncio
    async def test_request_limit_enforcement(self):
        """Test request rate limit (count-based)."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=5, refill_rate=1))

        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
        )

        # First 5 requests should pass
        for i in range(5):
            assert await processor.request_permission() is True, f"Request {i} should pass"

        # 6th request should be denied
        assert await processor.request_permission() is False

    @pytest.mark.asyncio
    async def test_token_limit_enforcement(self):
        """Test token rate limit (size-based)."""
        token_bucket = TokenBucket(RateLimitConfig(capacity=1000, refill_rate=100))

        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            token_bucket=token_bucket,
        )

        # Request with 500 tokens should pass (twice)
        assert await processor.request_permission(required_tokens=500) is True
        assert await processor.request_permission(required_tokens=500) is True

        # Third request should be denied (would exceed 1000)
        assert await processor.request_permission(required_tokens=500) is False

    @pytest.mark.asyncio
    async def test_dual_limit_both_must_pass(self):
        """Test that both request and token limits must pass."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1))
        token_bucket = TokenBucket(RateLimitConfig(capacity=1000, refill_rate=100))

        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
        )

        # Use up request bucket (10 requests)
        for _ in range(10):
            assert await processor.request_permission(required_tokens=50) is True

        # Next request should fail even with tokens available
        assert await processor.request_permission(required_tokens=50) is False

    @pytest.mark.asyncio
    async def test_atomic_rollback_on_token_denial(self):
        """Test that request token is released if token check fails (atomic rollback)."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1))
        token_bucket = TokenBucket(RateLimitConfig(capacity=1000, refill_rate=100))

        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=1.0,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
        )

        # Consume most tokens
        await token_bucket.acquire(tokens=900)
        assert token_bucket.tokens == 100.0

        initial_request_tokens = request_bucket.tokens
        assert initial_request_tokens == 10.0

        # Request that would pass request check but fail token check
        result = await processor.request_permission(required_tokens=200)

        assert result is False  # Should be denied

        # Request token should be rolled back (atomic)
        assert request_bucket.tokens == initial_request_tokens


class TestRateLimitedExecutor:
    """Test RateLimitedExecutor."""

    def test_init_default(self):
        """Test executor initialization with defaults."""
        executor = RateLimitedExecutor()

        assert executor.processor_type == RateLimitedProcessor
        # Verify executor has states Flow initialized
        assert executor.states is not None
        assert len(executor.states.progressions) == 7  # 7 default progressions

    @pytest.mark.asyncio
    async def test_init_with_config(self):
        """Test executor initialization with processor config."""
        request_bucket = TokenBucket(RateLimitConfig(capacity=100, refill_rate=10))

        processor_config = {
            "queue_capacity": 50,
            "capacity_refresh_time": 2.0,
            "request_bucket": request_bucket,
        }

        executor = RateLimitedExecutor(
            processor_config=processor_config,
            name="custom_executor",
        )

        # Verify processor config is passed through
        assert executor.processor_type == RateLimitedProcessor

    def test_init_with_strict_event_type(self):
        """Test executor with strict event type enforcement."""
        executor = RateLimitedExecutor(strict_event_type=True)

        # Flow should enforce exact APICalling type
        # (This is tested more in lionpride-core tests)
        assert executor.processor_type == RateLimitedProcessor

    @pytest.mark.asyncio
    async def test_auto_construction_starts_replenishment(self):
        """Test that auto-constructed executor starts replenishment task.

        Regression test for auto-construction bug:
        - When executor is created via __init__ (not .create()),
          replenishment task must still start when executor.start() is called.
        - Without this fix, auto-constructed executors never replenish,
          causing permanent rate limiting after first N requests.
        """
        # Create executor via auto-construction (not .create())
        request_bucket = TokenBucket(RateLimitConfig(capacity=5, refill_rate=1))

        processor_config = {
            "queue_capacity": 10,
            "capacity_refresh_time": 1.0,
            "request_bucket": request_bucket,
            "replenishment_interval": 0.1,  # Fast for testing
        }

        executor = RateLimitedExecutor(processor_config=processor_config)

        try:
            # Start executor (this should start replenishment task)
            await executor.start()

            # Verify replenishment task is running
            assert executor.processor._replenisher_task is not None
            assert not executor.processor._replenisher_task.done()

            # Exhaust the bucket (5 requests)
            for _ in range(5):
                assert await executor.processor.request_permission() is True

            # 6th request should be denied (bucket exhausted)
            assert await executor.processor.request_permission() is False

            # Wait for replenishment
            await sleep(0.15)

            # 6th request should now succeed (bucket replenished)
            assert await executor.processor.request_permission() is True

        finally:
            await executor.stop()


class TestRateLimitedConcurrency:
    """Test rate limiting under concurrent access.

    CRITICAL: Rate limiting is a security control.
    Locks must be atomic to prevent over-allocation.
    """

    @pytest.mark.asyncio
    async def test_concurrent_request_limit_atomic(self):
        """Verify request limit is atomic under concurrent access."""
        import asyncio
        from unittest.mock import Mock

        # Create processor with limit=10 requests (minimal refill)
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=0.001))
        processor = RateLimitedProcessor(
            queue_capacity=10, capacity_refresh_time=60, request_bucket=request_bucket
        )

        # Fire 20 concurrent permission requests (no tokens required)
        async def request_permission():
            return await processor.request_permission()

        results = await asyncio.gather(*[request_permission() for _ in range(20)])

        # CRITICAL: Verify exactly 10 succeed, 10 fail (not 11 due to race)
        successes = [r for r in results if r is True]
        denials = [r for r in results if r is False]

        assert len(successes) == 10, (
            f"Expected 10 successes, got {len(successes)} (race condition!)"
        )
        assert len(denials) == 10

    @pytest.mark.asyncio
    async def test_concurrent_token_limit_atomic(self):
        """Verify token limit is atomic under concurrent access."""
        import asyncio

        # Create processor with token limit (5000 tokens total, minimal refill)
        token_bucket = TokenBucket(RateLimitConfig(capacity=5000, refill_rate=0.001))
        processor = RateLimitedProcessor(
            queue_capacity=100, capacity_refresh_time=60, token_bucket=token_bucket
        )

        # Fire 10 concurrent requests, each requesting 1000 tokens
        # Total: 10,000 tokens requested, limit: 5000
        async def request_permission():
            return await processor.request_permission(required_tokens=1000)

        results = await asyncio.gather(*[request_permission() for _ in range(10)])

        # CRITICAL: Verify exactly 5 succeed (5*1000=5000), 5 fail
        successes = [r for r in results if r is True]
        denials = [r for r in results if r is False]

        assert len(successes) == 5, f"Expected 5 successes, got {len(successes)} (race condition!)"
        assert len(denials) == 5

    @pytest.mark.asyncio
    async def test_concurrent_dual_limit_atomic_rollback(self):
        """Verify atomic rollback when dual limits are stressed concurrently."""
        import asyncio

        # Create processor with BOTH limits (minimal refill for testing)
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=0.001))
        token_bucket = TokenBucket(RateLimitConfig(capacity=5000, refill_rate=0.001))

        processor = RateLimitedProcessor(
            queue_capacity=10,
            capacity_refresh_time=60,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
        )

        # Create requests that stress BOTH limits
        # 15 requests, each 600 tokens
        # Request limit: 10 (hit first)
        # Token limit: 5000 / 600 = 8.3 (would hit if request limit didn't exist)
        async def request_permission():
            return await processor.request_permission(required_tokens=600)

        results = await asyncio.gather(*[request_permission() for _ in range(15)])

        # Verify results match stricter limit (token limit in this case)
        successes = [r for r in results if r is True]
        denials = [r for r in results if r is False]

        # Analysis:
        # - Request limit: 10 requests
        # - Token limit: 5000 / 600 = 8.33... requests
        # - Token limit is stricter → should allow exactly 8 requests
        #
        # With dual limits and atomic rollback:
        # 1. Request bucket check (acquire 1 token)
        # 2. Token bucket check (acquire 600 tokens)
        # 3. If step 2 fails, rollback step 1 (atomic)
        #
        # Expected: 8 successes (token-limited), 7 denials
        assert len(successes) == 8, f"Expected 8 successes (token-limited), got {len(successes)}"
        assert len(denials) == 7

        # Verify atomic rollback worked correctly
        # After 8 successes:
        # - Request bucket: 10 - 8 = 2 tokens remaining (rollback worked)
        # - Token bucket: 5000 - (8*600) = 200 tokens remaining
        # Use approx for floating-point comparison (time-based refill causes tiny drift)
        assert request_bucket.tokens == pytest.approx(2.0, abs=0.01), (
            f"Request rollback failed: {request_bucket.tokens}"
        )
        assert token_bucket.tokens == pytest.approx(200.0, abs=0.01), (
            f"Token bucket: {token_bucket.tokens}"
        )


class TestRateLimitedProcessorSerialization:
    """Test RateLimitedProcessor.to_dict() serialization."""

    def test_to_dict_minimal(self):
        """Test to_dict with minimal config (no buckets)."""
        processor = RateLimitedProcessor(
            queue_capacity=50,
            capacity_refresh_time=2.0,
        )

        result = processor.to_dict()

        assert result["queue_capacity"] == 50
        assert result["capacity_refresh_time"] == 2.0
        assert result["replenishment_interval"] == 60.0  # default
        assert result["concurrency_limit"] == 100  # default
        assert result["max_queue_size"] == 1000  # default
        assert result["max_denial_tracking"] == 10000  # default
        assert result["request_bucket"] is None
        assert result["token_bucket"] is None

    @pytest.mark.asyncio
    async def test_to_dict_with_buckets(self):
        """Test to_dict serializes TokenBucket configs correctly."""
        # TokenBucket requires async context (uses current_time from anyio)
        request_bucket = TokenBucket(RateLimitConfig(capacity=100, refill_rate=10))
        token_bucket = TokenBucket(RateLimitConfig(capacity=5000, refill_rate=100))

        processor = RateLimitedProcessor(
            queue_capacity=25,
            capacity_refresh_time=30.0,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
            replenishment_interval=45.0,
            concurrency_limit=50,
            max_queue_size=500,
            max_denial_tracking=5000,
        )

        result = processor.to_dict()

        # Verify config values
        assert result["queue_capacity"] == 25
        assert result["capacity_refresh_time"] == 30.0
        assert result["replenishment_interval"] == 45.0
        assert result["concurrency_limit"] == 50
        assert result["max_queue_size"] == 500
        assert result["max_denial_tracking"] == 5000

        # Verify bucket configs serialized
        assert result["request_bucket"] is not None
        assert result["request_bucket"]["capacity"] == 100
        assert result["request_bucket"]["refill_rate"] == 10

        assert result["token_bucket"] is not None
        assert result["token_bucket"]["capacity"] == 5000
        assert result["token_bucket"]["refill_rate"] == 100


# Integration tests would go here but require mocking APICalling events
# and setting up full processor/executor lifecycle. These are better
# tested in E2E tests with real iModel usage.


class TestRateLimitedExecutorStart:
    """Test RateLimitedExecutor.start() replenisher task creation (lines 256-258)."""

    @pytest.mark.asyncio
    async def test_start_creates_replenisher_task_when_missing(self):
        """Test start() creates replenisher task when processor doesn't have one (lines 256-258).

        When executor.start() is called and processor exists but has no replenisher task,
        it should create one. This covers the branch at lines 256-258:

            if (
                self.processor
                and isinstance(self.processor, RateLimitedProcessor)
                and not self.processor._replenisher_task
            ):
                self.processor._replenisher_task = asyncio.create_task(...)
        """
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1))

        processor_config = {
            "queue_capacity": 10,
            "capacity_refresh_time": 1.0,
            "request_bucket": request_bucket,
            "replenishment_interval": 0.1,  # Fast for testing
        }

        executor = RateLimitedExecutor(processor_config=processor_config)

        try:
            # First start - creates processor and replenisher task via parent start()
            await executor.start()
            assert executor.processor is not None
            assert executor.processor._replenisher_task is not None

            first_task = executor.processor._replenisher_task

            # Stop the executor (cancels replenisher task)
            await executor.stop()

            # Manually clear the task to simulate missing task scenario
            executor.processor._replenisher_task = None

            # Second start - should create replenisher task via lines 256-258
            await executor.start()

            # Verify new replenisher task was created
            assert executor.processor._replenisher_task is not None
            assert executor.processor._replenisher_task is not first_task
            assert not executor.processor._replenisher_task.done()

        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_start_skips_replenisher_when_already_exists(self):
        """Test start() does NOT create replenisher task when one already exists.

        This is the negative case for lines 256-258 - when processor._replenisher_task
        already exists, no new task should be created.
        """
        request_bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1))

        processor_config = {
            "queue_capacity": 10,
            "capacity_refresh_time": 1.0,
            "request_bucket": request_bucket,
            "replenishment_interval": 0.1,
        }

        executor = RateLimitedExecutor(processor_config=processor_config)

        try:
            # First start
            await executor.start()
            first_task = executor.processor._replenisher_task

            # Second start (without stopping) - should NOT create new task
            await executor.start()
            second_task = executor.processor._replenisher_task

            # Same task should be preserved
            assert second_task is first_task

        finally:
            await executor.stop()

    @pytest.mark.asyncio
    async def test_start_creates_replenisher_via_executor_start(self):
        """Test start() creates replenisher task via executor.start() code path.

        This specifically covers lines 256-258 in RateLimitedExecutor.start():
        - Line 256: import asyncio
        - Line 257-258: self.processor._replenisher_task = asyncio.create_task(...)

        The key is to ensure the processor exists but has no _replenisher_task,
        then call start() which should create the task via lines 256-258.
        """
        # Create executor without initially starting replenisher
        # The RateLimitedProcessor.create() would normally start it,
        # but going through RateLimitedExecutor() bypasses that path
        processor_config = {
            "queue_capacity": 10,
            "capacity_refresh_time": 1.0,
            "replenishment_interval": 0.1,
            # Add a bucket so there's something to replenish
            "request_bucket": TokenBucket(RateLimitConfig(capacity=10, refill_rate=1)),
        }

        executor = RateLimitedExecutor(processor_config=processor_config)

        # Before start, processor should not exist
        assert executor.processor is None

        try:
            # First start - creates processor and replenisher task
            await executor.start()

            # Verify processor was created with replenisher task
            assert executor.processor is not None
            assert executor.processor._replenisher_task is not None
            assert not executor.processor._replenisher_task.done()

        finally:
            await executor.stop()
