# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for rate_limiter.py to achieve 100% coverage."""

import pytest

from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket


class TestRateLimitConfig:
    """Test suite for RateLimitConfig."""

    def test_valid_config_with_defaults(self):
        """Test valid configuration with default initial_tokens."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0)
        assert config.capacity == 10
        assert config.refill_rate == 2.0
        assert config.initial_tokens == 10  # defaults to capacity

    def test_valid_config_with_custom_initial_tokens(self):
        """Test valid configuration with custom initial_tokens."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=5)
        assert config.capacity == 10
        assert config.refill_rate == 2.0
        assert config.initial_tokens == 5

    def test_invalid_capacity_zero(self):
        """Test that capacity of 0 raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be > 0"):
            RateLimitConfig(capacity=0, refill_rate=2.0)

    def test_invalid_capacity_negative(self):
        """Test that negative capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be > 0"):
            RateLimitConfig(capacity=-5, refill_rate=2.0)

    def test_invalid_refill_rate_zero(self):
        """Test that refill_rate of 0 raises ValueError."""
        with pytest.raises(ValueError, match="refill_rate must be > 0"):
            RateLimitConfig(capacity=10, refill_rate=0)

    def test_invalid_refill_rate_negative(self):
        """Test that negative refill_rate raises ValueError."""
        with pytest.raises(ValueError, match="refill_rate must be > 0"):
            RateLimitConfig(capacity=10, refill_rate=-1.5)

    def test_invalid_initial_tokens_negative(self):
        """Test that negative initial_tokens raises ValueError."""
        with pytest.raises(ValueError, match="initial_tokens must be >= 0"):
            RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=-5)

    def test_invalid_initial_tokens_exceeds_capacity(self):
        """Test that initial_tokens > capacity raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed capacity"):
            RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=20)


class TestTokenBucket:
    """Test suite for TokenBucket."""

    @pytest.mark.asyncio
    async def test_acquire_immediate_success(self):
        """Test acquiring tokens when sufficient tokens are available."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=10)
        bucket = TokenBucket(config)

        result = await bucket.acquire(tokens=5)
        assert result is True
        assert bucket.tokens == 5.0

    @pytest.mark.asyncio
    async def test_acquire_all_tokens(self):
        """Test acquiring all available tokens."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=10)
        bucket = TokenBucket(config)

        result = await bucket.acquire(tokens=10)
        assert result is True
        assert bucket.tokens == 0.0

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self):
        """Test acquiring tokens when we need to wait for refill."""
        config = RateLimitConfig(capacity=10, refill_rate=10.0, initial_tokens=2)
        bucket = TokenBucket(config)

        # Need 5 tokens, have 2, need to wait ~0.3s for 3 more tokens
        result = await bucket.acquire(tokens=5, timeout=2.0)
        assert result is True
        # Tokens should be approximately 0 (5 acquired, some may have refilled during wait)
        assert bucket.tokens >= 0

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test that acquire returns False when timeout is exceeded."""
        config = RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=0)
        bucket = TokenBucket(config)

        # Need 10 tokens, refill rate is 1/s, so needs 10s but timeout is 0.1s
        result = await bucket.acquire(tokens=10, timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_multiple_sequential(self):
        """Test multiple sequential acquire calls."""
        config = RateLimitConfig(capacity=10, refill_rate=5.0, initial_tokens=10)
        bucket = TokenBucket(config)

        result1 = await bucket.acquire(tokens=3)
        assert result1 is True

        result2 = await bucket.acquire(tokens=3)
        assert result2 is True

        # Should have approximately 4 tokens left (10 - 3 - 3 = 4)
        assert bucket.tokens >= 3.0  # May have refilled slightly

    @pytest.mark.asyncio
    async def test_try_acquire_success(self):
        """Test try_acquire when tokens are available."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=10)
        bucket = TokenBucket(config)

        result = await bucket.try_acquire(tokens=5)
        assert result is True
        assert bucket.tokens == 5.0

    @pytest.mark.asyncio
    async def test_try_acquire_failure(self):
        """Test try_acquire when insufficient tokens."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=3)
        bucket = TokenBucket(config)

        result = await bucket.try_acquire(tokens=5)
        assert result is False
        # Tokens may have refilled slightly due to time elapsed
        assert bucket.tokens >= 3.0

    @pytest.mark.asyncio
    async def test_try_acquire_exact_tokens(self):
        """Test try_acquire when exactly enough tokens available."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=5)
        bucket = TokenBucket(config)

        result = await bucket.try_acquire(tokens=5)
        assert result is True
        # Tokens should be approximately 0 (may have slight refill)
        assert bucket.tokens < 1.0

    @pytest.mark.asyncio
    async def test_refill_increases_tokens(self):
        """Test that refill increases token count over time."""
        config = RateLimitConfig(capacity=10, refill_rate=10.0, initial_tokens=0)
        bucket = TokenBucket(config)

        # Manually trigger refill by calling _refill indirectly via acquire
        import asyncio

        await asyncio.sleep(0.5)  # Wait 0.5s for refill

        result = await bucket.try_acquire(tokens=1)
        # After 0.5s with 10 tokens/s rate, should have ~5 tokens
        assert result is True

    @pytest.mark.asyncio
    async def test_refill_respects_capacity(self):
        """Test that refill doesn't exceed capacity."""
        config = RateLimitConfig(capacity=10, refill_rate=100.0, initial_tokens=8)
        bucket = TokenBucket(config)

        import asyncio

        await asyncio.sleep(0.5)  # Wait for refill (would add 50 tokens if unlimited)

        # Access tokens after refill via try_acquire
        result = await bucket.try_acquire(tokens=1)
        assert result is True
        # Should be capped at capacity (10), so after acquiring 1, should have ~9
        assert bucket.tokens <= 9.0
        assert bucket.tokens >= 8.0  # At least the original amount minus acquired

    @pytest.mark.asyncio
    async def test_to_dict(self):
        """Test serialization to dictionary (config only, no runtime state)."""
        config = RateLimitConfig(capacity=10, refill_rate=2.5, initial_tokens=7)
        bucket = TokenBucket(config)

        result = bucket.to_dict()
        assert result["capacity"] == 10
        assert result["refill_rate"] == 2.5
        # Runtime state is always excluded (YAGNI - no persistence use case)
        assert "tokens" not in result
        assert "last_refill" not in result

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity_raises(self):
        """Test that acquiring more than capacity raises ValueError."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=10)
        bucket = TokenBucket(config)

        # Requesting more tokens than capacity would wait forever
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await bucket.acquire(tokens=20)

    @pytest.mark.asyncio
    async def test_acquire_zero_tokens_raises(self):
        """Test that acquiring zero tokens raises ValueError."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.acquire(tokens=0)

    @pytest.mark.asyncio
    async def test_acquire_negative_tokens_raises(self):
        """Test that acquiring negative tokens raises ValueError."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.acquire(tokens=-5)

    @pytest.mark.asyncio
    async def test_try_acquire_zero_tokens_raises(self):
        """Test that try_acquire with zero tokens raises ValueError."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.try_acquire(tokens=0)

    @pytest.mark.asyncio
    async def test_release_zero_tokens_raises(self):
        """Test that releasing zero tokens raises ValueError."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0)
        bucket = TokenBucket(config)

        with pytest.raises(ValueError, match="tokens must be > 0"):
            await bucket.release(tokens=0)

    @pytest.mark.asyncio
    async def test_acquire_default_single_token(self):
        """Test that acquire defaults to 1 token."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=10)
        bucket = TokenBucket(config)

        result = await bucket.acquire()  # No tokens parameter
        assert result is True
        assert bucket.tokens == 9.0

    @pytest.mark.asyncio
    async def test_try_acquire_default_single_token(self):
        """Test that try_acquire defaults to 1 token."""
        config = RateLimitConfig(capacity=10, refill_rate=2.0, initial_tokens=10)
        bucket = TokenBucket(config)

        result = await bucket.try_acquire()  # No tokens parameter
        assert result is True
        assert bucket.tokens == 9.0

    @pytest.mark.asyncio
    async def test_acquire_with_none_timeout_waits_indefinitely(self):
        """Test that acquire with timeout=None will wait for tokens."""
        config = RateLimitConfig(capacity=10, refill_rate=20.0, initial_tokens=0)
        bucket = TokenBucket(config)

        # Need 5 tokens, refill rate 20/s, should wait ~0.25s
        result = await bucket.acquire(tokens=5, timeout=None)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_timeout_adjusted_for_remaining_time(self):
        """Test that timeout is adjusted correctly in wait loop."""
        config = RateLimitConfig(capacity=10, refill_rate=5.0, initial_tokens=0)
        bucket = TokenBucket(config)

        # Need 2 tokens, refill rate 5/s, needs 0.4s
        # Set timeout slightly higher to allow success
        result = await bucket.acquire(tokens=2, timeout=0.6)
        assert result is True

    @pytest.mark.asyncio
    async def test_reset_restores_full_capacity(self):
        """Test that reset() restores bucket to full capacity (lines 110-112)."""
        config = RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=10)
        bucket = TokenBucket(config)

        # Consume all tokens
        await bucket.acquire(tokens=10)
        assert bucket.tokens < 1.0

        # Reset should restore to full capacity
        await bucket.reset()
        assert bucket.tokens == 10.0

    @pytest.mark.asyncio
    async def test_reset_updates_last_refill_time(self):
        """Test that reset() updates last_refill timestamp."""
        config = RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=5)
        bucket = TokenBucket(config)

        import asyncio

        await asyncio.sleep(0.1)  # Let some time pass

        old_last_refill = bucket.last_refill
        await bucket.reset()

        # last_refill should be updated to current time
        assert bucket.last_refill >= old_last_refill

    @pytest.mark.asyncio
    async def test_release_returns_tokens_to_bucket(self):
        """Test that release() returns tokens back to bucket (lines 125-126)."""
        config = RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=10)
        bucket = TokenBucket(config)

        # Consume some tokens
        await bucket.acquire(tokens=5)
        assert bucket.tokens == 5.0

        # Release tokens back
        await bucket.release(tokens=3)
        assert bucket.tokens == 8.0

    @pytest.mark.asyncio
    async def test_release_respects_capacity_limit(self):
        """Test that release() doesn't exceed capacity."""
        config = RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=8)
        bucket = TokenBucket(config)

        # Try to release more tokens than would fit
        await bucket.release(tokens=5)

        # Should be capped at capacity
        assert bucket.tokens == 10.0

    @pytest.mark.asyncio
    async def test_release_default_single_token(self):
        """Test that release() defaults to 1 token."""
        config = RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=5)
        bucket = TokenBucket(config)

        await bucket.release()  # No tokens parameter
        assert bucket.tokens == 6.0
