# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for retry logic using lionpride-core primitives."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from lionpride.services.utilities.resilience import (
    RetryConfig,
    retry_with_backoff,
)


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from lionpride.errors import ConnectionError
        from lionpride.services.utilities.resilience import CircuitBreakerOpenError

        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retry_on == (ConnectionError, CircuitBreakerOpenError)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            retry_on=(ValueError, TypeError),
        )
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retry_on == (ValueError, TypeError)

    def test_calculate_delay_without_jitter(self):
        """Test delay calculation without jitter."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False)

        # Test exponential backoff: delay = initial * base^attempt
        assert config.calculate_delay(0) == 1.0  # 1.0 * 2^0 = 1.0
        assert config.calculate_delay(1) == 2.0  # 1.0 * 2^1 = 2.0
        assert config.calculate_delay(2) == 4.0  # 1.0 * 2^2 = 4.0
        assert config.calculate_delay(3) == 8.0  # 1.0 * 2^3 = 8.0

    def test_calculate_delay_with_max_cap(self):
        """Test delay capped at max_delay."""
        config = RetryConfig(initial_delay=10.0, exponential_base=2.0, max_delay=30.0, jitter=False)

        # delay = 10 * 2^3 = 80, but capped at 30
        assert config.calculate_delay(3) == 30.0

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(initial_delay=10.0, exponential_base=2.0, max_delay=60.0, jitter=True)

        # With jitter, delay should be in range [0.5 * base_delay, 1.0 * base_delay]
        for attempt in range(5):
            delay = config.calculate_delay(attempt)
            base_delay = min(10.0 * (2.0**attempt), 60.0)
            assert 0.5 * base_delay <= delay <= base_delay

    def test_as_kwargs(self):
        """Test conversion to kwargs dict."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=100.0,
            exponential_base=3.0,
            jitter=False,
            retry_on=(ValueError,),
        )

        kwargs = config.as_kwargs()
        assert kwargs == {
            "max_retries": 5,
            "initial_delay": 2.0,
            "max_delay": 100.0,
            "exponential_base": 3.0,
            "jitter": False,
            "retry_on": (ValueError,),
        }


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test function succeeds on first attempt."""
        mock_func = AsyncMock(return_value="success")

        result = await retry_with_backoff(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Test function succeeds after N retries."""
        mock_func = AsyncMock(side_effect=[ValueError("fail1"), ValueError("fail2"), "success"])

        with patch("lionpride.services.utilities.resilience.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(
                mock_func, max_retries=3, initial_delay=0.1, retry_on=(ValueError,)
            )

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test raises exception when all retries exhausted."""
        mock_func = AsyncMock(side_effect=ValueError("persistent failure"))

        with (
            patch("lionpride.services.utilities.resilience.sleep", new_callable=AsyncMock),
            pytest.raises(ValueError, match="persistent failure"),
        ):
            await retry_with_backoff(
                mock_func, max_retries=2, initial_delay=0.1, retry_on=(ValueError,)
            )

        # Should be called: initial + 2 retries = 3 times
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_selective_retry(self):
        """Test only retries on specified exception types."""
        # ValueError should be retried, TypeError should not
        mock_func = AsyncMock(side_effect=TypeError("wrong type"))

        with pytest.raises(TypeError, match="wrong type"):
            await retry_with_backoff(
                mock_func,
                max_retries=3,
                retry_on=(ValueError,),  # Only retry ValueError
            )

        # Should fail immediately without retry
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Test exponential backoff delay progression."""
        mock_func = AsyncMock(
            side_effect=[ValueError("1"), ValueError("2"), ValueError("3"), "success"]
        )
        mock_sleep = AsyncMock()

        with patch("lionpride.services.utilities.resilience.sleep", mock_sleep):
            result = await retry_with_backoff(
                mock_func,
                max_retries=3,
                initial_delay=1.0,
                exponential_base=2.0,
                jitter=False,  # Disable jitter for predictable delays
                retry_on=(ValueError,),
            )

        assert result == "success"
        assert mock_func.call_count == 4

        # Check sleep was called with exponential delays
        assert mock_sleep.call_count == 3
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls[0] == 1.0  # 1.0 * 2^0
        assert sleep_calls[1] == 2.0  # 1.0 * 2^1
        assert sleep_calls[2] == 4.0  # 1.0 * 2^2

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        mock_func = AsyncMock(side_effect=[ValueError("1"), ValueError("2"), "success"])
        mock_sleep = AsyncMock()

        with patch("lionpride.services.utilities.resilience.sleep", mock_sleep):
            result = await retry_with_backoff(
                mock_func,
                max_retries=2,
                initial_delay=10.0,
                max_delay=15.0,  # Cap delay at 15
                exponential_base=2.0,
                jitter=False,
                retry_on=(ValueError,),
            )

        assert result == "success"

        # Check delays are capped
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls[0] == 10.0  # 10.0 * 2^0 = 10.0
        assert sleep_calls[1] == 15.0  # 10.0 * 2^1 = 20.0, capped at 15.0

    @pytest.mark.asyncio
    async def test_jitter_randomization(self):
        """Test jitter adds randomization to delays."""
        mock_func = AsyncMock(side_effect=[ValueError("1"), ValueError("2"), "success"])
        mock_sleep = AsyncMock()

        with patch("lionpride.services.utilities.resilience.sleep", mock_sleep):
            result = await retry_with_backoff(
                mock_func,
                max_retries=2,
                initial_delay=10.0,
                max_delay=100.0,
                exponential_base=2.0,
                jitter=True,
                retry_on=(ValueError,),
            )

        assert result == "success"

        # With jitter, delays should be in range [0.5 * base, 1.0 * base]
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]

        # First retry: base = 10.0 * 2^0 = 10.0
        assert 5.0 <= sleep_calls[0] <= 10.0

        # Second retry: base = 10.0 * 2^1 = 20.0
        assert 10.0 <= sleep_calls[1] <= 20.0

    @pytest.mark.asyncio
    async def test_multiple_exception_types(self):
        """Test retrying on multiple exception types."""
        # Create a mock that fails with different exception types
        mock_func = AsyncMock(side_effect=[ValueError("val"), TypeError("type"), "success"])

        with patch("lionpride.services.utilities.resilience.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(
                mock_func, max_retries=3, initial_delay=0.1, retry_on=(ValueError, TypeError)
            )

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_immediate_failure_without_retry(self):
        """Test immediate failure when exception not in retry_on."""
        # Use a simple exception that won't cause logging issues
        mock_func = AsyncMock(side_effect=RuntimeError("not retryable"))

        with pytest.raises(RuntimeError, match="not retryable"):
            await retry_with_backoff(
                mock_func,
                max_retries=3,
                retry_on=(ValueError,),  # Only retry ValueError, not RuntimeError
            )

        # Should fail immediately - no retries
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_different_error_types(self):
        """Test retrying different error types in sequence."""
        # Simulate recoverable errors of different types, then success
        mock_func = AsyncMock(
            side_effect=[ConnectionError("network error"), TimeoutError("timeout"), "success"]
        )

        with patch("lionpride.services.utilities.resilience.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(
                mock_func,
                max_retries=3,
                initial_delay=0.1,
                retry_on=(ConnectionError, TimeoutError),
            )

        assert result == "success"
        assert mock_func.call_count == 3
