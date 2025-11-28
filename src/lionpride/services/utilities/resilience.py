# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from lionpride.errors import ConnectionError
from lionpride.libs.concurrency import Lock, current_time, sleep

__all__ = (
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "RetryConfig",
    "retry_with_backoff",
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(ConnectionError):
    """Circuit breaker is open."""

    default_message = "Circuit breaker is open"
    default_retryable = True  # Circuit breaker errors are inherently retryable

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize with message and optional retry_after.

        Args:
            message: Error message (uses default_message if None)
            retry_after: Seconds until retry should be attempted
            details: Additional context dict
        """
        # Add retry_after to details if provided
        if retry_after is not None:
            details = details or {}
            details["retry_after"] = retry_after

        super().__init__(message=message, details=details, retryable=True)
        self.retry_after = retry_after


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Fail-fast circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_time: float = 30.0,
        half_open_max_calls: int = 1,
        excluded_exceptions: set[type[Exception]] | None = None,
        name: str = "default",
    ):
        """Initialize circuit breaker.

        Raises:
            ValueError: If parameters are invalid (non-positive thresholds/times)
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if recovery_time <= 0:
            raise ValueError("recovery_time must be > 0")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be > 0")

        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or set()
        self.name = name

        # Warn if Exception in excluded_exceptions (circuit would never open)
        if Exception in self.excluded_exceptions:
            logger.warning(
                f"CircuitBreaker '{name}': excluding base Exception means circuit will never open"
            )

        # State variables
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = Lock()  # libs.concurrency.Lock

        # Metrics
        self._metrics = {
            "success_count": 0,
            "failure_count": 0,
            "rejected_count": 0,
            "state_changes": [],
        }

        logger.debug(
            f"Initialized CircuitBreaker '{self.name}' with failure_threshold={failure_threshold}, "
            f"recovery_time={recovery_time}, half_open_max_calls={half_open_max_calls}"
        )

    @property
    def metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics (deep copy for thread-safety)."""
        # Note: This is a sync property, so we can't use async lock here.
        # For observability use only - not guaranteed consistent under concurrent writes.
        return {
            "success_count": self._metrics["success_count"],
            "failure_count": self._metrics["failure_count"],
            "rejected_count": self._metrics["rejected_count"],
            "state_changes": list(self._metrics["state_changes"]),  # Deep copy list
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit breaker configuration."""
        return {
            "failure_threshold": self.failure_threshold,
            "recovery_time": self.recovery_time,
            "half_open_max_calls": self.half_open_max_calls,
            "name": self.name,
        }

    async def _change_state(self, new_state: CircuitState) -> None:
        """Change state with logging."""
        old_state = self.state
        if new_state != old_state:
            self.state = new_state
            self._metrics["state_changes"].append(
                {
                    "time": current_time(),
                    "from": old_state.value,
                    "to": new_state.value,
                }
            )

            logger.info(
                f"Circuit '{self.name}' state changed from {old_state.value} to {new_state.value}"
            )

            # Reset counters on state change
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
            elif new_state == CircuitState.CLOSED:
                self.failure_count = 0

    async def _check_state(self) -> tuple[bool, float]:
        """Check if request can proceed.

        Returns:
            Tuple of (can_proceed, retry_after_seconds)

        Note:
            Uses monotonic clock (via anyio.current_time() → time.monotonic())
            to ensure recovery_time calculations are immune to system clock changes.
        """
        async with self._lock:
            now = current_time()  # Monotonic time from event loop

            if self.state == CircuitState.OPEN:
                # Check if recovery time has elapsed
                if now - self.last_failure_time >= self.recovery_time:
                    await self._change_state(CircuitState.HALF_OPEN)
                else:
                    recovery_remaining = self.recovery_time - (now - self.last_failure_time)
                    self._metrics["rejected_count"] += 1

                    logger.warning(
                        f"Circuit '{self.name}' is OPEN, rejecting request. "
                        f"Try again in {recovery_remaining:.2f}s"
                    )

                    return False, recovery_remaining

            if self.state == CircuitState.HALF_OPEN:
                # Only allow a limited number of calls in half-open state
                if self._half_open_calls >= self.half_open_max_calls:
                    self._metrics["rejected_count"] += 1

                    logger.warning(
                        f"Circuit '{self.name}' is HALF_OPEN and at capacity. Try again later."
                    )

                    return False, self.recovery_time

                self._half_open_calls += 1

            return True, 0.0

    async def execute(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute with circuit breaker protection."""
        # Check if circuit allows this call (atomically returns retry_after to avoid TOCTOU)
        can_proceed, retry_after = await self._check_state()
        if not can_proceed:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open. Retry after {retry_after:.2f} seconds",
                retry_after=retry_after,
            )

        try:
            logger.debug(
                f"Executing {func.__name__} with circuit '{self.name}' state: {self.state.value}"
            )
            result = await func(*args, **kwargs)

            # Handle success
            async with self._lock:
                self._metrics["success_count"] += 1

                # On success in half-open state, close the circuit
                if self.state == CircuitState.HALF_OPEN:
                    await self._change_state(CircuitState.CLOSED)

            return result

        except Exception as e:
            # Determine if this exception should count as a circuit failure
            is_excluded = any(isinstance(e, exc_type) for exc_type in self.excluded_exceptions)

            if not is_excluded:
                async with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = current_time()
                    self._metrics["failure_count"] += 1

                    # Log failure
                    logger.warning(
                        f"Circuit '{self.name}' failure: {e}. "
                        f"Count: {self.failure_count}/{self.failure_threshold}"
                    )

                    # Check if we need to open the circuit
                    if (
                        self.state == CircuitState.CLOSED
                        and self.failure_count >= self.failure_threshold
                    ) or self.state == CircuitState.HALF_OPEN:
                        await self._change_state(CircuitState.OPEN)

            logger.exception(f"Circuit breaker '{self.name}' caught exception")
            raise


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry configuration with exponential backoff + jitter."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = field(
        default=(ConnectionError, CircuitBreakerOpenError)
    )

    def __post_init__(self):
        """Validate retry configuration parameters."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be > 0")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be > 0")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if self.exponential_base <= 0:
            raise ValueError("exponential_base must be > 0")

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + optional jitter.

        Args:
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        delay = min(self.initial_delay * (self.exponential_base**attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
            "retry_on": self.retry_on,
        }

    def as_kwargs(self) -> dict[str, Any]:
        """Convert config to kwargs for retry_with_backoff."""
        return self.to_dict()


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] = (
        ConnectionError,
        CircuitBreakerOpenError,
    ),
    **kwargs,
) -> T:
    """Retry async function with exponential backoff + jitter.

    Only retries on network transient errors by default. Does NOT retry
    programming errors, file system errors, or timeouts (opt-in explicitly).

    Raises last exception if all retries exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retry_on as e:
            last_exception = e

            # If this was the last attempt, raise the exception
            if attempt >= max_retries:
                logger.error(f"All {max_retries} retry attempts exhausted for {func.__name__}: {e}")
                raise

            # Calculate delay with exponential backoff
            delay = min(initial_delay * (exponential_base**attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)

            logger.debug(
                f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__} "
                f"after {delay:.2f}s: {e}"
            )

            await sleep(delay)

    # This should never be reached, but for type safety  # pragma: no cover
    if last_exception:  # pragma: no cover
        raise last_exception  # pragma: no cover
    raise RuntimeError("Unexpected retry loop exit")  # pragma: no cover
