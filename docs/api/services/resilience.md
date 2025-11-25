# Resilience Patterns

> Circuit breaker, retry with exponential backoff, and fail-fast error handling

## Overview

The resilience module provides production-ready patterns for handling transient failures, cascading failures, and degraded service states. It implements circuit breaker (fail-fast) and retry with exponential backoff (transient failure recovery) patterns used by `Endpoint` backends.

**Core Components:**

- **`CircuitBreaker`**: Fail-fast pattern to prevent cascading failures
- **`RetryConfig`**: Exponential backoff configuration for retry logic
- **`retry_with_backoff()`**: Async retry function with configurable backoff
- **`CircuitBreakerOpenError`**: Exception raised when circuit is open

**Key Features:**

- **Three-State Circuit**: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
- **Exponential Backoff**: Configurable base, max delay, and jitter
- **Excluded Exceptions**: Skip circuit breaker for specific exception types
- **Metrics**: Track success/failure counts and state transitions
- **Thread-Safe**: Async locks for concurrent access
- **Serialization**: Configuration serializable (runtime state not preserved)

**When to Use:**

- Protecting downstream services from cascading failures
- Implementing retry logic for transient network errors
- Degrading gracefully under partial service outages
- Preventing thundering herd during recovery

## Module Exports

```python
from lionpride.services.utilities.resilience import (
    # Circuit breaker
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,

    # Retry
    RetryConfig,
    retry_with_backoff,
)
```

## CircuitState

Enumeration of circuit breaker states.

```python
class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered
```

**State Transitions:**

```
CLOSED --[failure_threshold failures]--> OPEN
OPEN --[recovery_time elapsed]--> HALF_OPEN
HALF_OPEN --[success]--> CLOSED
HALF_OPEN --[failure]--> OPEN
```

## CircuitBreakerOpenError

Exception raised when circuit breaker is open.

### Class Signature

```python
class CircuitBreakerOpenError(ConnectionError):
    """Circuit breaker is open."""

    default_message = "Circuit breaker is open"
    default_retryable = True

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ): ...
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `retry_after` | `float \| None` | Seconds until retry should be attempted |
| `details` | `dict[str, Any]` | Additional context dict |
| `message` | `str` | Error message |
| `retryable` | `bool` | Always `True` (circuit breaker errors are retryable) |

### Example

```python
from lionpride.services.utilities.resilience import CircuitBreakerOpenError

try:
    response = await circuit.execute(api_call)
except CircuitBreakerOpenError as e:
    print(f"Circuit open, retry after {e.retry_after}s")
    print(f"Details: {e.details}")
```

## CircuitBreaker

Fail-fast circuit breaker to prevent cascading failures.

### Class Signature

```python
class CircuitBreaker:
    """Fail-fast circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_time: float = 30.0,
        half_open_max_calls: int = 1,
        excluded_exceptions: set[type[Exception]] | None = None,
        name: str = "default",
    ): ...

    @property
    def metrics(self) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]: ...

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> T: ...
```

### Parameters

- `failure_threshold` (int): Number of consecutive failures before opening circuit (default: 5, must be > 0)
- `recovery_time` (float): Seconds to wait before transitioning to HALF_OPEN (default: 30.0, must be > 0)
- `half_open_max_calls` (int): Max concurrent calls allowed in HALF_OPEN state (default: 1, must be > 0)
- `excluded_exceptions` (set[type[Exception]] | None): Exception types that don't count as failures (default: None)
- `name` (str): Circuit name for logging (default: "default")

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `failure_count` | `int` | Current consecutive failure count |
| `state` | `CircuitState` | Current circuit state |
| `last_failure_time` | `float` | Timestamp of last failure (monotonic time) |
| `metrics` | `dict[str, Any]` | Success/failure counts and state transitions (property) |

### Methods

#### `__init__()`

Initialize circuit breaker.

**Raises**:

- `ValueError`: If parameters invalid (non-positive thresholds/times)

**Warnings**:

- If `Exception` in `excluded_exceptions`: Circuit would never open

**Examples:**

```python
from lionpride.services.utilities.resilience import CircuitBreaker

# Basic circuit
circuit = CircuitBreaker(
    failure_threshold=5,
    recovery_time=30.0,
)

# With excluded exceptions (don't count validation errors)
circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_time=60.0,
    excluded_exceptions={ValueError, TypeError},
    name="api_circuit",
)

# Validation errors
circuit = CircuitBreaker(failure_threshold=0)  # ValueError: must be > 0
circuit = CircuitBreaker(recovery_time=-1.0)  # ValueError: must be > 0
```

#### `execute()`

Execute function with circuit breaker protection.

**Signature:**

```python
async def execute(
    self,
    func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any
) -> T: ...
```

**Parameters:**

- `func` (Callable): Async function to execute
- `*args`: Positional arguments for func
- `**kwargs`: Keyword arguments for func

**Returns**: Result from successful func execution

**Raises**:

- `CircuitBreakerOpenError`: Circuit is OPEN or HALF_OPEN at capacity
- Propagates exceptions from func (increments failure count if not excluded)

**State Transitions**:

- **CLOSED → OPEN**: After `failure_threshold` consecutive failures
- **OPEN → HALF_OPEN**: After `recovery_time` elapsed
- **HALF_OPEN → CLOSED**: On first success
- **HALF_OPEN → OPEN**: On any failure

**Examples:**

```python
from lionpride.services.utilities.resilience import CircuitBreaker, CircuitBreakerOpenError

circuit = CircuitBreaker(failure_threshold=3, recovery_time=30.0)

# Normal execution
try:
    result = await circuit.execute(api_call, arg1, arg2)
except CircuitBreakerOpenError as e:
    print(f"Circuit open, retry after {e.retry_after}s")
except Exception as e:
    print(f"API call failed: {e}")

# Check circuit state
print(circuit.state)  # CircuitState.CLOSED or OPEN or HALF_OPEN
print(circuit.failure_count)  # Current consecutive failures
```

#### `metrics` (property)

Get circuit breaker metrics (deep copy for thread-safety).

**Returns**: `dict[str, Any]` with:

- `success_count` (int): Total successful calls
- `failure_count` (int): Total failed calls (not consecutive)
- `rejected_count` (int): Total rejected calls (circuit open)
- `state_changes` (list[dict]): State transition history

**Note**: Not guaranteed consistent under concurrent writes (observability use only).

**Example:**

```python
metrics = circuit.metrics
print(metrics)
# {
#     "success_count": 42,
#     "failure_count": 8,
#     "rejected_count": 15,
#     "state_changes": [
#         {"time": 1234.56, "from": "closed", "to": "open"},
#         {"time": 1264.56, "from": "open", "to": "half_open"},
#         {"time": 1265.12, "from": "half_open", "to": "closed"},
#     ]
# }
```

#### `to_dict()`

Serialize circuit breaker configuration (excludes runtime state).

**Signature:**

```python
def to_dict(self) -> dict[str, Any]: ...
```

**Returns**: `dict[str, Any]` - Configuration dict with `failure_threshold`, `recovery_time`, `half_open_max_calls`, `name`

**Note**: Runtime state (`failure_count`, `state`, `last_failure_time`, `metrics`) NOT serialized.

**Example:**

```python
circuit = CircuitBreaker(failure_threshold=5, recovery_time=30.0, name="api")
config = circuit.to_dict()
# {"failure_threshold": 5, "recovery_time": 30.0, "half_open_max_calls": 1, "name": "api"}

# Deserialize (fresh state)
new_circuit = CircuitBreaker(**config)
print(new_circuit.state)  # CircuitState.CLOSED (fresh circuit)
```

## RetryConfig

Frozen configuration dataclass for exponential backoff retry.

### Class Signature

```python
@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry configuration with exponential backoff + jitter."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (
        ConnectionError,
        CircuitBreakerOpenError,
    )
```

### Parameters

- `max_retries` (int): Maximum retry attempts (default: 3, must be ≥ 0)
- `initial_delay` (float): Initial delay in seconds (default: 1.0, must be > 0)
- `max_delay` (float): Maximum delay cap in seconds (default: 60.0, must be > 0 and ≥ initial_delay)
- `exponential_base` (float): Base for exponential calculation (default: 2.0, must be > 0)
- `jitter` (bool): Add random jitter to prevent thundering herd (default: True)
- `retry_on` (tuple[type[Exception], ...]): Exception types to retry (default: `(ConnectionError, CircuitBreakerOpenError)`)

### Validation

Post-initialization validation ensures:

- `max_retries ≥ 0`
- `initial_delay > 0`
- `max_delay > 0`
- `max_delay ≥ initial_delay`
- `exponential_base > 0`

### Methods

#### `calculate_delay()`

Calculate delay with exponential backoff + optional jitter.

**Signature:**

```python
def calculate_delay(self, attempt: int) -> float: ...
```

**Parameters:**

- `attempt` (int): Current retry attempt number (0-indexed)

**Returns**: `float` - Delay in seconds before next retry

**Formula**:

```python
delay = min(initial_delay * (exponential_base ** attempt), max_delay)
if jitter:
    delay = delay * (0.5 + random() * 0.5)  # 50-100% of calculated delay
```

**Examples:**

```python
from lionpride.services.utilities.resilience import RetryConfig

config = RetryConfig(
    initial_delay=1.0,
    exponential_base=2.0,
    max_delay=60.0,
    jitter=False,
)

print(config.calculate_delay(0))  # 1.0 * (2^0) = 1.0
print(config.calculate_delay(1))  # 1.0 * (2^1) = 2.0
print(config.calculate_delay(2))  # 1.0 * (2^2) = 4.0
print(config.calculate_delay(3))  # 1.0 * (2^3) = 8.0
print(config.calculate_delay(10)) # min(1024.0, 60.0) = 60.0 (capped)
```

#### `to_dict()`

Serialize config to dict.

**Signature:**

```python
def to_dict(self) -> dict[str, Any]: ...
```

**Returns**: `dict[str, Any]` - All config fields

**Example:**

```python
config = RetryConfig(max_retries=5, initial_delay=2.0)
config_dict = config.to_dict()
# {
#     "max_retries": 5,
#     "initial_delay": 2.0,
#     "max_delay": 60.0,
#     "exponential_base": 2.0,
#     "jitter": True,
#     "retry_on": (ConnectionError, CircuitBreakerOpenError)
# }
```

#### `as_kwargs()`

Convert config to kwargs for `retry_with_backoff()`.

**Signature:**

```python
def as_kwargs(self) -> dict[str, Any]: ...
```

**Returns**: `dict[str, Any]` - Config dict (same as `to_dict()`)

**Example:**

```python
config = RetryConfig(max_retries=5)
result = await retry_with_backoff(api_call, **config.as_kwargs())
```

## retry_with_backoff()

Retry async function with exponential backoff.

### Signature

```python
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
) -> T: ...
```

### Parameters

- `func` (Callable): Async function to retry
- `*args`: Positional arguments for func
- `max_retries` (int): Maximum retry attempts (default: 3)
- `initial_delay` (float): Initial delay in seconds (default: 1.0)
- `max_delay` (float): Maximum delay cap in seconds (default: 60.0)
- `exponential_base` (float): Base for exponential calculation (default: 2.0)
- `jitter` (bool): Add random jitter to prevent thundering herd (default: True)
- `retry_on` (tuple[type[Exception], ...]): Exception types to retry (default: `(ConnectionError, CircuitBreakerOpenError)`)
- `**kwargs`: Keyword arguments for func

### Returns

Result from successful func execution

### Raises

Last exception if all retries exhausted

### Default Retry Exceptions

**Retries by default**:

- `ConnectionError`: Network transient errors
- `CircuitBreakerOpenError`: Circuit breaker open (retryable by design)

**Does NOT retry by default** (opt-in explicitly via `retry_on`):

- Programming errors: `TypeError`, `ValueError`, `AttributeError`, `KeyError`
- File system errors: `FileNotFoundError`, `PermissionError`, `OSError`
- Timeouts: `TimeoutError` (context-dependent, opt-in if needed)

### Examples

```python
from lionpride.services.utilities.resilience import retry_with_backoff

# Basic retry
result = await retry_with_backoff(api_call)

# Custom retry config
result = await retry_with_backoff(
    api_call,
    max_retries=5,
    initial_delay=2.0,
    exponential_base=3.0,
)

# Retry on additional exceptions
result = await retry_with_backoff(
    api_call,
    retry_on=(ConnectionError, TimeoutError, httpx.HTTPStatusError),
)

# With RetryConfig
from lionpride.services.utilities.resilience import RetryConfig
config = RetryConfig(max_retries=10, initial_delay=0.5)
result = await retry_with_backoff(api_call, **config.as_kwargs())
```

## Integration with Endpoint

`Endpoint` backends use circuit breaker and retry together:

```python
from lionpride.services.types.endpoint import Endpoint, EndpointConfig
from lionpride.services.utilities.resilience import CircuitBreaker, RetryConfig

config = EndpointConfig(
    provider="custom",
    name="custom_api",
    base_url="https://api.custom.com",
    endpoint="query",
    api_key="API_KEY",
    request_options=MyRequestModel,
)

endpoint = Endpoint(
    config=config,
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        recovery_time=30.0,
    ),
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        exponential_base=2.0,
    ),
)
```

**Resilience Pattern** (from `Endpoint.call()`):

1. Retry wraps circuit breaker (outer)
2. Circuit breaker wraps `_call()` (inner)
3. Each retry attempt counts against circuit breaker metrics

```python
# Conceptual flow
async def call(request):
    base_call = _call  # HTTP request

    # Step 1: Wrap with circuit breaker
    if circuit_breaker:
        cb_call = lambda p, h, **kw: circuit_breaker.execute(base_call, p, h, **kw)
    else:
        cb_call = base_call

    # Step 2: Wrap with retry
    if retry_config:
        response = await retry_with_backoff(cb_call, payload, headers, **retry_config.as_kwargs())
    else:
        response = await cb_call(payload, headers)

    return normalize_response(response)
```

**Why this order?**

- Retry outer ensures each retry attempt is protected by circuit breaker
- Circuit breaker metrics track all attempts (not just final failure)
- Circuit opens after `failure_threshold` attempts across all retries

## Usage Patterns

### Basic Usage: Circuit Breaker Only

```python
from lionpride.services.utilities.resilience import CircuitBreaker

circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_time=30.0,
    name="api_circuit",
)

# Execute protected call
try:
    result = await circuit.execute(api_call, arg1, arg2)
except CircuitBreakerOpenError as e:
    print(f"Circuit open, retry after {e.retry_after}s")
```

### Basic Usage: Retry Only

```python
from lionpride.services.utilities.resilience import retry_with_backoff

# Retry with defaults
result = await retry_with_backoff(
    api_call,
    max_retries=3,
    initial_delay=1.0,
)
```

### Advanced Usage: Circuit Breaker + Retry

```python
from lionpride.services.utilities.resilience import CircuitBreaker, retry_with_backoff

circuit = CircuitBreaker(failure_threshold=5, recovery_time=30.0)

# Retry wraps circuit breaker
result = await retry_with_backoff(
    circuit.execute,
    api_call,
    max_retries=3,
    initial_delay=1.0,
    exponential_base=2.0,
)
```

### Advanced Usage: Excluded Exceptions

```python
from lionpride.services.utilities.resilience import CircuitBreaker

# Don't count validation errors as circuit failures
circuit = CircuitBreaker(
    failure_threshold=3,
    recovery_time=60.0,
    excluded_exceptions={ValueError, TypeError, KeyError},
)

# ValidationError doesn't increment failure_count
try:
    result = await circuit.execute(api_call_with_validation)
except ValueError as e:
    print("Validation error (circuit not affected)")
```

### Advanced Usage: Metrics Tracking

```python
from lionpride.services.utilities.resilience import CircuitBreaker
import asyncio

circuit = CircuitBreaker(failure_threshold=3, recovery_time=30.0)

# Background metrics logger
async def log_metrics():
    while True:
        await asyncio.sleep(60)
        metrics = circuit.metrics
        print(f"Success: {metrics['success_count']}, "
              f"Failure: {metrics['failure_count']}, "
              f"Rejected: {metrics['rejected_count']}, "
              f"State: {circuit.state.value}")

asyncio.create_task(log_metrics())
```

## Common Pitfalls

### Issue: Excluding base Exception

**Problem**: Excluding `Exception` means circuit never opens.

```python
# ❌ WRONG: Circuit never opens
circuit = CircuitBreaker(
    failure_threshold=5,
    excluded_exceptions={Exception},  # All exceptions excluded
)
# UserWarning: excluding base Exception means circuit will never open
```

**Solution**: Only exclude specific exception types.

```python
# ✅ CORRECT: Exclude validation errors only
circuit = CircuitBreaker(
    failure_threshold=5,
    excluded_exceptions={ValueError, TypeError},
)
```

### Issue: Retry + Circuit Breaker wrong order

**Problem**: Circuit breaker wrapping retry means only final failure tracked.

```python
# ❌ WRONG: Circuit only sees final failure after all retries
async def call():
    async def retry_wrapped():
        return await retry_with_backoff(api_call, max_retries=5)
    return await circuit.execute(retry_wrapped)
```

**Solution**: Retry wraps circuit breaker (each attempt tracked).

```python
# ✅ CORRECT: Circuit sees all retry attempts
async def call():
    return await retry_with_backoff(
        circuit.execute,
        api_call,
        max_retries=5,
    )
```

### Issue: Circuit state not preserved on serialization

**Problem**: Assuming circuit state is serialized.

```python
# ❌ WRONG ASSUMPTION
circuit = CircuitBreaker(failure_threshold=3)
# Trigger 2 failures (not yet open)
for _ in range(2):
    try:
        await circuit.execute(failing_call)
    except:
        pass

config = circuit.to_dict()
new_circuit = CircuitBreaker(**config)
print(new_circuit.failure_count)  # 0 (fresh circuit, not 2)
```

**Solution**: Understand serialization resets state.

```python
# ✅ CORRECT UNDERSTANDING
# Serialization stores configuration, not state
# Deserialization creates fresh circuit with CLOSED state
```

### Issue: Retry timeout exceeds max_delay

**Problem**: Individual operation timeout exceeds max retry delay.

```python
# ❌ WRONG: timeout=120s, but max_delay=60s
async def slow_call():
    # Timeout after 120s
    return await api_call(timeout=120.0)

result = await retry_with_backoff(
    slow_call,
    max_retries=5,
    initial_delay=10.0,
    max_delay=60.0,  # Max wait is 60s, but call might take 120s
)
```

**Solution**: Ensure max_delay ≥ operation timeout.

```python
# ✅ CORRECT: max_delay >= operation timeout
result = await retry_with_backoff(
    slow_call,
    max_retries=5,
    initial_delay=10.0,
    max_delay=150.0,  # Allow enough time for slow operations
)
```

## Design Rationale

### Why three-state circuit?

Three states (CLOSED, OPEN, HALF_OPEN) enable graceful recovery:

1. **CLOSED**: Normal operation, track failures
2. **OPEN**: Fail-fast to prevent cascading failures
3. **HALF_OPEN**: Test if service recovered (limited traffic)

Without HALF_OPEN, circuit would oscillate between OPEN and CLOSED on first recovery attempt.

### Why monotonic time for recovery?

`libs.concurrency.current_time()` uses `time.monotonic()`:

- **Immune to system clock changes**: Manual adjustments don't affect recovery timing
- **Monotonically increasing**: Never goes backward
- **High resolution**: Sufficient precision for sub-second timing

### Why jitter in retry backoff?

Jitter prevents thundering herd:

- Without jitter: All failed clients retry at exact same time (synchronized)
- With jitter: Retry times spread over 50-100% of calculated delay (desynchronized)

This reduces load spikes during recovery.

### Why ConnectionError as default retry?

`ConnectionError` covers transient network errors:

- DNS resolution failures
- TCP connection timeouts
- Connection refused
- Network unreachable

These are typically retryable (vs programming errors like `ValueError` which are not).

## See Also

- **Integration**:
  - [`Endpoint`](providers.md#endpoint): Uses circuit breaker and retry
  - [`iModel`](imodel.md): Unified service interface
- **Rate Limiting**:
  - [`TokenBucket`](rate_limiting.md#tokenbucket): Rate limiting
- **Errors**:
  - [ConnectionError](../errors.md): Base error for network failures
- **Concurrency**:
  - [libs.concurrency](../libs/concurrency.md): Lock, sleep, current_time

## Examples

### Example 1: API with Circuit Breaker and Retry

```python
from lionpride.services.utilities.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    RetryConfig,
    retry_with_backoff,
)

circuit = CircuitBreaker(
    failure_threshold=5,
    recovery_time=60.0,
    name="api_circuit",
)

retry_config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    exponential_base=2.0,
)

async def resilient_api_call(data):
    """Call API with circuit breaker + retry."""
    return await retry_with_backoff(
        circuit.execute,
        api_call,
        data,
        **retry_config.as_kwargs()
    )

# Use
try:
    result = await resilient_api_call({"query": "test"})
except CircuitBreakerOpenError as e:
    print(f"Circuit open, retry after {e.retry_after}s")
except Exception as e:
    print(f"All retries exhausted: {e}")
```

### Example 2: Adaptive Circuit Breaker

```python
from lionpride.services.utilities.resilience import CircuitBreaker

class AdaptiveCircuit:
    def __init__(self):
        self.circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_time=30.0,
        )
        self.success_streak = 0

    async def execute(self, func, *args, **kwargs):
        try:
            result = await self.circuit.execute(func, *args, **kwargs)
            self.success_streak += 1

            # After 10 consecutive successes, reduce recovery time
            if self.success_streak >= 10:
                self.circuit.recovery_time = max(10.0, self.circuit.recovery_time * 0.8)
                self.success_streak = 0

            return result
        except Exception:
            self.success_streak = 0

            # After repeated failures, increase recovery time
            if self.circuit.failure_count >= 3:
                self.circuit.recovery_time = min(120.0, self.circuit.recovery_time * 1.2)

            raise
```

### Example 3: Custom Retry Logic with Hooks

```python
from lionpride.services.utilities.resilience import retry_with_backoff

# Track retry attempts
retry_log = []

async def logged_retry(func, *args, **kwargs):
    attempt = 0

    async def wrapper():
        nonlocal attempt
        attempt += 1
        retry_log.append({"attempt": attempt, "timestamp": time.time()})
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            retry_log[-1]["error"] = str(e)
            raise

    return await retry_with_backoff(
        wrapper,
        max_retries=5,
        initial_delay=1.0,
    )

# Use
result = await logged_retry(api_call, data)
print(retry_log)
# [
#     {"attempt": 1, "timestamp": 1234.56, "error": "Timeout"},
#     {"attempt": 2, "timestamp": 1236.12, "error": "Timeout"},
#     {"attempt": 3, "timestamp": 1240.45},
# ]
```
