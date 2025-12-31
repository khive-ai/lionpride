# Service Utilities

> Rate limiting, resilience patterns, and token calculation utilities for service
> operations.

## Overview

The `services.utilities` module provides infrastructure primitives for building reliable
LLM service integrations. These utilities are used internally by `Endpoint` and `iModel`
but can also be used directly for custom service implementations.

**Key Capabilities:**

- **Rate Limiting**: Token bucket algorithm for request and token rate limits
- **Resilience**: Circuit breaker pattern and exponential backoff retry
- **Token Calculation**: Tiktoken-based token estimation for rate limiting

These utilities follow proven patterns from production systems, providing thread-safe
async operations with proper error handling and observability.

---

## Rate Limiting

### RateLimitConfig

> Immutable configuration for token bucket rate limiting.

#### Class Signature

```python
@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """Token bucket rate limiting configuration."""

    capacity: int
    refill_rate: float
    initial_tokens: int | None = None
```

#### Parameters

| Parameter        | Type          | Default | Description                                                                   |
| ---------------- | ------------- | ------- | ----------------------------------------------------------------------------- |
| `capacity`       | `int`         | -       | Maximum tokens the bucket can hold. Must be > 0.                              |
| `refill_rate`    | `float`       | -       | Tokens added per second. Must be > 0.                                         |
| `initial_tokens` | `int or None` | `None`  | Starting token count. Defaults to `capacity`. Must be >= 0 and <= `capacity`. |

#### Validation

The configuration validates parameters on construction:

- `capacity` must be positive
- `refill_rate` must be positive
- `initial_tokens` cannot exceed `capacity` or be negative

```python
from lionpride.services.utilities import RateLimitConfig

# Valid configurations
config = RateLimitConfig(capacity=100, refill_rate=10.0)  # 100 tokens, 10/sec refill
config = RateLimitConfig(capacity=100, refill_rate=10.0, initial_tokens=50)  # Start half-full

# Invalid configurations raise ValueError
RateLimitConfig(capacity=0, refill_rate=10.0)       # capacity must be > 0
RateLimitConfig(capacity=100, refill_rate=0)        # refill_rate must be > 0
RateLimitConfig(capacity=100, refill_rate=10.0, initial_tokens=150)  # exceeds capacity
```

---

### TokenBucket

> Async token bucket rate limiter with thread-safe operations.

The token bucket algorithm allows bursts up to `capacity` while maintaining a long-term
average rate of `refill_rate` tokens per second. Tokens are consumed on acquire and
refilled continuously based on elapsed time.

#### Class Signature

```python
class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig) -> None: ...
```

#### Attributes

| Attribute     | Type    | Description                                        |
| ------------- | ------- | -------------------------------------------------- |
| `capacity`    | `int`   | Maximum bucket capacity (from config)              |
| `refill_rate` | `float` | Tokens per second refill rate (from config)        |
| `tokens`      | `float` | Current available tokens (runtime state)           |
| `last_refill` | `float` | Monotonic timestamp of last refill (runtime state) |

#### Methods

##### `acquire()`

Acquire tokens, waiting if necessary until tokens are available or timeout expires.

```python
async def acquire(
    self,
    tokens: int = 1,
    *,
    timeout: float | None = None,
) -> bool: ...
```

**Parameters:**

- `tokens` (int): Number of tokens to acquire. Must be > 0 and <= `capacity`.
- `timeout` (float or None): Maximum wait time in seconds. `None` waits indefinitely.

**Returns:**

- `bool`: `True` if tokens acquired, `False` if timeout expired.

**Raises:**

- `ValueError`: If `tokens <= 0` or `tokens > capacity`.

**Example:**

```python
from lionpride.services.utilities import RateLimitConfig, TokenBucket

config = RateLimitConfig(capacity=10, refill_rate=2.0)  # 2 tokens/sec
bucket = TokenBucket(config)

# Acquire with timeout
if await bucket.acquire(tokens=5, timeout=10.0):
    # Proceed with rate-limited operation
    pass
else:
    # Timeout expired, handle rate limit
    pass
```

---

##### `try_acquire()`

Attempt to acquire tokens immediately without waiting.

```python
async def try_acquire(self, tokens: int = 1) -> bool: ...
```

**Parameters:**

- `tokens` (int): Number of tokens to acquire. Must be > 0.

**Returns:**

- `bool`: `True` if tokens acquired immediately, `False` if insufficient tokens.

**Raises:**

- `ValueError`: If `tokens <= 0`.

**Example:**

```python
# Non-blocking rate limit check
if await bucket.try_acquire(tokens=1):
    # Token available, proceed
    await make_request()
else:
    # Rate limited, skip or queue
    pass
```

---

##### `release()`

Release tokens back to the bucket (for atomic rollback scenarios).

```python
async def release(self, tokens: int = 1) -> None: ...
```

**Parameters:**

- `tokens` (int): Number of tokens to release. Must be > 0.

**Raises:**

- `ValueError`: If `tokens <= 0`.

Used when a dual-bucket acquire fails (e.g., request bucket succeeds but token bucket
fails) to restore the request bucket.

---

##### `reset()`

Reset bucket to full capacity.

```python
async def reset(self) -> None: ...
```

Used by `RateLimitedProcessor` for interval-based capacity replenishment (e.g., reset
every 60 seconds for API rate limits).

---

##### `to_dict()`

Serialize configuration for persistence.

```python
def to_dict(self) -> dict[str, float]: ...
```

**Returns:**

- `dict`: Configuration dict with `capacity` and `refill_rate`. Excludes runtime state.

---

## Resilience

### CircuitState

> Enumeration of circuit breaker states.

```python
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failing, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed
```

**State Transitions:**

```text
CLOSED --[failures >= threshold]--> OPEN --[recovery_time elapsed]--> HALF_OPEN
                                                                          |
                    +--[success]--+                    +--[failure]--+    |
                    v             |                    v             |    |
                 CLOSED <---------+                  OPEN <----------+----+
```

---

### CircuitBreakerOpenError

> Exception raised when circuit breaker rejects a request.

```python
class CircuitBreakerOpenError(LionConnectionError):
    """Circuit breaker is open."""

    default_message = "Circuit breaker is open"
    default_retryable = True

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None: ...
```

**Attributes:**

| Attribute     | Type            | Description                                                     |
| ------------- | --------------- | --------------------------------------------------------------- |
| `retry_after` | `float or None` | Seconds until retry should be attempted                         |
| `retryable`   | `bool`          | Always `True` (circuit breaker errors are inherently retryable) |

---

### CircuitBreaker

> Fail-fast circuit breaker for service resilience.

The circuit breaker pattern prevents cascading failures by temporarily rejecting
requests to a failing service. After a recovery period, it allows limited test requests
to check if the service has recovered.

#### Class Signature

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
    ) -> None: ...
```

#### Parameters

| Parameter             | Type                           | Default     | Description                                               |
| --------------------- | ------------------------------ | ----------- | --------------------------------------------------------- |
| `failure_threshold`   | `int`                          | `5`         | Consecutive failures before opening circuit. Must be > 0. |
| `recovery_time`       | `float`                        | `30.0`      | Seconds to wait before testing recovery. Must be > 0.     |
| `half_open_max_calls` | `int`                          | `1`         | Max test calls in half-open state. Must be > 0.           |
| `excluded_exceptions` | `set[type[Exception]] or None` | `None`      | Exception types that do not count as failures.            |
| `name`                | `str`                          | `"default"` | Circuit name for logging and metrics.                     |

#### Attributes

| Attribute           | Type             | Description                                                            |
| ------------------- | ---------------- | ---------------------------------------------------------------------- |
| `state`             | `CircuitState`   | Current circuit state                                                  |
| `failure_count`     | `int`            | Consecutive failure count                                              |
| `last_failure_time` | `float`          | Monotonic timestamp of last failure                                    |
| `metrics`           | `dict[str, Any]` | Observability metrics (success/failure/rejected counts, state changes) |

#### Methods

##### `execute()`

Execute an async function with circuit breaker protection.

```python
async def execute(
    self,
    func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T: ...
```

**Parameters:**

- `func` (Callable): Async function to execute.
- `*args`: Positional arguments for `func`.
- `**kwargs`: Keyword arguments for `func`.

**Returns:**

- `T`: Result from `func`.

**Raises:**

- `CircuitBreakerOpenError`: If circuit is open or half-open at capacity.
- Any exception from `func` (re-raised after recording failure).

**Example:**

```python
from lionpride.services.utilities import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=3,
    recovery_time=60.0,
    name="openai_api",
)

async def call_api(payload: dict) -> dict:
    # Make API call
    return await http_client.post("/chat/completions", json=payload)

try:
    result = await cb.execute(call_api, {"model": "gpt-4o", "messages": [...]})
except CircuitBreakerOpenError as e:
    # Circuit is open, fail fast
    print(f"Service unavailable, retry after {e.retry_after:.1f}s")
except Exception as e:
    # API error (recorded as failure)
    print(f"API error: {e}")
```

---

##### `to_dict()`

Serialize circuit breaker configuration.

```python
def to_dict(self) -> dict[str, Any]: ...
```

**Returns:**

- `dict`: Configuration with `failure_threshold`, `recovery_time`,
  `half_open_max_calls`, `name`.

---

### RetryConfig

> Immutable configuration for exponential backoff retry.

#### Class Signature

```python
@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry configuration with exponential backoff + jitter."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (LionConnectionError, CircuitBreakerOpenError)
```

#### Parameters

| Parameter          | Type                          | Default                                          | Description                                    |
| ------------------ | ----------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| `max_retries`      | `int`                         | `3`                                              | Maximum retry attempts. Must be >= 0.          |
| `initial_delay`    | `float`                       | `1.0`                                            | First retry delay in seconds. Must be > 0.     |
| `max_delay`        | `float`                       | `60.0`                                           | Maximum delay cap. Must be >= `initial_delay`. |
| `exponential_base` | `float`                       | `2.0`                                            | Backoff multiplier. Must be > 0.               |
| `jitter`           | `bool`                        | `True`                                           | Add randomness to prevent thundering herd.     |
| `retry_on`         | `tuple[type[Exception], ...]` | `(LionConnectionError, CircuitBreakerOpenError)` | Exception types to retry.                      |

#### Methods

##### `calculate_delay()`

Calculate delay for a specific retry attempt.

```python
def calculate_delay(self, attempt: int) -> float: ...
```

**Parameters:**

- `attempt` (int): Current retry attempt (0-indexed).

**Returns:**

- `float`: Delay in seconds before next retry.

**Formula:**

```python
delay = min(initial_delay * (exponential_base ** attempt), max_delay)
if jitter:
    delay = delay * (0.5 + random() * 0.5)  # 50-100% of calculated delay
```

---

##### `to_dict()` / `as_kwargs()`

Serialize configuration for persistence or function call.

```python
def to_dict(self) -> dict[str, Any]: ...
def as_kwargs(self) -> dict[str, Any]: ...  # Same as to_dict()
```

---

### retry_with_backoff()

> Retry an async function with exponential backoff and jitter.

```python
async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] = (LionConnectionError, CircuitBreakerOpenError),
    **kwargs,
) -> T: ...
```

**Parameters:**

Same as `RetryConfig`, plus:

- `func` (Callable): Async function to execute.
- `*args`: Positional arguments for `func`.
- `**kwargs`: Keyword arguments for `func`.

**Returns:**

- `T`: Result from successful `func` call.

**Raises:**

- Last exception if all retries exhausted.
- Any exception not in `retry_on` (raised immediately, no retry).

**Example:**

```python
from lionpride.services.utilities import retry_with_backoff
from lionpride.errors import LionConnectionError

async def flaky_api_call(url: str) -> dict:
    response = await http_client.get(url)
    if response.status_code >= 500:
        raise LionConnectionError("Server error")
    return response.json()

# Retry up to 3 times with exponential backoff
result = await retry_with_backoff(
    flaky_api_call,
    "https://api.example.com/data",
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
)
```

**Using RetryConfig:**

```python
from lionpride.services.utilities import RetryConfig, retry_with_backoff

config = RetryConfig(max_retries=5, initial_delay=0.5, max_delay=30.0)
result = await retry_with_backoff(flaky_api_call, url, **config.as_kwargs())
```

---

## Token Calculation

### TokenCalculationError

> Exception raised when token calculation fails.

```python
class TokenCalculationError(Exception):
    """Raised when token calculation fails due to encoding/model errors."""
    pass
```

Raised for encoding failures, invalid inputs, or unsupported content types.

---

### TokenCalculator

> Token counting utilities using tiktoken.

Static methods for estimating token usage in chat messages and embeddings. Used by
`APICalling` for rate limiting token estimation.

#### Methods

##### `calculate_message_tokens()`

Calculate token count for chat messages.

```python
@staticmethod
def calculate_message_tokens(
    messages: list[dict],
    /,
    **kwargs,
) -> int: ...
```

**Parameters:**

- `messages` (list[dict]): OpenAI-format chat messages.
- `model` (str, keyword): Model name for encoding selection. Default: `"gpt-4o"`.
- `image_token_cost` (int, keyword): Token cost per image. Default: `500`.

**Returns:**

- `int`: Estimated token count.

**Example:**

```python
from lionpride.services.utilities import TokenCalculator

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]

tokens = TokenCalculator.calculate_message_tokens(messages, model="gpt-4o")
print(f"Estimated tokens: {tokens}")  # ~25 tokens
```

---

##### `calculate_embed_token()`

Calculate token count for embedding inputs.

```python
@staticmethod
def calculate_embed_token(
    inputs: list[str],
    /,
    **kwargs,
) -> int: ...
```

**Parameters:**

- `inputs` (list[str]): Text strings to embed. Must be non-empty.
- `model` (str, keyword): Model name for encoding selection. Default:
  `"text-embedding-3-small"`.

**Returns:**

- `int`: Total token count across all inputs.

**Raises:**

- `ValueError`: If `inputs` is empty.
- `TokenCalculationError`: If tokenization fails.

**Example:**

```python
texts = ["Hello world", "This is a test"]
tokens = TokenCalculator.calculate_embed_token(texts, model="text-embedding-3-small")
```

---

##### `tokenize()`

Low-level tokenization with optional token list and decoded string return.

```python
@staticmethod
def tokenize(
    s_: str | None = None,
    /,
    encoding_name: str | None = None,
    tokenizer: Callable | None = None,
    decoder: Callable | None = None,
    return_tokens: bool = False,
    return_decoded: bool = False,
) -> int | list[int] | tuple[int, str]: ...
```

**Parameters:**

- `s_` (str or None): Text to tokenize.
- `encoding_name` (str or None): Tiktoken encoding name (e.g., `"cl100k_base"`,
  `"o200k_base"`).
- `tokenizer` (Callable or None): Custom tokenizer function.
- `decoder` (Callable or None): Custom decoder function.
- `return_tokens` (bool): If `True`, return token list instead of count.
- `return_decoded` (bool): If `True` with `return_tokens`, return
  `(count, decoded_str)`.

**Returns:**

- `int`: Token count (default).
- `list[int]`: Token IDs if `return_tokens=True`.
- `tuple[int, str]`: `(count, decoded_string)` if both `return_tokens` and
  `return_decoded` are `True`.

**Raises:**

- `TokenCalculationError`: If tokenization fails.

---

### get_encoding_name()

> Resolve encoding name from model name with fallback.

```python
def get_encoding_name(value: str | None) -> str: ...
```

**Parameters:**

- `value` (str or None): Model name or encoding name.

**Returns:**

- `str`: Encoding name. Defaults to `"o200k_base"` if model/encoding not found.

**Example:**

```python
from lionpride.services.utilities.token_calculator import get_encoding_name

get_encoding_name("gpt-4o")           # "o200k_base"
get_encoding_name("gpt-3.5-turbo")    # "cl100k_base"
get_encoding_name("unknown-model")    # "o200k_base" (fallback)
get_encoding_name(None)               # "o200k_base" (default)
```

---

## Integration with iModel/Endpoint

### Rate Limiting in iModel

`iModel` supports two rate limiting approaches:

```python
from lionpride import iModel
from lionpride.services.utilities import RateLimitConfig, TokenBucket

# Approach 1: Auto-construction via limit_requests/limit_tokens
model = iModel(
    provider="openai",
    model="gpt-4o-mini",
    limit_requests=60,         # 60 requests per minute
    limit_tokens=100000,       # 100k tokens per minute
    capacity_refresh_time=60,  # Reset every 60 seconds
)

# Approach 2: Pre-constructed TokenBucket
config = RateLimitConfig(capacity=60, refill_rate=1.0)  # 60 capacity, 1/sec refill
bucket = TokenBucket(config)
model = iModel(provider="openai", model="gpt-4o-mini", rate_limiter=bucket)
```

### Resilience in Endpoint

`Endpoint` integrates circuit breaker and retry for HTTP resilience:

```python
from lionpride.services.types import Endpoint
from lionpride.services.utilities import CircuitBreaker, RetryConfig

# Configure resilience
cb = CircuitBreaker(
    failure_threshold=5,
    recovery_time=60.0,
    name="openai_chat",
)
retry = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
)

# Endpoint applies: retry(circuit_breaker(http_call))
endpoint = Endpoint(
    config={
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "endpoint": "chat/completions",
        "api_key": "OPENAI_API_KEY",  # Env var name
    },
    circuit_breaker=cb,
    retry_config=retry,
)

# Serialization/deserialization preserves resilience config
data = endpoint.model_dump()
restored = Endpoint.model_validate(data)
```

### Token Estimation in APICalling

`APICalling` uses `TokenCalculator` for rate limiting token estimation:

```python
# Internally, APICalling.required_tokens uses TokenCalculator:
calling = await endpoint.create_calling(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(calling.required_tokens)  # Token estimate for rate limiting
```

---

## Usage Patterns

### Basic Rate Limiting

```python
from lionpride.services.utilities import RateLimitConfig, TokenBucket

# API rate limit: 100 requests/minute with burst allowance
config = RateLimitConfig(
    capacity=100,       # Allow burst of 100
    refill_rate=1.67,   # ~100/min = 1.67/sec
)
bucket = TokenBucket(config)

async def rate_limited_call():
    # Non-blocking check
    if not await bucket.try_acquire():
        raise Exception("Rate limited")

    # Blocking with timeout
    if not await bucket.acquire(timeout=10.0):
        raise Exception("Rate limit timeout")
```

### Dual Bucket Rate Limiting

```python
from lionpride.services.utilities import RateLimitConfig, TokenBucket

# OpenAI rate limits: 60 RPM + 100k TPM
request_bucket = TokenBucket(RateLimitConfig(capacity=60, refill_rate=1.0))
token_bucket = TokenBucket(RateLimitConfig(capacity=100000, refill_rate=1667))

async def rate_limited_api_call(tokens_needed: int):
    # Atomic dual acquire with rollback
    if not await request_bucket.try_acquire():
        return False

    if not await token_bucket.try_acquire(tokens=tokens_needed):
        await request_bucket.release()  # Rollback request acquire
        return False

    return True
```

### Circuit Breaker with Excluded Exceptions

```python
from lionpride.services.utilities import CircuitBreaker

# Don't count validation errors as circuit failures
cb = CircuitBreaker(
    failure_threshold=5,
    recovery_time=30.0,
    excluded_exceptions={ValueError, TypeError},
    name="api_circuit",
)

# Only network/server errors trip the circuit
try:
    result = await cb.execute(api_call, payload)
except ValueError:
    # Doesn't count toward failure_threshold
    pass
```

### Retry with Custom Exception Types

```python
import httpx
from lionpride.services.utilities import retry_with_backoff

# Retry on HTTP and connection errors
result = await retry_with_backoff(
    make_request,
    max_retries=5,
    retry_on=(httpx.HTTPStatusError, httpx.ConnectError),
)
```

---

## Common Pitfalls

- **Token bucket overflow**: `initial_tokens` cannot exceed `capacity`. Use default
  (`None`) for full bucket start.

- **Circuit breaker excluded exceptions**: Excluding `Exception` base class means
  circuit never opens. Use specific exception types.

- **Retry thundering herd**: Always enable `jitter=True` (default) to prevent
  synchronized retry storms.

- **Token estimation accuracy**: `TokenCalculator` provides estimates. Actual token
  usage may vary by ~5% due to message formatting overhead.

- **Monotonic time**: All timing uses monotonic clock (`time.monotonic()`) for immunity
  to system clock changes.

---

## Design Rationale

**Token Bucket vs Fixed Window**: Token bucket allows bursts while maintaining average
rate, better matching API provider rate limit semantics.

**Circuit Breaker Integration**: Circuit breaker wraps the inner call, retry wraps the
circuit breaker. This ensures each retry attempt counts against circuit breaker metrics.

**Atomic Dual Acquire**: Dual bucket acquire (request + tokens) uses rollback pattern to
prevent capacity leakage on partial failure.

**Tiktoken Integration**: Uses OpenAI's tiktoken for accurate token estimation, with
automatic encoding detection and fallback.

---

## See Also

- [iModel](./imodel.md): Unified service interface using these utilities
- [ServiceRegistry](./registry.md): Service management
- [Concurrency Primitives](../libs/concurrency/primitives.md): Lock, sleep, current_time
  used by utilities
- [Errors](../errors.md): LionConnectionError base class
