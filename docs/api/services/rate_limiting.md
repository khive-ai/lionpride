# Rate Limiting

> Token bucket rate limiting for API consumption control

## Overview

The rate limiting module provides token bucket-based rate limiting for controlling API request rates and token consumption. It supports both simple blocking rate limiting (via `iModel.rate_limiter`) and event-driven rate limiting (via `iModel.executor` with `RateLimitedProcessor`).

**Core Components:**

- **`TokenBucket`**: Thread-safe token bucket implementation with time-based refill
- **`RateLimitConfig`**: Configuration for token bucket capacity and refill rate
- **Integration**: Used by `iModel` for request/token rate limiting

**Key Features:**

- **Token Bucket Algorithm**: Standard algorithm with configurable capacity and refill rate
- **Thread-Safe**: Async locks ensure correct concurrent access
- **Timeout Support**: Configurable timeout for token acquisition
- **Dual-Bucket**: Separate buckets for request count and token count (via executor)
- **Serialization**: Configuration serializable (runtime state not preserved)

**When to Use:**

- Rate-limiting API calls to comply with provider limits
- Preventing quota exhaustion in multi-tenant systems
- Controlling token consumption for cost management
- Implementing fair-share scheduling across concurrent tasks

## Module Exports

```python
from lionpride.services.utilities.rate_limiter import (
    RateLimitConfig,
    TokenBucket,
)
```

## RateLimitConfig

Frozen configuration dataclass for token bucket parameters.

### Class Signature

```python
@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """Token bucket rate limiting configuration."""

    capacity: int
    refill_rate: float
    initial_tokens: int | None = None
```

### Parameters

- `capacity` (int): Maximum token capacity (must be > 0)
- `refill_rate` (float): Tokens added per second (must be > 0)
- `initial_tokens` (int | None): Initial token count (default: `capacity`, must be ≤ capacity)

### Validation

Post-initialization validation ensures:

- `capacity > 0`
- `refill_rate > 0`
- `initial_tokens` defaults to `capacity` if None
- `0 ≤ initial_tokens ≤ capacity`

### Examples

```python
from lionpride.services.utilities.rate_limiter import RateLimitConfig

# 50 requests per minute (refill_rate = 50/60 = 0.833 tokens/sec)
config = RateLimitConfig(
    capacity=50,
    refill_rate=50 / 60,
)

# 100k tokens per minute with half capacity start
config = RateLimitConfig(
    capacity=100000,
    refill_rate=100000 / 60,
    initial_tokens=50000,
)

# Validation errors
config = RateLimitConfig(capacity=0, refill_rate=1.0)  # ValueError: capacity must be > 0
config = RateLimitConfig(capacity=10, refill_rate=-1.0)  # ValueError: refill_rate must be > 0
config = RateLimitConfig(capacity=10, refill_rate=1.0, initial_tokens=20)  # ValueError: exceeds capacity
```

## TokenBucket

Thread-safe token bucket rate limiter with time-based refill.

### Class Signature

```python
class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig): ...

    async def acquire(
        self,
        tokens: int = 1,
        *,
        timeout: float | None = None
    ) -> bool: ...

    async def try_acquire(self, tokens: int = 1) -> bool: ...

    async def reset(self) -> None: ...

    async def release(self, tokens: int = 1) -> None: ...

    def to_dict(self) -> dict[str, float]: ...
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `capacity` | `int` | Maximum token capacity |
| `refill_rate` | `float` | Tokens added per second |
| `tokens` | `float` | Current available tokens (mutable) |
| `last_refill` | `float` | Last refill timestamp (monotonic time) |

### Methods

#### `__init__()`

Initialize token bucket with configuration.

**Signature:**

```python
def __init__(self, config: RateLimitConfig): ...
```

**Parameters:**

- `config` (RateLimitConfig): Token bucket configuration

**Example:**

```python
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket

config = RateLimitConfig(capacity=50, refill_rate=50/60)
bucket = TokenBucket(config)
```

#### `acquire()`

Acquire N tokens, waiting if necessary.

**Signature:**

```python
async def acquire(
    self,
    tokens: int = 1,
    *,
    timeout: float | None = None
) -> bool: ...
```

**Parameters:**

- `tokens` (int): Number of tokens to acquire (default: 1, must be > 0 and ≤ capacity)
- `timeout` (float | None): Max wait time in seconds (None = wait forever)

**Returns**: `bool` - True if acquired, False if timeout

**Raises**:

- `ValueError`: If `tokens ≤ 0` or `tokens > capacity`

**Algorithm**:

1. Refill tokens based on elapsed time
2. If sufficient tokens: acquire immediately
3. Otherwise: calculate wait time (`deficit / refill_rate`)
4. Sleep and retry (checks timeout before sleeping)

**Examples:**

```python
# Acquire 1 token (wait indefinitely)
acquired = await bucket.acquire()

# Acquire 5 tokens with 10s timeout
acquired = await bucket.acquire(tokens=5, timeout=10.0)
if not acquired:
    print("Timeout: insufficient tokens")

# Acquire more than capacity (error)
acquired = await bucket.acquire(tokens=100)  # ValueError: exceeds capacity
```

#### `try_acquire()`

Try to acquire tokens without waiting.

**Signature:**

```python
async def try_acquire(self, tokens: int = 1) -> bool: ...
```

**Parameters:**

- `tokens` (int): Number of tokens to acquire (must be > 0)

**Returns**: `bool` - True if acquired immediately, False if insufficient tokens

**Raises**:

- `ValueError`: If `tokens ≤ 0`

**Example:**

```python
# Try acquire without blocking
if await bucket.try_acquire(tokens=5):
    print("Acquired 5 tokens")
else:
    print("Insufficient tokens")
```

#### `reset()`

Reset bucket to full capacity (for interval-based replenishment).

**Signature:**

```python
async def reset(self) -> None: ...
```

**Purpose**: Used by `RateLimitedProcessor` background task to reset capacity at regular intervals (e.g., every 60 seconds).

**Thread-Safe**: Acquires lock before resetting both `tokens` and `last_refill` atomically.

**Example:**

```python
# Reset to full capacity
await bucket.reset()
print(bucket.tokens)  # capacity (e.g., 50.0)
```

#### `release()`

Release tokens back to bucket (for atomic dual acquire rollback).

**Signature:**

```python
async def release(self, tokens: int = 1) -> None: ...
```

**Parameters:**

- `tokens` (int): Number of tokens to release back (must be > 0)

**Raises**:

- `ValueError`: If `tokens ≤ 0`

**Purpose**: Used when request bucket acquires but token bucket fails. Prevents capacity leakage by releasing the request token.

**Thread-Safe**: Acquires lock before modification.

**Example:**

```python
# Acquire request token
acquired_req = await request_bucket.acquire()

# Try acquire token bucket
acquired_tok = await token_bucket.acquire(tokens=500)

if not acquired_tok:
    # Rollback: release request token
    await request_bucket.release()
    raise TimeoutError("Token bucket timeout")
```

#### `to_dict()`

Serialize configuration (excludes runtime state).

**Signature:**

```python
def to_dict(self) -> dict[str, float]: ...
```

**Returns**: `dict[str, float]` - Configuration dict with `capacity` and `refill_rate`

**Note**: Runtime state (`tokens`, `last_refill`) NOT serialized. Deserialization creates fresh bucket with full capacity.

**Example:**

```python
bucket = TokenBucket(config)
# Use bucket (consume 20 tokens)
await bucket.acquire(tokens=20)

# Serialize
config_dict = bucket.to_dict()
# {"capacity": 50, "refill_rate": 0.833}

# Deserialize (fresh capacity)
new_bucket = TokenBucket(RateLimitConfig(**config_dict))
print(new_bucket.tokens)  # 50.0 (full capacity, not 30.0)
```

## Usage Patterns

### Basic Usage: Simple Blocking Rate Limiting

```python
from lionpride.services import iModel
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket

# Create token bucket (50 requests per minute)
config = RateLimitConfig(capacity=50, refill_rate=50/60)
bucket = TokenBucket(config)

# Attach to iModel (simple blocking)
model = iModel(
    provider="openai",
    api_key="OPENAI_API_KEY",
    rate_limiter=bucket,
)

# Calls are rate-limited
calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Advanced Usage: Dual-Bucket Rate Limiting (Request + Token)

```python
from lionpride.services import iModel

# Auto-construct executor with dual buckets (lionagi v0 pattern)
model = iModel(
    provider="openai",
    api_key="OPENAI_API_KEY",
    limit_requests=50,      # 50 requests per minute
    limit_tokens=100000,    # 100k tokens per minute
    capacity_refresh_time=60.0,
)

# Executor automatically manages both buckets
# Request bucket: 1 token per request
# Token bucket: estimated tokens per request (from payload)
calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Advanced Usage: Manual Dual-Bucket with Rollback

```python
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket

# Create two buckets
request_bucket = TokenBucket(RateLimitConfig(capacity=50, refill_rate=50/60))
token_bucket = TokenBucket(RateLimitConfig(capacity=100000, refill_rate=100000/60))

# Acquire atomically (rollback on partial failure)
async def acquire_dual(tokens_needed: int) -> bool:
    # Acquire request token
    acquired_req = await request_bucket.acquire(timeout=10.0)
    if not acquired_req:
        return False

    # Try acquire token bucket
    acquired_tok = await token_bucket.acquire(tokens=tokens_needed, timeout=10.0)

    if not acquired_tok:
        # Rollback: release request token
        await request_bucket.release()
        return False

    return True

# Use in code
if await acquire_dual(tokens_needed=500):
    # Make API call
    response = await api_call()
else:
    print("Rate limit exceeded")
```

### Advanced Usage: Periodic Reset for Interval-Based Limits

```python
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket
from lionpride.libs.concurrency import sleep
import asyncio

# Create bucket for hourly limit (1000 requests per hour)
bucket = TokenBucket(RateLimitConfig(capacity=1000, refill_rate=0))  # No continuous refill

# Background task to reset every hour
async def reset_hourly():
    while True:
        await sleep(3600)  # 1 hour
        await bucket.reset()

# Start background task
asyncio.create_task(reset_hourly())

# Use bucket
acquired = await bucket.acquire()
```

## Integration with iModel

### Simple Blocking (rate_limiter)

```python
from lionpride.services import iModel
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket

bucket = TokenBucket(RateLimitConfig(capacity=50, refill_rate=50/60))

model = iModel(
    provider="anthropic",
    api_key="ANTHROPIC_API_KEY",
    rate_limiter=bucket,
)

# Direct invocation path (no executor)
# Acquires 1 token before each call (timeout: 30s)
calling = await model.invoke(...)
```

**Flow**:

1. `iModel.invoke()` checks if `rate_limiter` present
2. `await rate_limiter.acquire(timeout=30.0)`
3. If timeout: raise `TimeoutError`
4. Otherwise: proceed with call

### Event-Driven (executor)

```python
from lionpride.services import iModel

# Auto-construct executor with rate limiting
model = iModel(
    provider="anthropic",
    api_key="ANTHROPIC_API_KEY",
    limit_requests=50,
    limit_tokens=100000,
    capacity_refresh_time=60.0,
)

# Executor path (event-driven processing)
# RateLimitedProcessor manages dual buckets
calling = await model.invoke(...)
```

**Flow**:

1. `iModel.invoke()` creates `Calling` event
2. Enqueues event to executor
3. `RateLimitedProcessor.request_permission()`:
   - Acquire request bucket (1 token)
   - Acquire token bucket (`event.required_tokens`)
   - If both succeed: grant permission
   - If either fails: deny (event retried later)
4. After 3 denials: abort event
5. Background task resets buckets every `capacity_refresh_time`

## Token Estimation

`APICalling.required_tokens` estimates token usage for rate limiting:

```python
def _estimate_message_tokens(self, messages: list[dict]) -> int:
    """Estimate token usage (~4 chars per token)."""
    total_chars = sum(
        len(str(msg.get("content", ""))) + len(msg.get("role", ""))
        for msg in messages
    )
    return int(total_chars / 4) + 10  # +10 for message overhead
```

**Accuracy**: Rough approximation. For production, use `tiktoken` or provider-specific libraries.

**Edge Cases** (F2):

- Empty messages: Returns `None` (can't estimate)
- Missing fields: Returns `None` (unknown structure)
- Unknown payload format: Returns `None` (fallback)

When `None` returned, token rate limiting skipped (only request limit enforced).

## Common Pitfalls

### Issue: Timeout too short for slow refill

**Problem**: Timeout expires before sufficient tokens refill.

```python
# ❌ WRONG: 10s timeout, but need 60s to refill 50 tokens
bucket = TokenBucket(RateLimitConfig(capacity=50, refill_rate=50/60))
# Consume all 50 tokens
for _ in range(50):
    await bucket.acquire()
# Next acquire waits 60s, but timeout is 10s
acquired = await bucket.acquire(timeout=10.0)  # Returns False (timeout)
```

**Solution**: Increase timeout or adjust refill rate.

```python
# ✅ CORRECT: Increase timeout
acquired = await bucket.acquire(timeout=120.0)

# ✅ CORRECT: Faster refill rate (10 tokens/sec)
bucket = TokenBucket(RateLimitConfig(capacity=50, refill_rate=10.0))
```

### Issue: Serialization doesn't preserve state

**Problem**: Assuming depleted capacity is serialized.

```python
# ❌ WRONG ASSUMPTION
bucket = TokenBucket(RateLimitConfig(capacity=50, refill_rate=50/60))
# Consume 40 tokens (10 remaining)
for _ in range(40):
    await bucket.acquire()

# Serialize
config_dict = bucket.to_dict()

# Deserialize (expects 10 tokens)
new_bucket = TokenBucket(RateLimitConfig(**config_dict))
print(new_bucket.tokens)  # 50.0 (full capacity, not 10.0)
```

**Solution**: Understand serialization resets capacity. Implement custom state persistence if needed.

```python
# ✅ CORRECT UNDERSTANDING
# Serialization stores configuration, not state
# Deserialization creates fresh bucket with full capacity
# This is intentional to prevent capacity leak across restarts
```

### Issue: Acquiring more tokens than capacity

**Problem**: Request exceeds bucket capacity.

```python
# ❌ WRONG: Request 100 tokens, capacity is 50
bucket = TokenBucket(RateLimitConfig(capacity=50, refill_rate=50/60))
acquired = await bucket.acquire(tokens=100)  # ValueError: exceeds capacity
```

**Solution**: Request ≤ capacity or increase capacity.

```python
# ✅ CORRECT: Request within capacity
acquired = await bucket.acquire(tokens=50)

# ✅ CORRECT: Increase capacity
bucket = TokenBucket(RateLimitConfig(capacity=150, refill_rate=150/60))
acquired = await bucket.acquire(tokens=100)
```

### Issue: Dual-bucket without rollback

**Problem**: Partial acquisition without rollback leaks capacity.

```python
# ❌ WRONG: No rollback on partial failure
acquired_req = await request_bucket.acquire()  # Success
acquired_tok = await token_bucket.acquire(tokens=500, timeout=5.0)  # Timeout
# request_bucket token leaked (not released)
```

**Solution**: Always release on partial failure.

```python
# ✅ CORRECT: Rollback on partial failure
acquired_req = await request_bucket.acquire()
acquired_tok = await token_bucket.acquire(tokens=500, timeout=5.0)

if not acquired_tok:
    await request_bucket.release()  # Rollback
    raise TimeoutError("Token bucket timeout")
```

## Design Rationale

### Why token bucket algorithm?

Token bucket is ideal for API rate limiting because:

1. **Burst Handling**: Allows bursts up to capacity (not strict per-second)
2. **Fairness**: Refill rate ensures long-term average rate
3. **Simplicity**: Easy to reason about and implement correctly
4. **Industry Standard**: Used by AWS, GCP, Anthropic, OpenAI

### Why monotonic time for refill?

`libs.concurrency.current_time()` uses `time.monotonic()` (via event loop):

- **Immune to system clock changes**: Manual adjustments don't affect refill
- **Monotonically increasing**: Never goes backward (no negative elapsed time)
- **High resolution**: Sufficient precision for sub-second refill

### Why separate reset() and release()?

Different use cases:

- **`reset()`**: Interval-based replenishment (e.g., hourly quota reset)
- **`release()`**: Atomic rollback on partial dual-bucket failure

Separating ensures clear semantics and prevents misuse.

### Why not serialize runtime state?

Serializing depleted capacity causes issues:

1. **Capacity Leak**: Restart with depleted bucket unfair to new tasks
2. **Time Dependency**: State validity depends on deserialization timing
3. **Simplicity**: Configuration-only serialization easier to reason about

For persistent state tracking, implement custom solution (e.g., database-backed bucket).

## See Also

- **Integration**:
  - [`iModel`](imodel.md): Uses TokenBucket for rate limiting
  - [`Executor`](../base/executor.md): Event-driven processing with rate limiting
- **Resilience**:
  - [`CircuitBreaker`](resilience.md#circuitbreaker): Fail-fast pattern
  - [`RetryConfig`](resilience.md#retryconfig): Retry with backoff
- **Concurrency**:
  - [libs.concurrency](../libs/concurrency.md): Lock, sleep, current_time

## Examples

### Example 1: Hourly Quota with Reset

```python
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket
from lionpride.libs.concurrency import sleep
import asyncio

# 1000 requests per hour (interval-based, not continuous)
bucket = TokenBucket(RateLimitConfig(capacity=1000, refill_rate=0))

async def reset_hourly():
    while True:
        await sleep(3600)
        await bucket.reset()
        print("Quota reset: 1000 requests available")

asyncio.create_task(reset_hourly())

# Use bucket
for i in range(1500):
    acquired = await bucket.acquire()
    if not acquired:
        print(f"Quota exhausted at request {i}")
        break
```

### Example 2: Dual-Bucket with Metrics

```python
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket

request_bucket = TokenBucket(RateLimitConfig(capacity=50, refill_rate=50/60))
token_bucket = TokenBucket(RateLimitConfig(capacity=100000, refill_rate=100000/60))

# Metrics
total_requests = 0
total_tokens = 0
timeouts = 0

async def rate_limited_call(tokens_needed: int):
    global total_requests, total_tokens, timeouts

    # Acquire request token
    acquired_req = await request_bucket.acquire(timeout=30.0)
    if not acquired_req:
        timeouts += 1
        return False

    # Acquire token bucket
    acquired_tok = await token_bucket.acquire(tokens=tokens_needed, timeout=30.0)
    if not acquired_tok:
        await request_bucket.release()  # Rollback
        timeouts += 1
        return False

    # Track metrics
    total_requests += 1
    total_tokens += tokens_needed

    return True

# Use
success = await rate_limited_call(tokens_needed=500)
print(f"Requests: {total_requests}, Tokens: {total_tokens}, Timeouts: {timeouts}")
```

### Example 3: Dynamic Rate Adjustment

```python
from lionpride.services.utilities.rate_limiter import RateLimitConfig, TokenBucket

# Start with conservative rate
bucket = TokenBucket(RateLimitConfig(capacity=10, refill_rate=10/60))

# Monitor error rate
error_rate = 0.0

async def adaptive_acquire():
    global error_rate, bucket

    acquired = await bucket.try_acquire()
    if not acquired:
        # Backpressure: reduce rate
        new_rate = bucket.refill_rate * 0.8
        bucket = TokenBucket(RateLimitConfig(capacity=bucket.capacity, refill_rate=new_rate))
        print(f"Rate reduced to {new_rate:.3f} tokens/sec")
        return False

    # Make call
    try:
        response = await api_call()
        error_rate = error_rate * 0.9  # Decay error rate
    except Exception:
        error_rate = error_rate * 0.9 + 0.1  # Increase error rate

    # If low error rate, increase capacity
    if error_rate < 0.05 and bucket.refill_rate < 1.0:
        new_rate = bucket.refill_rate * 1.1
        bucket = TokenBucket(RateLimitConfig(capacity=bucket.capacity, refill_rate=new_rate))
        print(f"Rate increased to {new_rate:.3f} tokens/sec")

    return True
```
