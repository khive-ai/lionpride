# iModel

> Unified service interface wrapping ServiceBackend with rate limiting and hooks.

## Overview

`iModel` wraps a `ServiceBackend` (Endpoint or Tool) providing multi-provider support (OpenAI, Anthropic, Gemini, Claude Code), rate limiting, and hook lifecycle callbacks. Inherits from `Element` for UUID-based identity.

## Class Signature

```python
@implements(Invocable)
class iModel(Element):
    """Unified service interface wrapping ServiceBackend with rate limiting and hooks."""

    def __init__(
        self,
        backend: ServiceBackend | None = None,
        provider: str | None = None,
        endpoint: str = "chat/completions",
        rate_limiter: TokenBucket | None = None,
        executor: Executor | None = None,
        hook_registry: HookRegistry | None = None,
        # Executor auto-construction params:
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        limit_requests: int | None = None,
        limit_tokens: int | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `ServiceBackend or None` | `None` | Pre-constructed backend (Endpoint, Tool). Mutually exclusive with `provider`. |
| `provider` | `str or None` | `None` | Provider string for auto-matching via `match_endpoint()`. One of: `"openai"`, `"anthropic"`, `"gemini"`, `"claude_code"`. |
| `endpoint` | `str` | `"chat/completions"` | API endpoint path (used with `provider`). |
| `rate_limiter` | `TokenBucket or None` | `None` | Simple blocking rate limiter. |
| `executor` | `Executor or None` | `None` | Event-driven processor with rate limiting. Auto-constructed if `limit_requests`/`limit_tokens` provided. |
| `hook_registry` | `HookRegistry or None` | `None` | Registry for lifecycle hooks. |
| `queue_capacity` | `int` | `100` | Max events in executor queue. |
| `capacity_refresh_time` | `float` | `60` | Seconds for rate limit bucket refill. |
| `limit_requests` | `int or None` | `None` | Request rate limit (triggers executor auto-construction). |
| `limit_tokens` | `int or None` | `None` | Token rate limit (triggers executor auto-construction). |
| `**kwargs` | `Any` | - | Passed to `match_endpoint()` (e.g., `model`, `api_key`). |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Unique identifier (inherited from Element, auto-generated, frozen) |
| `created_at` | `datetime` | UTC timestamp (inherited from Element, auto-generated, frozen) |
| `metadata` | `dict[str, Any]` | Arbitrary metadata (inherited from Element) |
| `backend` | `ServiceBackend` | The wrapped service backend (Endpoint or Tool) |
| `rate_limiter` | `TokenBucket or None` | Simple blocking rate limiter |
| `executor` | `Executor or None` | Event-driven processor for rate-limited execution |
| `hook_registry` | `HookRegistry or None` | Registry for invocation lifecycle hooks |
| `provider_metadata` | `dict[str, Any]` | Provider-specific metadata (e.g., Claude Code `session_id`) |
| `name` | `str` | Service name from backend (property) |
| `version` | `str` | Service version from backend (property) |
| `tags` | `set[str]` | Service tags from backend (property) |

## Methods

### Core Operations

#### `create_calling()`

Create a `Calling` instance for service invocation.

```python
async def create_calling(
    self,
    timeout: float | None = None,
    streaming: bool = False,
    **arguments: Any,  # + hook params (exit_hook, timeout, params for each phase)
) -> Calling: ...
```

Returns `APICalling` for Endpoint or `ToolCalling` for Tool backends.

---

#### `invoke()`

Invoke a calling with optional event-driven processing.

```python
async def invoke(
    self,
    calling: Calling | None = None,
    poll_timeout: float | None = None,  # Default 10s, increase for large models
    poll_interval: float | None = None,  # Default 0.1s
    **arguments: Any,  # Ignored if calling provided
) -> Calling: ...
```

Thread-safe. Raises `TimeoutError` on poll timeout, `RuntimeError` after 3 permission denials.

```python
calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(calling.response.data["choices"][0]["message"]["content"])
```

---

### Serialization

#### `to_dict()`

Serialize to dictionary. Identity fields excluded; Tool backends serialize as `None` (callables not serializable).

```python
data = model.to_dict()
restored = iModel.from_dict(data)  # Fresh id/created_at
```

## Protocol Implementations

Implements: **Observable**, **Serializable**, **Deserializable**, **Hashable** (from Element), **Invocable**.

## Usage Patterns

### Basic Usage

```python
from lionpride import iModel

model = iModel(provider="openai", model="gpt-4o-mini")
calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(calling.response.data["choices"][0]["message"]["content"])
```

### Rate Limiting

```python
model = iModel(
    provider="openai",
    model="gpt-4o-mini",
    limit_requests=50,        # 50 requests per refresh
    limit_tokens=100000,      # 100k tokens per refresh
    capacity_refresh_time=60, # Refill every 60s
)
# All invocations automatically respect rate limits
```

### Multi-Provider with Registry

```python
from lionpride import iModel, ServiceRegistry

registry = ServiceRegistry()
registry.register(iModel(provider="openai", model="gpt-4o-mini", name="gpt4"))
registry.register(iModel(provider="anthropic", model="claude-3-5-sonnet", name="claude"))

model = registry.get("claude")
```

## Common Pitfalls

- **Missing backend or provider**: Either `backend` or `provider` must be provided (not both, not neither).

- **Executor polling timeout**: Long LLM calls timeout with default 10s. Use `poll_timeout=120.0` for large models.

- **Tool serialization**: Tool-backed iModel cannot be deserialized. Use Endpoint backends for persistence.

## Design Rationale

**Unified Interface**: Same `invoke()` pattern for Endpoint (HTTP) and Tool (local function) backends.

**Event-Driven Rate Limiting**: Executor pattern queues events and processes them according to token bucket policies without blocking.

**Hook Lifecycle**: Extension points at `PreEventCreate`, `PreInvocation`, `PostInvocation` phases.

## See Also

- [`ServiceBackend`](./backend.md), [`Endpoint`](./endpoint.md), [`Tool`](./tool.md): Backend classes
- [`ServiceRegistry`](./registry.md): Service management
- [`HookRegistry`](./hook.md): Hook lifecycle

## Example: Batch Processing

```python
import asyncio
from lionpride import iModel

async def process_batch(prompts: list[str]) -> list[str]:
    model = iModel(
        provider="openai", model="gpt-4o-mini",
        limit_requests=50, limit_tokens=100000, capacity_refresh_time=60,
    )
    results = []
    for prompt in prompts:
        calling = await model.invoke(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        results.append(calling.response.data["choices"][0]["message"]["content"])
    return results

prompts = [f"Summarize topic {i}" for i in range(100)]
results = asyncio.run(process_batch(prompts))
```
