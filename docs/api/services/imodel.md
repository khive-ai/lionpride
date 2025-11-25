# iModel

> Unified service interface wrapping ServiceBackend with rate limiting, hooks, and event-driven processing

## Overview

`iModel` is the primary interface for interacting with language models, APIs, and tools in lionpride. It wraps a `ServiceBackend` (Endpoint or Tool) with optional rate limiting, circuit breaking, retry logic, and lifecycle hooks, providing a unified interface regardless of the underlying service provider.

**Core Design:**

- **Backend Abstraction**: Delegates execution to `ServiceBackend` (Endpoint, Tool, etc.)
- **Auto-Matching**: `iModel(provider="anthropic")` auto-configures appropriate endpoint
- **Rate Limiting**: Optional `TokenBucket` for simple blocking or `Executor` for event-driven processing
- **Lifecycle Hooks**: Pre/post invocation hooks for logging, metrics, transformations
- **Serialization**: Full serialization support (Endpoint backends only, Tools contain callables)
- **Streaming**: Built-in streaming with channel abstraction for fan-out

**When to Use:**

- Building LLM-powered applications with multiple providers
- Rate-limited API consumption with automatic backoff
- Tool-based agent systems with schema validation
- Production deployments requiring resilience patterns
- Multi-model orchestration with service registry

## Class Signature

```python
@implements(Invocable)
class iModel(Element):
    """Unified service interface wrapping ServiceBackend with rate limiting and hooks."""

    backend: ServiceBackend | None = Field(...)
    rate_limiter: TokenBucket | None = Field(None)
    executor: Executor | None = Field(None)
    hook_registry: HookRegistry | None = Field(None)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        backend: ServiceBackend | None = None,
        provider: str | None = None,
        endpoint: str = "chat/completions",
        rate_limiter: TokenBucket | None = None,
        executor: Executor | None = None,
        hook_registry: HookRegistry | None = None,
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        limit_requests: int | None = None,
        limit_tokens: int | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

**Backend Configuration (choose one):**

- `backend` (ServiceBackend | None): Explicit backend instance (Endpoint, Tool, etc.). If None, use provider matching.
- `provider` (str | None): Provider name for auto-matching (`"anthropic"`, `"openai"`, `"groq"`, etc.). Requires `backend=None`.
- `endpoint` (str): Endpoint path for auto-matching (default: `"chat/completions"`). Only used with `provider`.

**Rate Limiting (optional):**

- `rate_limiter` (TokenBucket | None): Simple blocking rate limiter (no executor).
- `executor` (Executor | None): Event-driven processor with advanced rate limiting. If None and `limit_requests`/`limit_tokens` provided, auto-constructed.
- `limit_requests` (int | None): Max requests per `capacity_refresh_time` (triggers executor auto-construction).
- `limit_tokens` (int | None): Max tokens per `capacity_refresh_time` (triggers executor auto-construction).
- `queue_capacity` (int): Max events per batch for auto-constructed executor (default: 100).
- `capacity_refresh_time` (float): Seconds before capacity reset for auto-constructed executor (default: 60).

**Lifecycle Hooks (optional):**

- `hook_registry` (HookRegistry | None): Registry for pre/post invocation hooks.

**Additional:**

- `**kwargs`: Passed to `match_endpoint()` or `Element.__init__()`.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `backend` | `ServiceBackend` | Service backend (Endpoint, Tool, etc.) |
| `rate_limiter` | `TokenBucket \| None` | Simple blocking rate limiter |
| `executor` | `Executor \| None` | Event-driven processor for rate limiting |
| `hook_registry` | `HookRegistry \| None` | Lifecycle hook registry |
| `provider_metadata` | `dict[str, Any]` | Provider-specific metadata (e.g., Claude Code session_id) |
| `name` | `str` | Service name from backend (property) |
| `version` | `str` | Service version from backend (property) |
| `tags` | `set[str]` | Service tags from backend (property) |

## Methods

### Core Operations

#### `__init__()`

Initialize iModel with ServiceBackend or auto-match from provider.

**Signature:**

```python
def __init__(
    self,
    backend: ServiceBackend | None = None,
    provider: str | None = None,
    endpoint: str = "chat/completions",
    rate_limiter: TokenBucket | None = None,
    executor: Executor | None = None,
    hook_registry: HookRegistry | None = None,
    queue_capacity: int = 100,
    capacity_refresh_time: float = 60,
    limit_requests: int | None = None,
    limit_tokens: int | None = None,
    **kwargs: Any,
) -> None: ...
```

**Examples:**

```python
# Manual backend
from lionpride.services import iModel
from lionpride.services.providers.anthropic_messages import AnthropicMessagesEndpoint

backend = AnthropicMessagesEndpoint(api_key="ANTHROPIC_API_KEY")
model = iModel(backend=backend)

# Auto-match from provider
model = iModel(provider="anthropic", api_key="ANTHROPIC_API_KEY")
model = iModel(provider="openai", endpoint="chat/completions")

# With auto-constructed executor (lionagi v0 pattern)
model = iModel(
    provider="anthropic",
    api_key="ANTHROPIC_API_KEY",
    limit_requests=50,      # 50 requests per minute
    limit_tokens=100000,    # 100k tokens per minute
    capacity_refresh_time=60.0,
)
```

#### `create_calling()`

Create `Calling` instance via backend without invoking.

**Signature:**

```python
async def create_calling(
    self,
    timeout: float | None = None,
    streaming: bool = False,
    create_event_exit_hook: bool | None = None,
    create_event_hook_timeout: float = 10.0,
    create_event_hook_params: dict | None = None,
    pre_invoke_exit_hook: bool | None = None,
    pre_invoke_hook_timeout: float = 30.0,
    pre_invoke_hook_params: dict | None = None,
    post_invoke_exit_hook: bool | None = None,
    post_invoke_hook_timeout: float = 30.0,
    post_invoke_hook_params: dict | None = None,
    **arguments: Any,
) -> Calling: ...
```

**Parameters:**

- `timeout` (float | None): Event timeout in seconds (enforced in `Event.invoke()`)
- `streaming` (bool): Whether this is a streaming request (default: False)
- `create_event_exit_hook` (bool | None): Whether pre-event-create hook should trigger exit on failure
- `create_event_hook_timeout` (float): Timeout for pre-event-create hook (default: 10.0)
- `create_event_hook_params` (dict | None): Parameters for pre-event-create hook
- `pre_invoke_exit_hook` (bool | None): Whether pre-invoke hook should trigger exit on failure
- `pre_invoke_hook_timeout` (float): Timeout for pre-invoke hook (default: 30.0)
- `pre_invoke_hook_params` (dict | None): Parameters for pre-invoke hook
- `post_invoke_exit_hook` (bool | None): Whether post-invoke hook should trigger exit on failure
- `post_invoke_hook_timeout` (float): Timeout for post-invoke hook (default: 30.0)
- `post_invoke_hook_params` (dict | None): Parameters for post-invoke hook
- `**arguments`: Request arguments to pass to backend

**Returns**: `Calling` instance (not yet invoked)

**Examples:**

```python
# Create calling for later invocation
calling = await model.create_calling(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}],
    timeout=120.0,
)

# Invoke later (useful for custom timeout configuration)
calling.timeout = 180.0  # Override timeout
calling = await model.invoke(calling=calling)
```

#### `invoke()`

Invoke calling with optional event-driven processing.

**Signature:**

```python
async def invoke(
    self,
    calling: Calling | None = None,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
    **arguments: Any,
) -> Calling: ...
```

**Parameters:**

- `calling` (Calling | None): Pre-created Calling instance. If provided, `**arguments` are IGNORED.
- `poll_timeout` (float | None): Max seconds to wait for executor completion (default: 10s). For long-running LLM calls, increase this (e.g., 120s).
- `poll_interval` (float | None): Seconds between status checks (default: 0.1s).
- `**arguments`: Request arguments passed to `create_calling()`. IGNORED if `calling` provided.

**Returns**: `Calling` instance with `execution.response` populated

**Raises**:

- `TimeoutError`: If rate limit acquisition or polling times out
- `RuntimeError`: If event aborted after 3 permission denials (executor path)

**Routing Logic**:

- If `executor` configured: Event-driven processing with rate limiting (lionagi v0 pattern)
- Otherwise: Direct invocation with optional simple rate limiting

**Examples:**

```python
# Standard usage - create and invoke in one call
calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(calling.response.data)  # Extracted text

# Pre-created calling with custom timeout
calling = await model.create_calling(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
calling.timeout = 120.0  # 2 minute timeout
calling = await model.invoke(calling=calling)

# Long-running LLM call with executor
calling = await model.invoke(
    poll_timeout=180.0,  # 3 minutes
    model="claude-opus-4",
    messages=[{"role": "user", "content": "Write an essay"}]
)
```

**Performance Note (BLIND-4)**:

Executor polling adds latency overhead:

- Fast backends (<100ms): 100-200% overhead (not recommended)
- Slow backends (>1s): <10% overhead (acceptable)

Use executor for rate-limited batch processing, not single fast calls.

#### `invoke_stream()`

Stream invoke - returns async iterator of chunks.

**Signature:**

```python
async def invoke_stream(
    self,
    calling: Calling | None = None,
    **arguments: Any,
) -> AsyncIterator[str]: ...
```

**Parameters:**

- `calling` (Calling | None): Pre-created Calling instance with `streaming=True`. If provided, `**arguments` are IGNORED.
- `**arguments`: Request arguments passed to `create_calling()`. IGNORED if `calling` provided.

**Yields**: String chunks from the streaming response

**Raises**:

- `TimeoutError`: If rate limit acquisition times out
- `NotImplementedError`: If backend doesn't support streaming

**Note**: Streaming bypasses executor path (requires direct consumption).

**Examples:**

```python
# Stream chunks directly
async for chunk in model.invoke_stream(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story"}]
):
    print(chunk, end="", flush=True)

# Pre-created calling
calling = await model.create_calling(
    streaming=True,
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story"}]
)
async for chunk in model.invoke_stream(calling=calling):
    print(chunk, end="", flush=True)
```

#### `invoke_stream_with_channel()`

Stream with channel abstraction for fan-out.

**Signature:**

```python
async def invoke_stream_with_channel(
    self,
    **arguments: Any,
) -> StreamChannel: ...
```

**Parameters:**

- `**arguments`: Request arguments passed to `create_calling()`

**Returns**: `StreamChannel` instance ready for iteration and consumer attachment

**Examples:**

```python
from lionpride.operations.streaming import StreamChannel

# Create channel
channel = await model.invoke_stream_with_channel(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story"}]
)

# Add consumers for real-time processing
channel.add_consumer(lambda chunk: print(chunk.content, end=""))
channel.add_consumer(lambda chunk: log_to_file(chunk.content))

# Consume stream
async for chunk in channel:
    pass  # Consumers already processing

# Get accumulated text
full_text = channel.get_accumulated()
```

### Serialization

#### `to_dict()`

Serialize to dict, excluding `id`/`created_at` for fresh identity on reconstruction.

**Signature:**

```python
def to_dict(self, **kwargs: Any) -> dict[str, Any]: ...
```

**Returns**: `dict[str, Any]` - Serialized iModel (excludes `id`, `created_at`, `backend.circuit_breaker`, `backend.retry_config`)

**Note**:

- Tool backends serialize `backend` as None and will fail on deserialization
- Endpoint backends serialize fully (except callables in circuit_breaker/retry_config)
- API keys: Only env var names serialized (if `api_key_is_env=True`), raw credentials cleared

**Examples:**

```python
# Serialize Endpoint-backed iModel
model = iModel(provider="anthropic", api_key="ANTHROPIC_API_KEY")
config = model.to_dict()

import json
with open("model.json", "w") as f:
    json.dump(config, f)

# Tool-backed iModel fails
from lionpride.services import Tool
tool_model = iModel(backend=Tool(func_callable=my_func))
config = tool_model.to_dict()  # backend=None, deserialization will fail
```

#### `from_dict()`

Reconstruct iModel from dict via Element polymorphic deserialization.

**Signature:**

```python
@classmethod
def from_dict(cls, data: dict[str, Any], **kwargs) -> iModel: ...
```

**Parameters**:

- `data` (dict): Serialized iModel dict (from `to_dict()`)
- `**kwargs`: Additional kwargs (not typically needed)

**Returns**: `iModel` instance with fresh identity (new UUID, fresh rate limiter capacity)

**Raises**:

- `ValueError`: If `backend` is None (Tool serialization) or env var not found

**Examples:**

```python
# Load from file
import json
with open("model.json", "r") as f:
    config = json.load(f)

restored = iModel.from_dict(config)

# API key env var must still exist
calling = await restored.invoke(...)
```

### Properties

#### `name`

Service name from backend.

**Returns**: `str` - Backend name

```python
model = iModel(provider="anthropic")
print(model.name)  # "anthropic_messages"
```

#### `version`

Service version from backend.

**Returns**: `str` - Backend version

```python
model = iModel(provider="anthropic")
print(model.version)  # "2023-06-01" or None
```

#### `tags`

Service tags from backend.

**Returns**: `set[str]` - Backend tags

```python
model = iModel(provider="anthropic")
print(model.tags)  # {"llm", "anthropic"}
```

## Protocol Implementations

This class implements the following protocols (declared via `@implements()`):

- **Invocable**: Async execution via `invoke()`
- **Observable**: UUID identifier via `id` property (inherited from Element)
- **Serializable**: Supports `to_dict()` (inherited from Element)
- **Deserializable**: Supports `from_dict()` (inherited from Element)

See [Protocols Guide](../protocols.md) for implementation patterns.

## Usage Patterns

### Basic Usage: Auto-Matching

```python
from lionpride.services import iModel

# Auto-match provider to endpoint
model = iModel(provider="anthropic", api_key="ANTHROPIC_API_KEY")

# Single call
calling = await model.invoke(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(calling.response.data)  # "Hello! How can I help you?"
```

### Advanced Usage: Rate Limiting with Executor

```python
from lionpride.services import iModel

# Auto-construct executor with rate limits (lionagi v0 pattern)
model = iModel(
    provider="openai",
    api_key="OPENAI_API_KEY",
    limit_requests=50,      # 50 requests per minute
    limit_tokens=100000,    # 100k tokens per minute
    capacity_refresh_time=60.0,
)

# Process batch with automatic rate limiting
tasks = [
    model.invoke(model="gpt-4o-mini", messages=[{"role": "user", "content": f"Task {i}"}])
    for i in range(100)
]
results = await asyncio.gather(*tasks)
```

### Advanced Usage: Circuit Breaker and Retry

```python
from lionpride.services import iModel
from lionpride.services.utilities.resilience import CircuitBreaker, RetryConfig
from lionpride.services.providers.anthropic_messages import AnthropicMessagesEndpoint

# Create backend with resilience patterns
backend = AnthropicMessagesEndpoint(
    api_key="ANTHROPIC_API_KEY",
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

model = iModel(backend=backend)

# Calls automatically retry on transient failures
# Circuit opens after 5 consecutive failures
calling = await model.invoke(...)
```

### Advanced Usage: Lifecycle Hooks

```python
from lionpride.services import iModel
from lionpride.services.types.hook import HookRegistry, HookPhase

# Define hooks
async def log_request(event, **kwargs):
    print(f"[PRE] Request: {event.payload}")

async def log_response(event, **kwargs):
    print(f"[POST] Response: {event.execution.response.data}")

# Create hook registry
hooks = HookRegistry()
hooks.register(HookPhase.PreInvocation, log_request)
hooks.register(HookPhase.PostInvocation, log_response)

# Attach to model
model = iModel(
    provider="openai",
    api_key="OPENAI_API_KEY",
    hook_registry=hooks,
)

# Hooks execute automatically
calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
# Output:
# [PRE] Request: {...}
# [POST] Response: Hello! How can I help you?
```

### Advanced Usage: Streaming with Channel

```python
from lionpride.services import iModel

model = iModel(provider="openai", api_key="OPENAI_API_KEY")

# Create channel for fan-out
channel = await model.invoke_stream_with_channel(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story"}]
)

# Add consumers
chunks_list = []
channel.add_consumer(lambda chunk: print(chunk.content, end="", flush=True))
channel.add_consumer(lambda chunk: chunks_list.append(chunk.content))

# Consume stream
async for chunk in channel:
    pass

# Get accumulated text
full_text = channel.get_accumulated()
print(f"\n\nTotal chunks: {len(chunks_list)}")
```

### Advanced Usage: Serialization and Persistence

```python
from lionpride.services import iModel
import json

# Create model with Endpoint backend
model = iModel(
    provider="anthropic",
    api_key="ANTHROPIC_API_KEY",
    limit_requests=50,
    limit_tokens=100000,
)

# Serialize to dict (excludes runtime state)
config = model.to_dict()

# Store in database or file
with open("model_config.json", "w") as f:
    json.dump(config, f)

# Restore from dict (fresh capacity)
with open("model_config.json", "r") as f:
    config = json.load(f)

restored = iModel.from_dict(config)

# API key must still be in environment
calling = await restored.invoke(...)
```

## Common Pitfalls

### Issue: Tool backends cannot serialize

**Problem**: `Tool` backends contain callables which are not serializable.

```python
# ❌ WRONG
from lionpride.services import iModel, Tool
tool_model = iModel(backend=Tool(func_callable=my_function))
config = tool_model.to_dict()  # backend=None in serialized dict
restored = iModel.from_dict(config)  # ValueError: backend is required
```

**Solution**: Use Endpoint backends for persistence scenarios.

```python
# ✅ CORRECT
from lionpride.services import iModel
model = iModel(provider="anthropic", api_key="...")
config = model.to_dict()  # Full serialization
restored = iModel.from_dict(config)  # Works
```

### Issue: API key env var missing after deserialization

**Problem**: Env var renamed or removed between serialization and deserialization.

```python
# ❌ WRONG
model = iModel(provider="anthropic", api_key="OLD_API_KEY_ENV_VAR")
config = model.to_dict()
# ... env var renamed to NEW_API_KEY_ENV_VAR
restored = iModel.from_dict(config)  # ValueError: env var not found
```

**Solution**: Ensure env var consistency or use `SecretStr` for raw credentials (cleared on serialization).

```python
# ✅ CORRECT (env var)
model = iModel(provider="anthropic", api_key="ANTHROPIC_API_KEY")
config = model.to_dict()
# Ensure ANTHROPIC_API_KEY still exists
restored = iModel.from_dict(config)

# ✅ CORRECT (raw credential - not serialized)
from pydantic import SecretStr
model = iModel(provider="anthropic", api_key=SecretStr("sk-..."))
config = model.to_dict()  # api_key=None
# Reconstructed model won't have credential, must provide again
```

### Issue: Poll timeout too short for slow models

**Problem**: Default `poll_timeout` (10s) too short for large models or long tasks.

```python
# ❌ WRONG
model = iModel(provider="anthropic", limit_requests=10)
calling = await model.invoke(
    model="claude-opus-4",
    messages=[{"role": "user", "content": "Write an essay"}]
)  # TimeoutError after 10s
```

**Solution**: Increase `poll_timeout` for slow models.

```python
# ✅ CORRECT
calling = await model.invoke(
    poll_timeout=180.0,  # 3 minutes for large models
    model="claude-opus-4",
    messages=[{"role": "user", "content": "Write an essay"}]
)
```

### Issue: Executor state not preserved on serialization

**Problem**: Assuming depleted capacity is serialized.

```python
# ❌ WRONG ASSUMPTION
model = iModel(provider="openai", limit_requests=50)
# ... consume 40 requests (10 remaining)
config = model.to_dict()
restored = iModel.from_dict(config)
# Restored model has FULL capacity (50 requests), not 10
```

**Solution**: Understand serialization resets capacity. If persistent tracking needed, implement custom solution.

```python
# ✅ CORRECT UNDERSTANDING
# Serialization stores configuration (capacity=50), not state (tokens=10)
# Deserialization creates fresh TokenBucket with full capacity
# This is intentional to prevent capacity leak across restarts
```

## Design Rationale

### Why wrap ServiceBackend?

`iModel` adds layers that `ServiceBackend` doesn't provide:

1. **Rate Limiting**: `TokenBucket` for simple blocking, `Executor` for event-driven processing
2. **Lifecycle Hooks**: Pre/post invocation without backend changes
3. **Provider Auto-Matching**: `provider="anthropic"` → `AnthropicMessagesEndpoint`
4. **Serialization**: Persist model configuration for multi-session workflows
5. **Streaming Channel**: Fan-out streaming to multiple consumers

### Why separate Calling creation from invocation?

`create_calling()` + `invoke(calling=...)` enables:

1. **Custom Configuration**: Set timeout, streaming flags after creation
2. **Hook Attachment**: Configure hooks before invocation
3. **Retry Logic**: Re-invoke same calling with different settings
4. **Testing**: Create calling without invoking for unit tests

### Why auto-construct Executor?

Executor auto-construction (lionagi v0 pattern) simplifies rate limiting:

```python
# Before (manual):
from lionpride.core import Executor
from lionpride.services.execution import RateLimitedExecutor
executor = RateLimitedExecutor(processor_config={...})
model = iModel(backend=backend, executor=executor)

# After (auto):
model = iModel(provider="openai", limit_requests=50, limit_tokens=100000)
```

Auto-construction reduces boilerplate while maintaining flexibility (explicit `executor` overrides).

### Why store provider_metadata?

Provider-specific state (e.g., Claude Code `session_id`) enables:

1. **Context Continuation**: Resume previous sessions automatically
2. **Provider Quirks**: Handle provider-specific requirements without subclassing
3. **Metadata Passthrough**: Store custom provider data for hooks/observers

## See Also

- **Related Classes**:
  - [`ServiceBackend`](providers.md#servicebackend): Abstract backend protocol
  - [`Endpoint`](providers.md#endpoint): HTTP API backend
  - [`Tool`](providers.md#tool): Python callable backend
  - [`ServiceRegistry`](registry.md): Service discovery and management
- **Resilience**:
  - [`TokenBucket`](rate_limiting.md#tokenbucket): Rate limiting
  - [`CircuitBreaker`](resilience.md#circuitbreaker): Fail-fast pattern
  - [`RetryConfig`](resilience.md#retryconfig): Retry with backoff
- **Core**:
  - [Event](../base/event.md): Base event system
  - [Executor](../base/executor.md): Event-driven processing
  - [Element](../base/element.md): UUID-based identity

## Examples

### Example 1: Multi-Provider Setup

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Register multiple providers
for provider in ["anthropic", "openai", "groq"]:
    model = iModel(
        provider=provider,
        api_key=f"{provider.upper()}_API_KEY",
        limit_requests=50,
        limit_tokens=100000,
    )
    registry.register(model)

# Use based on requirements
fast_model = registry.get("groq_chat")  # For speed
smart_model = registry.get("anthropic_messages")  # For reasoning
cheap_model = registry.get("openai_chat")  # For cost

# Process with appropriate model
calling = await smart_model.invoke(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### Example 2: Custom Timeout and Retry

```python
from lionpride.services import iModel
from lionpride.services.providers.anthropic_messages import AnthropicMessagesEndpoint
from lionpride.services.utilities.resilience import RetryConfig

backend = AnthropicMessagesEndpoint(
    api_key="ANTHROPIC_API_KEY",
    retry_config=RetryConfig(
        max_retries=5,
        initial_delay=2.0,
        exponential_base=2.0,
    ),
)

model = iModel(backend=backend)

# Create calling with custom timeout
calling = await model.create_calling(
    timeout=180.0,  # 3 minutes
    model="claude-opus-4",
    messages=[{"role": "user", "content": "Write a detailed essay"}]
)

# Invoke with custom poll timeout
calling = await model.invoke(calling=calling, poll_timeout=200.0)
```

### Example 3: Streaming with Multiple Consumers

```python
from lionpride.services import iModel
import asyncio

model = iModel(provider="openai", api_key="OPENAI_API_KEY")

# Create channel
channel = await model.invoke_stream_with_channel(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story about AI"}]
)

# Add multiple consumers
terminal_output = []
file_output = []

channel.add_consumer(lambda chunk: print(chunk.content, end="", flush=True))
channel.add_consumer(lambda chunk: terminal_output.append(chunk.content))
channel.add_consumer(lambda chunk: file_output.append(chunk.content))

# Consume stream
async for chunk in channel:
    pass

# Get results
full_text = channel.get_accumulated()
with open("story.txt", "w") as f:
    f.write("".join(file_output))
```
