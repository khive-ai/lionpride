# Hook System

> Lifecycle hooks for event invocation phases with registry-based management.

## Overview

The hook system provides extension points at critical phases of event execution.
`HookRegistry` manages callbacks for `PreEventCreate`, `PreInvocation`,
`PostInvocation`, and `ErrorHandling` phases, plus stream handlers for chunk processing.
`HookEvent` wraps hook execution as an `Event` for lifecycle tracking, while
`HookBroadcaster` enables pub/sub notification of hook events.

Hooks enable:

- **Validation**: Reject or modify events before creation/invocation
- **Logging/Monitoring**: Observe event lifecycle without modifying behavior
- **Error Recovery**: Custom error handling and retry logic
- **Streaming**: Process chunks during streaming responses

## Architecture

```text
                    +------------------+
                    |    iModel        |
                    |  (hook_registry) |
                    +--------+---------+
                             |
                             v
+------------+     +------------------+     +----------------+
| HookPhase  |---->|   HookRegistry   |<----| StreamHandlers |
| (enum)     |     |                  |     | (chunk types)  |
+------------+     +--------+---------+     +----------------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
  PreEventCreate     PreInvocation      PostInvocation
     (type)            (event)            (event)
          |                  |                  |
          v                  v                  v
    +-----------+      +-----------+      +-----------+
    | HookEvent |      | HookEvent |      | HookEvent |
    +-----------+      +-----------+      +-----------+
          |                  |                  |
          +------------------+------------------+
                             |
                             v
                    +------------------+
                    | HookBroadcaster  |
                    |   (singleton)    |
                    +------------------+
```

## HookPhase

Enum defining the four lifecycle phases where hooks can intercept execution.

```python
class HookPhase(Enum):
    PreEventCreate = "pre_event_create"
    PreInvocation = "pre_invocation"
    PostInvocation = "post_invocation"
    ErrorHandling = "error_handling"
```

### Phase Descriptions

| Phase            | Trigger Point                  | Event Type       | Use Cases                          |
| ---------------- | ------------------------------ | ---------------- | ---------------------------------- |
| `PreEventCreate` | Before event instantiation     | `type[Event]`    | Validation, caching, rate limiting |
| `PreInvocation`  | Before `_invoke()` execution   | `Event` instance | Logging, modification, blocking    |
| `PostInvocation` | After successful `_invoke()`   | `Event` instance | Logging, metrics, transformation   |
| `ErrorHandling`  | On exception during invocation | `Event` instance | Recovery, retry logic, alerting    |

## HookEvent

Event subclass that delegates to `HookRegistry` for hook execution. Extends
`lionpride.Event` with hook-specific fields and invocation logic.

### Class Signature

```python
class HookEvent(Event):
    """Hook execution event that delegates to HookRegistry."""

    def __init__(
        self,
        registry: HookRegistry,
        hook_phase: HookPhase,
        event_like: Event | type[Event],
        exit: bool = False,
        params: dict[str, Any] = ...,
        **kwargs: Any,
    ) -> None: ...
```

### Parameters

| Parameter    | Type                   | Default | Description                                    |
| ------------ | ---------------------- | ------- | ---------------------------------------------- |
| `registry`   | `HookRegistry`         | -       | Registry containing hook callbacks             |
| `hook_phase` | `HookPhase`            | -       | Lifecycle phase being executed                 |
| `event_like` | `Event or type[Event]` | -       | Target event instance or type                  |
| `exit`       | `bool`                 | `False` | Whether to exit pipeline on hook completion    |
| `params`     | `dict[str, Any]`       | `{}`    | Additional parameters passed to hook callbacks |

### Attributes

| Attribute               | Type                          | Description                                              |
| ----------------------- | ----------------------------- | -------------------------------------------------------- |
| `registry`              | `HookRegistry`                | Registry for hook dispatch (excluded from serialization) |
| `hook_phase`            | `HookPhase`                   | Current lifecycle phase                                  |
| `exit`                  | `bool`                        | Exit flag for pipeline control                           |
| `params`                | `dict[str, Any]`              | Hook parameters (excluded from serialization)            |
| `event_like`            | `Event or type[Event]`        | Target event (excluded from serialization)               |
| `associated_event_info` | `AssociatedEventInfo or None` | Metadata about the target event                          |

### Private Attributes

| Attribute      | Type                    | Description                |
| -------------- | ----------------------- | -------------------------- |
| `_should_exit` | `bool`                  | Whether hook signaled exit |
| `_exit_cause`  | `BaseException or None` | Exception that caused exit |

## HookRegistry

Central registry for hook callbacks and stream handlers. Thread-safe for concurrent
access. Validates all handlers on registration.

### Class Signature

```python
class HookRegistry:
    """Registry for hook callbacks at event lifecycle phases."""

    def __init__(
        self,
        hooks: dict[HookPhase, Callable[..., Any]] | None = None,
        stream_handlers: StreamHandlers[Any] | None = None,
    ) -> None: ...
```

### Parameters

| Parameter         | Type                                | Default | Description                   |
| ----------------- | ----------------------------------- | ------- | ----------------------------- |
| `hooks`           | `dict[HookPhase, Callable] or None` | `None`  | Phase-to-callback mapping     |
| `stream_handlers` | `StreamHandlers[Any] or None`       | `None`  | Chunk type-to-handler mapping |

### Methods

#### Lifecycle Phase Methods

##### `pre_event_create()`

Call hook before event creation. Receives event type, can return modified params or
raise to cancel.

```python
async def pre_event_create(
    self,
    event_type: type[Event],
    /,
    exit: bool = False,
    **kw: Any,
) -> tuple[Any, bool, EventStatus]: ...
```

**Returns**: `(result, should_exit, status)` tuple

- `result`: Hook return value, `(Undefined, exception)` on cancel, or exception on error
- `should_exit`: `True` if pipeline should terminate
- `status`: `COMPLETED`, `CANCELLED`, or error status

---

##### `pre_invocation()`

Call hook before event execution. Receives event instance, can modify or cancel.

```python
async def pre_invocation(
    self,
    event: Event,
    /,
    exit: bool = False,
    **kw: Any,
) -> tuple[Any, bool, EventStatus]: ...
```

---

##### `post_invocation()`

Call hook after successful execution. Receives completed event, can transform result or
abort.

```python
async def post_invocation(
    self,
    event: Event,
    /,
    exit: bool = False,
    **kw: Any,
) -> tuple[Any, bool, EventStatus]: ...
```

---

##### `handle_streaming_chunk()`

Process streaming chunk via registered stream handler.

```python
async def handle_streaming_chunk(
    self,
    chunk_type: str | type,
    chunk: Any,
    /,
    exit: bool = False,
    **kw: Any,
) -> tuple[Any, bool, EventStatus | None]: ...
```

**Raises**: `ValueError` if `chunk_type` is `None`

---

#### Unified Call Method

##### `call()`

Dispatch to appropriate lifecycle method or stream handler based on parameters.

```python
async def call(
    self,
    event_like: Event | type[Event],
    /,
    *,
    hook_phase: HookPhase | None = None,
    chunk_type: str | type | None = None,
    chunk: Any = None,
    exit: bool = False,
    **kw: Any,
) -> tuple[tuple[Any, bool, EventStatus], dict[str, Any]] | tuple[Any, bool, EventStatus | None]: ...
```

**Parameters**:

- `event_like`: Event instance or type depending on phase
- `hook_phase`: Lifecycle phase to call (mutually exclusive with `chunk_type`)
- `chunk_type`: Stream chunk type (mutually exclusive with `hook_phase`)
- `chunk`: Chunk data for stream handlers
- `exit`: Exit flag
- `**kw`: Additional parameters passed to hook

**Returns**:

- For hooks: `((result, should_exit, status), meta_dict)`
- For streams: `(result, should_exit, status)`

**Raises**: `ValueError` if neither `hook_phase` nor `chunk_type` provided

---

#### Capability Check

##### `_can_handle()`

Check if registry has handler for given phase or chunk type.

```python
def _can_handle(
    self,
    /,
    *,
    hp_: HookPhase | None = None,
    ct_: str | type | None = None,
) -> bool: ...
```

## HookBroadcaster

Singleton broadcaster for `HookEvent` notifications. Extends `Broadcaster` with
`HookEvent` as the event type. Uses weakref-based subscriber management for automatic
cleanup.

### Class Signature

```python
class HookBroadcaster(Broadcaster):
    _event_type: ClassVar[type[HookEvent]] = HookEvent
```

### Inherited Methods

| Method                   | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `subscribe(callback)`    | Add subscriber callback (weakref for auto-cleanup) |
| `unsubscribe(callback)`  | Remove subscriber callback                         |
| `broadcast(event)`       | Notify all subscribers (async)                     |
| `get_subscriber_count()` | Count live subscribers                             |

## Type Aliases

### StreamHandlers

```python
SC = TypeVar("SC")
StreamHandlers = dict[str | type, Callable[[SC], Awaitable[None]]]
```

Mapping of chunk type names (strings) or types to async handler functions. Each handler
receives the chunk and processes it (e.g., logging, accumulation, transformation).

### AssociatedEventInfo

```python
class AssociatedEventInfo(TypedDict, total=False):
    lion_class: str        # Fully qualified event class name
    event_id: str          # Event UUID (for instance phases)
    event_created_at: str  # ISO timestamp (for instance phases)
```

## Validation Functions

### `validate_hooks()`

Validate hook dictionary structure and callables.

```python
def validate_hooks(kw: dict) -> None: ...
```

**Raises**: `ValueError` if:

- `kw` is not a dictionary
- Any key is not a valid `HookPhase`
- Any value is not callable

---

### `validate_stream_handlers()`

Validate stream handler dictionary structure.

```python
def validate_stream_handlers(kw: dict) -> None: ...
```

**Raises**: `ValueError` if:

- `kw` is not a dictionary
- Any key is not a string or type
- Any value is not callable

## Usage Patterns

### Basic Hook Registration

```python
from lionpride.services.types import HookPhase, HookRegistry

async def log_pre_invoke(event, **kwargs):
    print(f"About to invoke: {event.id}")
    return event  # Pass through

async def log_post_invoke(event, **kwargs):
    print(f"Completed: {event.id}, status={event.status}")
    return event.response

registry = HookRegistry(
    hooks={
        HookPhase.PreInvocation: log_pre_invoke,
        HookPhase.PostInvocation: log_post_invoke,
    }
)
```

### Integration with iModel

```python
from lionpride import iModel
from lionpride.services.types import HookPhase, HookRegistry

async def validate_request(event_type, **kwargs):
    """Validate request before event creation."""
    messages = kwargs.get("messages", [])
    if not messages:
        raise ValueError("Messages cannot be empty")
    return kwargs

async def log_response(event, **kwargs):
    """Log response after invocation."""
    print(f"Response received: {event.status}")
    return event.response

registry = HookRegistry(
    hooks={
        HookPhase.PreEventCreate: validate_request,
        HookPhase.PostInvocation: log_response,
    }
)

model = iModel(
    provider="openai",
    model="gpt-4o-mini",
    hook_registry=registry,
)

calling = await model.invoke(
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Stream Handler for Token Counting

```python
from lionpride.services.types import HookRegistry

token_count = 0

async def count_tokens(event, chunk_type, chunk, **kwargs):
    """Count tokens from streaming chunks."""
    global token_count
    if hasattr(chunk, "usage"):
        token_count += chunk.usage.get("completion_tokens", 0)
    return chunk

registry = HookRegistry(
    stream_handlers={
        "text_chunk": count_tokens,
        "completion_chunk": count_tokens,
    }
)
```

### Cancellation Hook

```python
from lionpride.services.types import HookPhase, HookRegistry

async def rate_limit_check(event_type, **kwargs):
    """Cancel if rate limit exceeded."""
    if is_rate_limited():
        raise RuntimeError("Rate limit exceeded, try again later")
    return kwargs

registry = HookRegistry(
    hooks={HookPhase.PreEventCreate: rate_limit_check}
)
```

### Broadcasting Hook Events

```python
from lionpride.services.types import HookBroadcaster, HookEvent

# Subscribe to hook events
async def on_hook_event(event: HookEvent):
    print(f"Hook executed: {event.hook_phase}, exit={event.exit}")

HookBroadcaster.subscribe(on_hook_event)

# Later, unsubscribe
HookBroadcaster.unsubscribe(on_hook_event)
```

## Common Pitfalls

- **Sync handlers in async context**: All hook handlers should be async. Sync functions
  are automatically wrapped but add overhead. Always use `async def` for hooks.

- **Missing return in hooks**: Hooks that should pass through data must return it
  explicitly. A hook returning `None` will propagate `None` as the result.

- **Exception swallowing**: Exceptions in `PreEventCreate` and `PreInvocation` cancel
  the operation. In `PostInvocation`, they abort. Don't catch exceptions unless you
  intend to suppress the cancellation/abort behavior.

- **Circular hook registration**: Hooks can trigger other hooks via `HookEvent`. Avoid
  registering hooks that call `registry.call()` to prevent infinite loops.

- **Stream handler key mismatch**: Chunk types must match exactly (string or type).
  `"text_chunk"` and `TextChunk` are different keys.

## Design Rationale

**Phase-Based Architecture**: Four distinct phases provide clear interception points
without coupling to specific implementations. Each phase receives appropriate context
(type vs instance).

**Registry Pattern**: Centralized registration enables runtime reconfiguration and
testing. Single registry instance can be shared across multiple `iModel` instances.

**Event Wrapping**: `HookEvent` extends `Event` for consistent lifecycle management
(status tracking, timing, error capture). Hook execution benefits from the same
guarantees as service invocations.

**Weakref Subscribers**: `HookBroadcaster` uses weakrefs to prevent memory leaks from
forgotten unsubscribes. Callbacks are automatically cleaned up when their owners are
garbage collected.

**Validation on Registration**: `validate_hooks()` and `validate_stream_handlers()` fail
fast on invalid configuration, preventing runtime errors during critical invocation
paths.

## See Also

- [`iModel`](./imodel.md): Service interface that integrates `HookRegistry`
- [`Event`](../base/event.md): Base event class with lifecycle management
- [`Broadcaster`](../base/broadcaster.md): Pub/sub base class
- [`EventStatus`](../base/event.md): Event execution status enum

## Example: Complete Observability Pipeline

```python
import logging
from lionpride import iModel
from lionpride.services.types import HookPhase, HookRegistry, HookBroadcaster

logger = logging.getLogger(__name__)

# Metrics accumulator
metrics = {"requests": 0, "errors": 0, "tokens": 0}

async def track_request(event_type, **kwargs):
    """Track request initiation."""
    metrics["requests"] += 1
    logger.info(f"Request #{metrics['requests']} starting")
    return kwargs

async def handle_error(event, **kwargs):
    """Log and track errors."""
    metrics["errors"] += 1
    logger.error(f"Request failed: {event.execution.error}")
    return None  # Allow error to propagate

async def track_completion(event, **kwargs):
    """Track successful completion."""
    if hasattr(event, "response") and event.response:
        usage = getattr(event.response, "usage", {})
        metrics["tokens"] += usage.get("total_tokens", 0)
    logger.info(f"Request completed, total tokens: {metrics['tokens']}")
    return event.response

async def stream_token_counter(event, chunk_type, chunk, **kwargs):
    """Count tokens during streaming."""
    if hasattr(chunk, "choices"):
        for choice in chunk.choices:
            if hasattr(choice, "delta") and choice.delta.content:
                metrics["tokens"] += len(choice.delta.content.split())
    return chunk

# Create registry with full observability
registry = HookRegistry(
    hooks={
        HookPhase.PreEventCreate: track_request,
        HookPhase.PostInvocation: track_completion,
        HookPhase.ErrorHandling: handle_error,
    },
    stream_handlers={
        "stream_chunk": stream_token_counter,
    },
)

# Use with iModel
model = iModel(
    provider="openai",
    model="gpt-4o-mini",
    hook_registry=registry,
)

# Execute with full observability
calling = await model.invoke(
    messages=[{"role": "user", "content": "Explain hooks in 3 sentences."}],
)

print(f"Metrics: {metrics}")
# Metrics: {'requests': 1, 'errors': 0, 'tokens': 42}
```
