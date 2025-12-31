# ServiceBackend

> Base class for all service backends providing unified service abstraction with Element
> identity and hook lifecycle support.

## Overview

`ServiceBackend` is the abstract base class for lionpride service integrations. It
inherits from `Element` for UUID-based identity and defines the contract that all
backends must implement: an `event_type` property and an async `call()` method.

Two concrete implementations extend ServiceBackend:

- **[Tool](tool.md)**: Wraps Python callables for LLM tool calling
- **[Endpoint](endpoint.md)**: HTTP API integration with authentication and resilience

This module also provides `ServiceConfig` for backend configuration,
`NormalizedResponse` for consistent response handling, and `Calling` as the base event
class for hook-enabled invocations.

## Class Hierarchy

```text
Element
    |
    +-- ServiceBackend (abstract)
            |
            +-- Tool (callable wrapper)
            +-- Endpoint (HTTP API)

Event
    |
    +-- Calling (abstract, with hooks)
            |
            +-- ToolCalling (for Tool)
            +-- APICalling (for Endpoint)
```

---

## ServiceConfig

> Configuration container for service backends.

### Class Signature

```python
class ServiceConfig(HashableModel):
    """Configuration for service backends."""

    def __init__(
        self,
        provider: str,
        name: str,
        request_options: type[BaseModel] | None = None,
        timeout: int = 300,
        max_retries: int = 3,
        version: str | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

### Parameters

| Parameter         | Type                      | Default  | Description                                                                  |
| ----------------- | ------------------------- | -------- | ---------------------------------------------------------------------------- |
| `provider`        | `str`                     | Required | Provider identifier (4-50 chars). E.g., `"openai"`, `"anthropic"`, `"tool"`. |
| `name`            | `str`                     | Required | Service name (4-100 chars). Used for registry lookup.                        |
| `request_options` | `type[BaseModel] or None` | `None`   | Pydantic model for request validation.                                       |
| `timeout`         | `int`                     | `300`    | Request timeout in seconds (1-3600).                                         |
| `max_retries`     | `int`                     | `3`      | Maximum retry attempts (0-10).                                               |
| `version`         | `str or None`             | `None`   | Service version string.                                                      |
| `tags`            | `list[str]`               | `[]`     | Custom categorization tags.                                                  |
| `**kwargs`        | `Any`                     | -        | Extra fields stored in `kwargs` dict.                                        |

### Methods

#### `validate_payload(data) -> dict`

Validate request data against `request_options` schema.

```python
config = ServiceConfig(
    provider="openai",
    name="chat",
    request_options=ChatRequest,  # Pydantic model
)
validated = config.validate_payload({"model": "gpt-4o", "messages": [...]})
```

**Raises**: `ValueError` if validation fails.

---

## NormalizedResponse

> Generic normalized response for all service backends.

Provides a consistent interface regardless of backend type (HTTP endpoints, tools, LLM
APIs).

### Class Signature

```python
class NormalizedResponse(HashableModel):
    """Generic normalized response for all service backends."""

    def __init__(
        self,
        status: str,
        raw_response: dict[str, Any],
        data: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
```

### Attributes

| Attribute      | Type                     | Description                                            |
| -------------- | ------------------------ | ------------------------------------------------------ |
| `status`       | `str`                    | Response status: `"success"` or `"error"`.             |
| `data`         | `Any`                    | Processed response data (extracted from raw response). |
| `error`        | `str or None`            | Error message if `status="error"`.                     |
| `raw_response` | `dict[str, Any]`         | Original unmodified response from backend.             |
| `metadata`     | `dict[str, Any] or None` | Provider-specific metadata.                            |

### Usage

```python
# Success response
response = NormalizedResponse(
    status="success",
    data={"content": "Hello, world!"},
    raw_response={"choices": [{"message": {"content": "Hello, world!"}}]},
    metadata={"model": "gpt-4o", "usage": {"total_tokens": 50}},
)

# Access data
print(response.data["content"])  # "Hello, world!"
print(response.status)            # "success"

# Error response
error_response = NormalizedResponse(
    status="error",
    data=None,
    error="Rate limit exceeded",
    raw_response={"error": {"message": "Rate limit exceeded"}},
)
```

---

## Calling

> Base calling event with hook support for service invocations.

Extends `lionpride.Event` with pre/post invocation hooks. Always delegates to
`backend.call()` for actual service invocation.

### Class Signature

```python
class Calling(Event):
    """Base calling event with hook support."""

    def __init__(
        self,
        backend: ServiceBackend,
        payload: dict[str, Any],
        timeout: float | None = None,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None: ...
```

### Parameters

| Parameter   | Type             | Description                                                             |
| ----------- | ---------------- | ----------------------------------------------------------------------- |
| `backend`   | `ServiceBackend` | Service backend instance (Tool, Endpoint). Excluded from serialization. |
| `payload`   | `dict[str, Any]` | Request payload/arguments for backend call.                             |
| `timeout`   | `float or None`  | Event timeout in seconds (inherited from Event).                        |
| `streaming` | `bool`           | Enable streaming mode (inherited from Event).                           |

### Attributes

| Attribute  | Type                          | Description                                             |
| ---------- | ----------------------------- | ------------------------------------------------------- |
| `backend`  | `ServiceBackend`              | Service backend instance (excluded from serialization). |
| `payload`  | `dict[str, Any]`              | Request payload/arguments.                              |
| `response` | `NormalizedResponse or Unset` | Response after execution (property).                    |

### Properties

#### `call_args` (abstract)

Get arguments for `backend.call(**self.call_args)`. Subclasses must implement:

- `APICalling`: Returns
  `{"request": ..., "extra_headers": ..., "skip_payload_creation": True}`
- `ToolCalling`: Returns `{"arguments": ...}`

#### `response`

Get normalized response from execution. Returns `Unset` if not yet executed.

```python
calling = await backend.event_type(backend=backend, payload={"query": "test"})
await calling.invoke()
if calling.response is not Unset:
    print(calling.response.data)
```

### Methods

#### `create_pre_invoke_hook(registry, ...) -> None`

Create pre-invocation hook event.

```python
def create_pre_invoke_hook(
    self,
    hook_registry: HookRegistry,
    exit_hook: bool | None = None,
    hook_timeout: float = 30.0,
    hook_params: dict[str, Any] | None = None,
) -> None: ...
```

**Parameters**:

- `hook_registry`: Registry containing hook functions.
- `exit_hook`: If True, hook can abort invocation by raising.
- `hook_timeout`: Hook execution timeout.
- `hook_params`: Parameters passed to hook function.

#### `create_post_invoke_hook(registry, ...) -> None`

Create post-invocation hook event. Same signature as `create_pre_invoke_hook()`.

Post-hooks run in `finally` block (even on failure) for cleanup, metrics, or logging.

### Hook Lifecycle

```text
1. Pre-invocation hook (if configured)
   - Can abort invocation by raising
   - Propagates FAILED/CANCELLED status

2. backend.call(**self.call_args)
   - Actual service invocation

3. Post-invocation hook (if configured, runs in finally)
   - Runs even on failure
   - Failures logged but don't block response
```

---

## ServiceBackend

> Abstract base class for all service backends.

### Class Signature

```python
class ServiceBackend(Element):
    """Base class for all service backends (Tool, Endpoint, etc.)."""

    def __init__(
        self,
        config: ServiceConfig,
        **kwargs: Any,
    ) -> None: ...
```

### Parameters

| Parameter | Type            | Description                       |
| --------- | --------------- | --------------------------------- |
| `config`  | `ServiceConfig` | Service configuration (required). |

### Attributes

| Attribute    | Type             | Description                                                         |
| ------------ | ---------------- | ------------------------------------------------------------------- |
| `id`         | `UUID`           | Unique identifier (inherited from Element, auto-generated, frozen). |
| `created_at` | `datetime`       | UTC timestamp (inherited from Element, auto-generated, frozen).     |
| `metadata`   | `dict[str, Any]` | Arbitrary metadata (inherited from Element).                        |
| `config`     | `ServiceConfig`  | Service configuration.                                              |

### Properties

| Property          | Type                      | Description                                      |
| ----------------- | ------------------------- | ------------------------------------------------ |
| `provider`        | `str`                     | Provider name from config.                       |
| `name`            | `str`                     | Service name from config.                        |
| `version`         | `str or None`             | Service version from config.                     |
| `tags`            | `set[str]`                | Service tags from config.                        |
| `request_options` | `type[BaseModel] or None` | Request validation schema from config.           |
| `event_type`      | `type[Calling]`           | Abstract. Returns Calling type for this backend. |

### Methods

#### `event_type` (abstract property)

Return the Calling type for this backend. Implementations:

- `Tool.event_type` returns `ToolCalling`
- `Endpoint.event_type` returns `APICalling`

```python
@property
@abstractmethod
def event_type(self) -> type[Calling]:
    """Return Calling type for this backend (e.g., ToolCalling, APICalling)."""
    ...
```

#### `call(*args, **kwargs) -> NormalizedResponse` (abstract, async)

Execute service call and return normalized response.

```python
@abstractmethod
async def call(self, *args, **kw) -> NormalizedResponse:
    """Execute service call and return normalized response."""
    ...
```

**Implementations**:

- `Tool.call(arguments: dict)`: Executes wrapped callable.
- `Endpoint.call(request: dict, **kwargs)`: Makes HTTP request.

#### `normalize_response(raw_response) -> NormalizedResponse`

Normalize raw response into NormalizedResponse. Default wraps response as-is.

```python
def normalize_response(self, raw_response: Any) -> NormalizedResponse:
    """Normalize raw response into NormalizedResponse."""
    return NormalizedResponse(
        status="success",
        data=raw_response,
        raw_response=raw_response,
    )
```

Override in subclasses to extract specific fields or add metadata.

#### `stream(*args, **kwargs)` (async generator)

Stream responses. Default raises `NotImplementedError`.

```python
async def stream(self, *args, **kw):
    """Stream responses (not supported by default)."""
    raise NotImplementedError("This backend does not support streaming calls.")
```

`Endpoint` overrides this for SSE/streaming support.

---

## Protocol Implementations

### ServiceBackend

Inherits from Element: **Observable**, **Serializable**, **Deserializable**,
**Hashable**.

### Calling

Inherits from Event: **Observable**, **Serializable**, **Invocable**.

### ServiceConfig, NormalizedResponse

Extends **HashableModel**: **Serializable**, **Hashable**.

---

## Creating Custom Backends

### Step 1: Define Configuration (Optional)

Extend `ServiceConfig` if you need additional configuration fields:

```python
from lionpride.services.types import ServiceConfig

class MyBackendConfig(ServiceConfig):
    base_url: str
    custom_option: bool = False
```

### Step 2: Define Calling Event

Extend `Calling` to specify how your backend is invoked:

```python
from lionpride.services.types import Calling

class MyBackendCalling(Calling):
    @property
    def call_args(self) -> dict:
        """Arguments for backend.call()."""
        return {"data": self.payload, "option": self.payload.get("option")}
```

### Step 3: Implement ServiceBackend

```python
from lionpride.services.types import ServiceBackend, NormalizedResponse

class MyBackend(ServiceBackend):
    config: MyBackendConfig  # Type narrowing

    @property
    def event_type(self) -> type[MyBackendCalling]:
        return MyBackendCalling

    async def call(self, data: dict, option: str | None = None) -> NormalizedResponse:
        # Your implementation here
        result = await self._do_work(data, option)
        return self.normalize_response(result)

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        # Custom normalization
        return NormalizedResponse(
            status="success" if raw_response.get("ok") else "error",
            data=raw_response.get("result"),
            error=raw_response.get("error"),
            raw_response=raw_response,
        )
```

### Step 4: Register and Use

```python
from lionpride import ServiceRegistry, iModel

# Direct usage
backend = MyBackend(config=MyBackendConfig(
    provider="myservice",
    name="my-backend",
    base_url="https://api.example.com",
))
result = await backend.call(data={"key": "value"})

# With iModel wrapper (adds rate limiting, hooks)
model = iModel(backend=backend, limit_requests=100)
calling = await model.invoke(key="value")

# With registry
registry = ServiceRegistry()
registry.register(backend)
backend = registry.get("my-backend")
```

---

## Common Pitfalls

- **Missing abstract implementations**: `ServiceBackend` requires both `event_type`
  property and `call()` method. Subclasses that don't implement these will raise
  `TypeError` on instantiation.

- **Calling.call_args mismatch**: The `call_args` property must return kwargs that match
  `backend.call()` signature. Mismatches cause `TypeError` at runtime.

- **Hook failures**: Pre-invoke hook failures abort the call. Post-invoke hook failures
  are logged but don't affect the response.

- **Serialization of callables**: `Tool` backends cannot be serialized (callables are
  not serializable). Use `Endpoint` for persistence scenarios.

---

## Design Rationale

**Element Identity**: ServiceBackend inherits from Element for consistent UUID-based
identity across the framework, enabling O(1) lookup in Pile collections and registry.

**Abstract Contract**: The `event_type`/`call()` contract ensures all backends integrate
consistently with iModel, Processor, and the hook lifecycle.

**NormalizedResponse**: Provides a consistent interface regardless of backend type,
simplifying error handling and response processing.

**Calling + Hooks**: Event-based invocation with configurable hooks enables
observability, rate limiting, and cross-cutting concerns without modifying backend
logic.

---

## See Also

- [Tool](tool.md): Callable wrapper backend
- [Endpoint](endpoint.md): HTTP API backend
- [iModel](imodel.md): High-level service interface
- [ServiceRegistry](registry.md): Service management
- [Event](../core/event.md): Base event class
- [Element](../core/element.md): Base identity class
