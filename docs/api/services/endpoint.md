# Endpoint

> HTTP API backend with authentication, resilience patterns, and streaming support.

## Overview

`Endpoint` extends `ServiceBackend` for HTTP API integrations. It provides secure
credential handling (environment variable resolution), multiple authentication schemes,
resilience patterns (circuit breaker, retry with backoff), and SSE streaming.

**Use for**: LLM provider APIs (OpenAI, Anthropic, Gemini), REST APIs, webhooks. **Skip
for**: Local function execution (use [Tool](tool.md)).

This module provides:

- **EndpointConfig**: HTTP-specific configuration extending ServiceConfig
- **Endpoint**: HTTP backend with resilience and streaming
- **APICalling**: Event class for API invocations with token estimation

## Class Hierarchy

```text
ServiceBackend
    |
    +-- Endpoint (HTTP API)

Calling
    |
    +-- APICalling (for Endpoint)
```

---

## EndpointConfig

> Configuration for HTTP API endpoints with secure credential handling.

### Class Signature

```python
class EndpointConfig(ServiceConfig):
    """Configuration for Endpoint backend."""

    def __init__(
        self,
        provider: str,
        name: str,
        endpoint: str,
        base_url: str | None = None,
        endpoint_params: list[str] | None = None,
        method: str = "POST",
        params: dict[str, str] = {},
        content_type: str | None = "application/json",
        auth_type: AUTH_TYPES = "bearer",
        default_headers: dict = {},
        api_key: str | None = None,
        openai_compatible: bool = False,
        requires_tokens: bool = False,
        client_kwargs: dict = {},
        **kwargs: Any,
    ) -> None: ...
```

### Parameters

| Parameter           | Type                | Default              | Description                                                             |
| ------------------- | ------------------- | -------------------- | ----------------------------------------------------------------------- |
| `provider`          | `str`               | Required             | Provider identifier (inherited from ServiceConfig).                     |
| `name`              | `str`               | Required             | Service name for registry lookup (inherited from ServiceConfig).        |
| `endpoint`          | `str`               | Required             | API endpoint path (e.g., `"chat/completions"`).                         |
| `base_url`          | `str or None`       | `None`               | Base URL for API (e.g., `"https://api.openai.com/v1"`).                 |
| `endpoint_params`   | `list[str] or None` | `None`               | URL template parameters (e.g., `["model_id"]` for `models/{model_id}`). |
| `method`            | `str`               | `"POST"`             | HTTP method.                                                            |
| `params`            | `dict[str, str]`    | `{}`                 | Values for `endpoint_params` template substitution.                     |
| `content_type`      | `str or None`       | `"application/json"` | Content-Type header. Set to `None` to omit.                             |
| `auth_type`         | `AUTH_TYPES`        | `"bearer"`           | Authentication type: `"bearer"`, `"x-api-key"`, or `"none"`.            |
| `default_headers`   | `dict`              | `{}`                 | Additional headers merged into all requests.                            |
| `api_key`           | `str or None`       | `None`               | API key or environment variable name (see Security section).            |
| `openai_compatible` | `bool`              | `False`              | Flag for OpenAI-compatible APIs.                                        |
| `requires_tokens`   | `bool`              | `False`              | Enable token tracking for rate limiting.                                |
| `client_kwargs`     | `dict`              | `{}`                 | Extra kwargs passed to `httpx.AsyncClient`.                             |

### Properties

| Property      | Type          | Description                              |
| ------------- | ------------- | ---------------------------------------- |
| `full_url`    | `str`         | Constructed URL: `{base_url}/{endpoint}` |
| `api_key_env` | `str or None` | Alias for `api_key` (env var semantics)  |

### Authentication Types

The `auth_type` parameter controls how credentials are included in requests:

| Type          | Header Generated                  | Use Case                            |
| ------------- | --------------------------------- | ----------------------------------- |
| `"bearer"`    | `Authorization: Bearer {api_key}` | OpenAI, most OAuth2 APIs            |
| `"x-api-key"` | `x-api-key: {api_key}`            | Anthropic, AWS API Gateway          |
| `"none"`      | No auth header                    | Public APIs, custom auth in headers |

### API Key Security Model

EndpointConfig implements a secure credential handling model:

1. **Environment Variable Resolution**: If `api_key` matches pattern `^[A-Z][A-Z0-9_]*$`
   and exists in environment, it's treated as an env var name. The resolved value is
   stored in private `_api_key` (SecretStr, never serialized).

2. **Raw Credential Protection**: If `api_key` doesn't match env var pattern or env var
   doesn't exist, it's treated as a raw credential. The value is stored in `_api_key`
   and `api_key` is cleared (set to `None`) to prevent serialization leaks.

3. **System Variable Protection**: Common system env vars (`HOME`, `PATH`, `USER`, etc.)
   are blocked to prevent accidental collision.

4. **Serialization Safety**: Only the env var name (if applicable) is serialized, never
   the resolved credential.

```python
# Environment variable (recommended)
config = EndpointConfig(
    provider="openai",
    name="chat",
    endpoint="chat/completions",
    api_key="OPENAI_API_KEY",  # Resolved from os.environ
)
# Serializes api_key="OPENAI_API_KEY", api_key_is_env=True

# Raw credential (cleared after resolution)
config = EndpointConfig(
    provider="custom",
    name="api",
    endpoint="invoke",
    api_key="sk-abc123",  # Raw credential
)
# Serializes api_key=None (cleared for security)

# SecretStr for explicit raw credentials
from pydantic import SecretStr
endpoint = Endpoint(
    config={"provider": "custom", "name": "api", "endpoint": "invoke"},
    api_key=SecretStr("sk-abc123"),  # Bypasses env var detection
)
```

---

## Endpoint

> HTTP API backend with resilience patterns and streaming.

### Class Signature

```python
class Endpoint(ServiceBackend):
    """HTTP API backend with circuit breaker and retry support."""

    def __init__(
        self,
        config: dict | EndpointConfig,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
        **kwargs: Any,
    ) -> None: ...
```

### Parameters

| Parameter         | Type                     | Default  | Description                                   |
| ----------------- | ------------------------ | -------- | --------------------------------------------- |
| `config`          | `dict or EndpointConfig` | Required | Endpoint configuration.                       |
| `circuit_breaker` | `CircuitBreaker or None` | `None`   | Circuit breaker for fail-fast behavior.       |
| `retry_config`    | `RetryConfig or None`    | `None`   | Retry configuration with exponential backoff. |
| `**kwargs`        | `Any`                    | -        | Merged into config if config is dict.         |

### Attributes

| Attribute         | Type                     | Description                                                         |
| ----------------- | ------------------------ | ------------------------------------------------------------------- |
| `id`              | `UUID`                   | Unique identifier (inherited from Element, auto-generated, frozen). |
| `created_at`      | `datetime`               | UTC timestamp (inherited from Element, auto-generated, frozen).     |
| `metadata`        | `dict[str, Any]`         | Arbitrary metadata (inherited from Element).                        |
| `config`          | `EndpointConfig`         | Endpoint configuration.                                             |
| `circuit_breaker` | `CircuitBreaker or None` | Circuit breaker instance.                                           |
| `retry_config`    | `RetryConfig or None`    | Retry configuration.                                                |

### Properties

| Property          | Type                      | Description                                     |
| ----------------- | ------------------------- | ----------------------------------------------- |
| `full_url`        | `str`                     | Full URL from config (`{base_url}/{endpoint}`). |
| `event_type`      | `type[APICalling]`        | Returns `APICalling` class.                     |
| `name`            | `str`                     | Service name from config.                       |
| `provider`        | `str`                     | Provider from config.                           |
| `request_options` | `type[BaseModel] or None` | Request validation schema from config.          |

### Methods

#### `call(request, skip_payload_creation, **kwargs) -> NormalizedResponse` (async)

Make an HTTP request to the endpoint.

```python
async def call(
    self,
    request: dict | BaseModel,
    skip_payload_creation: bool = False,
    **kwargs: Any,
) -> NormalizedResponse: ...
```

**Parameters**:

- `request` (dict or BaseModel): Request payload.
- `skip_payload_creation` (bool): If True, bypass `create_payload()` validation. Caller
  provides ready payload and handles auth.
- `**kwargs`: Additional request parameters.

**Returns**: `NormalizedResponse` with status, data, and raw response.

**Resilience**: Applies circuit breaker (inner) then retry (outer) if configured.

---

#### `stream(request, extra_headers, **kwargs)` (async generator)

Stream responses from the endpoint (SSE).

```python
async def stream(
    self,
    request: dict | BaseModel,
    extra_headers: dict | None = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]: ...
```

**Parameters**:

- `request` (dict or BaseModel): Request payload.
- `extra_headers` (dict or None): Additional headers.
- `**kwargs`: Additional request parameters.

**Yields**: String chunks from SSE stream.

**Note**: Automatically sets `stream=True` in payload.

---

#### `create_payload(request, extra_headers, **kwargs) -> tuple[dict, dict]`

Create validated payload and headers for request.

```python
def create_payload(
    self,
    request: dict | BaseModel,
    extra_headers: dict | None = None,
    **kwargs: Any,
) -> tuple[dict, dict]: ...
```

**Returns**: Tuple of (validated_payload, headers).

**Raises**: `ValueError` if `request_options` not defined or validation fails.

---

## APICalling

> Event class for API calls via Endpoint with token estimation.

Extends `Calling` to execute HTTP requests through Endpoint. Provides token estimation
for rate limiting integration.

### Class Signature

```python
class APICalling(Calling):
    """API call via Endpoint."""

    def __init__(
        self,
        backend: Endpoint,
        payload: dict[str, Any],
        headers: dict = {},
        timeout: float | None = None,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None: ...
```

### Parameters

| Parameter   | Type             | Description                                      |
| ----------- | ---------------- | ------------------------------------------------ |
| `backend`   | `Endpoint`       | Endpoint instance (excluded from serialization). |
| `payload`   | `dict[str, Any]` | Request payload.                                 |
| `headers`   | `dict`           | Request headers (excluded from serialization).   |
| `timeout`   | `float or None`  | Event timeout in seconds (inherited from Event). |
| `streaming` | `bool`           | Enable streaming mode (inherited from Event).    |

### Properties

| Property          | Type                          | Description                                              |
| ----------------- | ----------------------------- | -------------------------------------------------------- |
| `required_tokens` | `int or None`                 | Estimated tokens for rate limiting. None skips tracking. |
| `request`         | `dict`                        | Permission data for Processor rate limiting.             |
| `call_args`       | `dict`                        | Arguments for `backend.call(**self.call_args)`.          |
| `response`        | `NormalizedResponse or Unset` | Response after execution (inherited from Calling).       |

### Token Estimation

`APICalling` estimates tokens for rate limiting using tiktoken:

- **Chat completions**: Counts tokens in `messages` array.
- **Embeddings**: Counts tokens in `input` field.
- **Other**: Returns `None` (no token tracking).

Disable with `requires_tokens=False` in EndpointConfig.

---

## Resilience Patterns

### Circuit Breaker

Fail-fast pattern to prevent cascading failures.

```python
from lionpride.services.utilities.resilience import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,     # Open after 5 failures
    recovery_time=30.0,      # Wait 30s before half-open
    half_open_max_calls=1,   # Allow 1 test call in half-open
    name="openai-circuit",
)

endpoint = Endpoint(
    config={...},
    circuit_breaker=cb,
)
```

**States**:

| State       | Behavior                                                   |
| ----------- | ---------------------------------------------------------- |
| `CLOSED`    | Normal operation, tracking failures.                       |
| `OPEN`      | Rejecting all requests, raising `CircuitBreakerOpenError`. |
| `HALF_OPEN` | Allowing limited test requests.                            |

### Retry with Backoff

Exponential backoff with jitter for transient failures.

```python
from lionpride.services.utilities.resilience import RetryConfig

retry = RetryConfig(
    max_retries=3,           # Retry up to 3 times
    initial_delay=1.0,       # Start with 1s delay
    max_delay=60.0,          # Cap at 60s
    exponential_base=2.0,    # Double delay each retry
    jitter=True,             # Add randomness (0.5-1.0x)
)

endpoint = Endpoint(
    config={...},
    retry_config=retry,
)
```

**Retry Order**: Circuit breaker wraps the call (inner), retry wraps circuit breaker
(outer). Each retry attempt counts against circuit breaker metrics.

---

## Protocol Implementations

Inherits from Element: **Observable**, **Serializable**, **Deserializable**,
**Hashable**.

Serialization includes `circuit_breaker` and `retry_config` as configuration dicts
(state is not preserved).

---

## Usage Patterns

### Basic Usage

```python
from lionpride.services.types import Endpoint, EndpointConfig
from pydantic import BaseModel

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]

config = EndpointConfig(
    provider="openai",
    name="chat",
    base_url="https://api.openai.com/v1",
    endpoint="chat/completions",
    api_key="OPENAI_API_KEY",  # Env var name
    request_options=ChatRequest,
)

endpoint = Endpoint(config=config)

response = await endpoint.call({
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
})
print(response.data)
```

### With iModel (Recommended)

```python
from lionpride import iModel

# Auto-matches OpenAI endpoint
model = iModel(
    provider="openai",
    model="gpt-4o-mini",
    limit_requests=50,
    limit_tokens=100000,
)

calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(calling.response.data["choices"][0]["message"]["content"])
```

### Streaming

```python
endpoint = Endpoint(config=config)

async for chunk in endpoint.stream({
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Tell me a story"}],
}):
    # chunk is SSE line: "data: {...}"
    if chunk.startswith("data: "):
        data = chunk[6:]
        if data != "[DONE]":
            print(json.loads(data)["choices"][0]["delta"].get("content", ""), end="")
```

### With Resilience

```python
from lionpride.services.types import Endpoint
from lionpride.services.utilities.resilience import CircuitBreaker, RetryConfig

endpoint = Endpoint(
    config={
        "provider": "openai",
        "name": "chat",
        "base_url": "https://api.openai.com/v1",
        "endpoint": "chat/completions",
        "api_key": "OPENAI_API_KEY",
    },
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        recovery_time=30.0,
    ),
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
    ),
)

# Automatic retry with circuit breaker protection
response = await endpoint.call({...})
```

### Custom Authentication

```python
# x-api-key header (Anthropic style)
config = EndpointConfig(
    provider="anthropic",
    name="messages",
    base_url="https://api.anthropic.com/v1",
    endpoint="messages",
    auth_type="x-api-key",
    api_key="ANTHROPIC_API_KEY",
    default_headers={"anthropic-version": "2024-01-01"},
)

# No authentication
config = EndpointConfig(
    provider="public",
    name="api",
    base_url="https://api.example.com",
    endpoint="data",
    auth_type="none",
)

# Custom headers with bearer auth
config = EndpointConfig(
    provider="custom",
    name="api",
    auth_type="bearer",
    api_key="API_KEY_ENV",
    default_headers={
        "X-Custom-Header": "value",
        "X-Request-ID": "12345",
    },
)
```

### URL Template Parameters

```python
# Dynamic URL construction
config = EndpointConfig(
    provider="custom",
    name="model-api",
    base_url="https://api.example.com",
    endpoint="models/{model_id}/generate",
    endpoint_params=["model_id"],
    params={"model_id": "gpt-4"},
)
# full_url: "https://api.example.com/models/gpt-4/generate"
```

---

## Common Pitfalls

- **Missing request_options**: `create_payload()` requires `request_options` schema.
  Define a Pydantic model for request validation.

- **API key not found**: If env var pattern matches but env var doesn't exist,
  credential is treated as raw and cleared. Ensure env var is set or use `SecretStr` for
  explicit raw credentials.

- **System env var collision**: Using `HOME`, `PATH`, etc. as `api_key` raises
  `ValueError`. Use a different env var name.

- **Circuit breaker state not serialized**: `CircuitBreaker` serializes configuration
  only. State (failure counts, open/closed) resets on deserialization.

- **Streaming without SSE parsing**: `stream()` yields raw SSE lines. Parse `data:`
  prefix and handle `[DONE]` marker.

---

## Design Rationale

**Secure Credentials**: Environment variable resolution prevents hardcoded secrets in
serialized configs. The private `_api_key` (SecretStr) is never serialized.

**Resilience First**: Built-in circuit breaker and retry patterns handle transient
failures without application-level boilerplate.

**Composable with iModel**: Endpoint provides low-level HTTP, iModel adds rate limiting
and hooks. Use Endpoint directly for custom control, iModel for standard patterns.

**Token Estimation**: APICalling integrates with Processor/Executor for token-based rate
limiting, essential for LLM API cost control.

---

## See Also

- [ServiceBackend](backend.md): Base class for all backends
- [Tool](tool.md): Local callable backend
- [iModel](imodel.md): High-level service interface with rate limiting
- [ServiceRegistry](registry.md): Service management
