# Providers and Backends

> Service backends for LLM APIs, CLI tools, and Python callables

## Overview

lionpride services support multiple backend types through the `ServiceBackend` protocol. This enables polymorphic service access: the same `iModel` interface works with HTTP APIs (OpenAI, Anthropic), CLI tools (Claude Code, Gemini Code), or Python callables (functions, tools).

**Backend Types:**

- **`Endpoint`**: HTTP API backends with request/response handling, circuit breaking, and retry
- **`Tool`**: Python callable backends with schema validation and sync/async detection
- **Custom**: Implement `ServiceBackend` protocol for any service type

**Provider Endpoints:**

- **`AnthropicMessagesEndpoint`**: Anthropic Messages API (Claude models)
- **`OAIChatEndpoint`**: OpenAI Chat Completions API (OpenAI, Groq, OpenRouter, NVIDIA NIM)
- **`GeminiCodeEndpoint`**: Google Gemini CLI (local agent execution)
- **`ClaudeCodeEndpoint`**: Claude Code CLI (local agent execution)

**Auto-Matching:**

The `match_endpoint()` function maps provider names to appropriate endpoints:

```python
from lionpride.services.providers import match_endpoint

endpoint = match_endpoint("anthropic", "messages", api_key="...")
endpoint = match_endpoint("openai", "chat/completions")
endpoint = match_endpoint("claude_code", "query_cli")
```

## Module Exports

```python
from lionpride.services.providers import (
    # Provider matching
    match_endpoint,

    # Endpoint implementations
    AnthropicMessagesEndpoint,
    OAIChatEndpoint,
    GeminiCodeEndpoint,
    ClaudeCodeEndpoint,

    # Config factories
    create_anthropic_config,
    create_oai_chat,
    create_claude_code_config,
    create_gemini_code_config,
)
```

## ServiceBackend Protocol

Abstract base class for all service backends.

### Class Signature

```python
class ServiceBackend(Element):
    """Base class for all service backends (Tool, Endpoint, etc.)."""

    config: ServiceConfig = Field(...)

    @property
    @abstractmethod
    def event_type(self) -> type[Calling]:
        """Return Calling type for this backend (e.g., ToolCalling, APICalling)."""
        ...

    @abstractmethod
    async def call(self, *args, **kw) -> NormalizedResponse:
        """Execute service call and return normalized response."""
        ...

    async def stream(self, *args, **kw):
        """Stream responses (not supported by default)."""
        raise NotImplementedError("This backend does not support streaming calls.")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `provider` | `str` | Provider name from config |
| `name` | `str` | Service name from config |
| `version` | `str \| None` | Service version from config |
| `tags` | `set[str]` | Service tags from config |
| `request_options` | `type[BaseModel] \| None` | Request schema (Pydantic model type) |
| `event_type` | `type[Calling]` | Calling type for this backend (abstract) |

### Methods

#### `normalize_response()`

Normalize raw response into `NormalizedResponse`.

**Signature:**

```python
def normalize_response(self, raw_response: Any) -> NormalizedResponse: ...
```

**Parameters:**

- `raw_response` (Any): Raw response from service call

**Returns**: `NormalizedResponse` with `status`, `data`, `raw_response`, `metadata`

Default implementation wraps response as-is. Subclasses override to extract specific fields.

## Endpoint

HTTP API backend with request/response handling, circuit breaking, and retry.

### Class Signature

```python
class Endpoint(ServiceBackend):
    """HTTP API backend with resilience patterns."""

    circuit_breaker: CircuitBreaker | None = None
    retry_config: RetryConfig | None = None
    config: EndpointConfig

    def __init__(
        self,
        config: dict | EndpointConfig,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
        **kwargs,
    ): ...
```

### EndpointConfig

Configuration for Endpoint backends.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Provider name (e.g., "anthropic", "openai") |
| `name` | `str` | Service name (e.g., "anthropic_messages") |
| `base_url` | `str \| None` | Base API URL (e.g., "<https://api.anthropic.com/v1>") |
| `endpoint` | `str` | Endpoint path (e.g., "messages", "chat/completions") |
| `endpoint_params` | `list[str] \| None` | URL template params (e.g., ["model_id"]) |
| `method` | `str` | HTTP method (default: "POST") |
| `params` | `dict[str, str]` | URL params for template formatting |
| `content_type` | `str \| None` | Content-Type header (default: "application/json") |
| `auth_type` | `AUTH_TYPES` | Auth type ("bearer", "x-api-key", etc.) |
| `default_headers` | `dict` | Default headers (e.g., {"anthropic-version": "2023-06-01"}) |
| `api_key` | `str \| None` | API key env var name (NOT raw credential) |
| `api_key_is_env` | `bool` | True if api_key is env var name (set automatically) |
| `openai_compatible` | `bool` | Whether endpoint is OpenAI-compatible |
| `requires_tokens` | `bool` | Whether token usage tracking required |
| `client_kwargs` | `dict` | Additional kwargs for httpx.AsyncClient |
| `request_options` | `type[BaseModel] \| None` | Request schema (Pydantic model) |
| `timeout` | `int` | Request timeout in seconds (1-3600) |
| `max_retries` | `int` | Max retries (0-10) |
| `version` | `str \| None` | Service version |
| `tags` | `list[str]` | Service tags |
| `kwargs` | `dict` | Additional config (auto-extracted from **init**) |

**API Key Security Model:**

- **Pattern Matching**: If `api_key` matches `^[A-Z][A-Z0-9_]*$`, try env var resolution
- **Env Var Resolution**: If env var exists, keep `api_key` name for serialization (`api_key_is_env=True`)
- **Raw Credential**: If no env var match, treat as raw credential and CLEAR `api_key` field (`api_key_is_env=False`)
- **SecretStr Support**: Pass `SecretStr("raw_credential")` to bypass pattern matching
- **Deserialization**: Verify env var still exists (`api_key_is_env=True`) or fail

### Methods

#### `create_payload()`

Create request payload and headers for API call.

**Signature:**

```python
def create_payload(
    self,
    request: dict | BaseModel,
    extra_headers: dict | None = None,
    **kwargs,
) -> tuple[dict, dict]: ...
```

**Parameters:**

- `request` (dict | BaseModel): Request parameters or Pydantic model
- `extra_headers` (dict | None): Additional headers
- `**kwargs`: Additional kwargs merged into request

**Returns**: `(payload_dict, headers_dict)`

**Process**:

1. Create headers via `HeaderFactory` (auth + content-type + defaults)
2. Convert request to dict if BaseModel
3. Merge config kwargs → request → call kwargs
4. Validate via `request_options` schema (required)
5. Filter to valid fields only

#### `call()`

Execute HTTP request with resilience patterns.

**Signature:**

```python
async def call(
    self,
    request: dict | BaseModel,
    skip_payload_creation: bool = False,
    **kwargs,
) -> NormalizedResponse: ...
```

**Parameters:**

- `request` (dict | BaseModel): Request parameters
- `skip_payload_creation` (bool): Skip `create_payload()` and use request as-is
- `**kwargs`: Additional kwargs

**Returns**: `NormalizedResponse` from endpoint

**Resilience Pattern**:

- Circuit breaker wraps `_call()` (inner)
- Retry wraps circuit breaker (outer)
- Each retry attempt counts against circuit breaker metrics

#### `stream()`

Stream responses from endpoint.

**Signature:**

```python
async def stream(
    self,
    request: dict | BaseModel,
    extra_headers: dict | None = None,
    **kwargs,
) -> AsyncIterator[str]: ...
```

**Yields**: String lines from server-sent events (SSE)

#### `normalize_response()`

Normalize raw response to `NormalizedResponse`.

**Signature:**

```python
def normalize_response(self, raw_response: Any) -> NormalizedResponse: ...
```

Default wraps response as-is. Subclasses extract provider-specific fields.

## AnthropicMessagesEndpoint

Anthropic Messages API endpoint for Claude models.

### Usage

```python
from lionpride.services.providers.anthropic_messages import AnthropicMessagesEndpoint

endpoint = AnthropicMessagesEndpoint(api_key="ANTHROPIC_API_KEY")

response = await endpoint.call({
    "model": "claude-sonnet-4-5-20250929",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024
})

print(response.data)  # Extracted text
print(response.metadata["usage"])  # Token usage
```

### Configuration

```python
from lionpride.services.providers.anthropic_messages import create_anthropic_config

config = create_anthropic_config(
    name="anthropic_messages",
    api_key="ANTHROPIC_API_KEY",
    base_url="https://api.anthropic.com/v1",
    endpoint="messages",
    anthropic_version="2023-06-01",
)

endpoint = AnthropicMessagesEndpoint(config=config)
```

### Response Normalization

Extracts:

- **Text**: Combined from content blocks (type="text")
- **Thinking**: Extended thinking blocks (type="thinking") in metadata["thinking"]
- **Tool Uses**: Tool use blocks in metadata["tool_uses"]
- **Usage**: Token usage in metadata["usage"]
- **Metadata**: `model`, `stop_reason`, `stop_sequence`, `id`

```python
response = await endpoint.call({
    "model": "claude-sonnet-4-5-20250929",
    "messages": [{"role": "user", "content": "Explain photosynthesis"}],
    "max_tokens": 2048,
    "thinking": {"type": "enabled", "budget_tokens": 1000}
})

print(response.data)  # Main response text
print(response.metadata["thinking"])  # Extended thinking process
print(response.metadata["usage"])  # {"input_tokens": 12, "output_tokens": 456}
```

## OAIChatEndpoint

OpenAI Chat Completions API endpoint (supports OpenAI, Groq, OpenRouter, NVIDIA NIM).

### Usage

```python
from lionpride.services.providers.oai_chat import OAIChatEndpoint

# OpenAI
endpoint = OAIChatEndpoint(provider="openai", api_key="OPENAI_API_KEY")

# Groq (fast inference)
endpoint = OAIChatEndpoint(provider="groq", api_key="GROQ_API_KEY")

# OpenRouter (multi-provider gateway)
endpoint = OAIChatEndpoint(provider="openrouter", api_key="OPENROUTER_API_KEY")

response = await endpoint.call({
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
})
```

### Configuration

```python
from lionpride.services.providers.oai_chat import create_oai_chat

config = create_oai_chat(
    provider="openai",
    name="openai_chat",
    base_url=None,  # Auto-detected from provider
    api_key="OPENAI_API_KEY",
    endpoint="chat/completions",
)

endpoint = OAIChatEndpoint(config=config)
```

### Supported Providers

| Provider | Base URL | API Key Env Var |
|----------|----------|-----------------|
| openai | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| openrouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| nvidia_nim | `https://integrate.api.nvidia.com/v1` | `NVIDIA_NIM_API_KEY` |

### Response Normalization

Extracts:

- **Text**: `choices[0].message.content`
- **Tool Calls**: `choices[0].message.tool_calls` in metadata
- **Metadata**: `model`, `usage`, `id`, `finish_reason`

```python
response = await endpoint.call({
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
})

print(response.data)  # "Hello! How can I help you?"
print(response.metadata["usage"])  # {"prompt_tokens": 8, "completion_tokens": 9, ...}
print(response.metadata["finish_reason"])  # "stop"
```

## ClaudeCodeEndpoint

Claude Code CLI endpoint for local AI agent execution.

### Usage

```python
from lionpride.services.providers.claude_code import ClaudeCodeEndpoint

endpoint = ClaudeCodeEndpoint()

# Messages format
response = await endpoint.call({
    "messages": [{"role": "user", "content": "List files in current directory"}]
})

# With session resumption
response = await endpoint.call({
    "messages": [{"role": "user", "content": "Now count the files"}],
    "resume": "session_abc123",  # Resume previous session
})
```

### Configuration

```python
from lionpride.services.providers.claude_code import create_claude_code_config

config = create_claude_code_config(
    name="claude_code_cli",
)

endpoint = ClaudeCodeEndpoint(config=config)
```

### Features

- **Auto-Resume**: iModel auto-injects `session_id` from previous calls
- **Auto-Finish**: `auto_finish=True` sends final prompt to get result message
- **Summary**: `cli_include_summary=True` populates tool use/result summary
- **Session State**: Tracks turns, duration, cost, tool uses

### Response Normalization

Extracts:

- **Text**: Combined result from all turns
- **Session ID**: `metadata["session_id"]` (auto-injected in next call)
- **Tool Uses**: `metadata["tool_uses"]` - list of tool calls
- **Tool Results**: `metadata["tool_results"]` - list of tool outputs
- **Thinking Log**: `metadata["thinking_log"]` - internal reasoning
- **Summary**: `metadata["summary"]` - consolidated tool use/result summary
- **Usage**: Token usage, cost, duration

```python
response = await endpoint.call({
    "messages": [{"role": "user", "content": "Analyze this codebase"}],
    "cli_include_summary": True,
    "auto_finish": True,
})

print(response.data)  # Final result text
print(response.metadata["session_id"])  # "abc123"
print(response.metadata["num_turns"])  # 3
print(response.metadata["total_cost_usd"])  # 0.0042
print(response.metadata["summary"])  # "Tool uses: 5\nTool results: 5\n..."
```

## GeminiCodeEndpoint

Google Gemini CLI endpoint for local AI agent execution.

### Usage

```python
from lionpride.services.providers.gemini import GeminiCodeEndpoint

endpoint = GeminiCodeEndpoint(model="gemini-2.5-pro")

# Messages format
response = await endpoint.call({
    "messages": [{"role": "user", "content": "List files"}]
})

# Prompt format with yolo mode
response = await endpoint.call({
    "prompt": "Analyze this codebase",
    "yolo": True,  # Auto-approve all actions
})
```

### Configuration

```python
from lionpride.services.providers.gemini import create_gemini_code_config

config = create_gemini_code_config(
    name="gemini_code_cli",
    model="gemini-2.5-pro",  # or "gemini-2.5-flash", "gemini-3-pro"
)

endpoint = GeminiCodeEndpoint(config=config)
```

### Features

- **Prompt or Messages**: Accepts either `prompt` (string) or `messages` (list)
- **YOLO Mode**: `yolo=True` auto-approves all tool executions
- **Summary**: `cli_include_summary=True` populates tool use/result summary
- **Session State**: Tracks turns, duration, cost, tool uses

### Response Normalization

Extracts:

- **Text**: Combined result from all chunks
- **Tool Uses**: `metadata["tool_uses"]` - list of tool calls
- **Tool Results**: `metadata["tool_results"]` - list of tool outputs
- **Summary**: `metadata["summary"]` - consolidated tool use/result summary
- **Usage**: Token usage, cost, duration
- **Session ID**: `metadata["session_id"]` (if available)

```python
response = await endpoint.call({
    "prompt": "Analyze this codebase",
    "yolo": True,
    "cli_include_summary": True,
})

print(response.data)  # Final result text
print(response.metadata["model"])  # "gemini-2.5-pro"
print(response.metadata["num_turns"])  # 2
print(response.metadata["total_cost_usd"])  # 0.0031
print(response.metadata["summary"])  # "Tool uses: 3\nTool results: 3\n..."
```

## Tool

Python callable backend with schema validation and sync/async detection.

### Class Signature

```python
class Tool(ServiceBackend):
    """Python callable backend with schema validation."""

    config: ToolConfig
    func_callable: Callable[..., Any] = Field(..., frozen=True, exclude=True)
    tool_schema: dict[str, Any] | None = None
```

### Usage

```python
from lionpride.services import Tool, iModel
from pydantic import BaseModel, Field

# Define schema
class WeatherRequest(BaseModel):
    location: str = Field(..., description="City name")
    units: str = Field("celsius", description="Temperature units")

def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather for a location."""
    return {"temp": 22, "conditions": "sunny", "location": location}

# Create tool
tool = Tool(
    func_callable=get_weather,
    config={"request_options": WeatherRequest}
)

# Execute via iModel
model = iModel(backend=tool)
calling = await model.invoke(location="San Francisco")
print(calling.response.data)  # {"temp": 22, "conditions": "sunny", ...}
```

### Schema Priority

Tool schema resolution order:

1. **`request_options`** (canonical source): Pydantic model from config
2. **`tool_schema`**: Explicit JSON Schema dict
3. **Auto-generated**: From function signature (fallback)

```python
# Priority 1: request_options (recommended)
tool = Tool(
    func_callable=my_func,
    config={"request_options": MyRequestModel}
)

# Priority 2: tool_schema
tool = Tool(
    func_callable=my_func,
    tool_schema={"type": "object", "properties": {...}}
)

# Priority 3: Auto-generate from signature
tool = Tool(func_callable=my_func)  # Inspects signature
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `function_name` | `str` | Callable's `__name__` |
| `rendered` | `str` | TypeScript schema for LLM consumption |
| `required_fields` | `frozenset[str]` | Required parameter names |
| `event_type` | `type[ToolCalling]` | Returns `ToolCalling` |

### Methods

#### `call()`

Execute callable with sync/async detection.

**Signature:**

```python
async def call(self, arguments: dict[str, Any]) -> NormalizedResponse: ...
```

**Parameters:**

- `arguments` (dict): Validated parameters for callable

**Returns**: `NormalizedResponse` with callable result in `data`

**Process**:

1. Detect if callable is coroutine function
2. If async: `await func(**arguments)`
3. If sync: `await run_sync(lambda: func(**arguments))`
4. Wrap result in `NormalizedResponse`

### Serialization

**Not Supported**: Tool backends contain callables which are not serializable.

```python
tool = Tool(func_callable=my_function)
model = iModel(backend=tool)
config = model.to_dict()  # backend=None in dict
restored = iModel.from_dict(config)  # ValueError: backend is required
```

**Solution**: Use Endpoint backends for persistence scenarios.

## match_endpoint()

Match provider and endpoint to appropriate Endpoint class.

### Signature

```python
def match_endpoint(
    provider: str,
    endpoint: str,
    **kwargs,
) -> Endpoint: ...
```

### Parameters

- `provider` (str): Provider name ("anthropic", "openai", "groq", "openrouter", "nvidia_nim", "claude_code", "gemini_code")
- `endpoint` (str): Endpoint path (e.g., "messages", "chat/completions", "query_cli")
- `**kwargs`: Additional kwargs passed to Endpoint constructor

### Returns

`Endpoint` instance configured for the provider

### Matching Rules

```python
# Anthropic
if provider == "anthropic" and ("messages" in endpoint or "chat" in endpoint):
    return AnthropicMessagesEndpoint(None, **kwargs)

# Claude Code
if provider == "claude_code":
    return ClaudeCodeEndpoint(None, **kwargs)

# OpenAI-compatible
if provider in ("openai", "groq", "openrouter", "nvidia_nim") and "chat" in endpoint:
    return OAIChatEndpoint(None, provider=provider, endpoint=endpoint, **kwargs)

# Unknown provider fallback (with warning)
return OAIChatEndpoint(None, **kwargs)
```

### Examples

```python
from lionpride.services.providers import match_endpoint

# Anthropic
endpoint = match_endpoint("anthropic", "messages", api_key="ANTHROPIC_API_KEY")

# OpenAI
endpoint = match_endpoint("openai", "chat/completions", api_key="OPENAI_API_KEY")

# Groq
endpoint = match_endpoint("groq", "chat/completions", api_key="GROQ_API_KEY")

# Claude Code
endpoint = match_endpoint("claude_code", "query_cli")

# Unknown provider (warning + fallback)
endpoint = match_endpoint("custom_provider", "chat/completions", api_key="...")
# UserWarning: Unknown provider 'custom_provider', falling back to OpenAI-compatible endpoint
```

## Common Patterns

### Pattern 1: Multi-Provider Registry

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Register multiple providers
providers = ["anthropic", "openai", "groq", "claude_code"]
for provider in providers:
    model = iModel(
        provider=provider,
        api_key=f"{provider.upper()}_API_KEY" if provider != "claude_code" else None,
        limit_requests=50,
    )
    registry.register(model)

# Use based on requirements
calling = await registry.get("anthropic_messages").invoke(...)
```

### Pattern 2: Custom Endpoint Configuration

```python
from lionpride.services.providers.anthropic_messages import AnthropicMessagesEndpoint
from lionpride.services.utilities.resilience import CircuitBreaker, RetryConfig

endpoint = AnthropicMessagesEndpoint(
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

from lionpride.services import iModel
model = iModel(backend=endpoint)
```

### Pattern 3: Tool with Schema Validation

```python
from lionpride.services import Tool, iModel
from pydantic import BaseModel, Field

class CalculateRequest(BaseModel):
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")

def calculate(operation: str, a: float, b: float) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return ops[operation]

tool = Tool(func_callable=calculate, config={"request_options": CalculateRequest})
model = iModel(backend=tool)

calling = await model.invoke(operation="add", a=5.0, b=3.0)
print(calling.response.data)  # 8.0
```

### Pattern 4: Session Continuation (Claude Code)

```python
from lionpride.services import iModel

model = iModel(provider="claude_code")

# First call
calling1 = await model.invoke(
    messages=[{"role": "user", "content": "List files"}]
)
# Session ID stored in model.provider_metadata["session_id"]

# Second call (auto-resumes)
calling2 = await model.invoke(
    messages=[{"role": "user", "content": "Now count them"}]
)
# Uses same session automatically
```

## Common Pitfalls

### Issue: API key pattern not recognized

**Problem**: Raw credential matches env var pattern but doesn't exist in environment.

```python
# ❌ WRONG: API key looks like env var but doesn't exist
endpoint = AnthropicMessagesEndpoint(api_key="MY_SECRET_KEY")
# Tries to resolve MY_SECRET_KEY from env, fails, treats as raw credential
# api_key field CLEARED on serialization (security)
```

**Solution**: Use `SecretStr` for raw credentials or ensure env var exists.

```python
# ✅ CORRECT: SecretStr for raw credentials
from pydantic import SecretStr
endpoint = AnthropicMessagesEndpoint(api_key=SecretStr("sk-ant-..."))

# ✅ CORRECT: Use actual env var
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
endpoint = AnthropicMessagesEndpoint(api_key="ANTHROPIC_API_KEY")
```

### Issue: Tool serialization fails

**Problem**: Tool backends cannot serialize callables.

```python
# ❌ WRONG
tool = Tool(func_callable=my_function)
config = tool.to_dict()  # NotImplementedError
```

**Solution**: Don't serialize Tool backends.

```python
# ✅ CORRECT: Create fresh Tool from callable
tool = Tool(func_callable=my_function)
# Don't serialize, keep in memory or recreate on demand
```

### Issue: Missing request_options schema

**Problem**: Endpoint created without `request_options`, validation fails.

```python
# ❌ WRONG: Custom endpoint without schema
from lionpride.services.types.endpoint import Endpoint, EndpointConfig
config = EndpointConfig(
    provider="custom",
    name="custom_api",
    base_url="https://api.custom.com",
    endpoint="query",
)
endpoint = Endpoint(config=config)
response = await endpoint.call({"param": "value"})  # ValueError: request_options required
```

**Solution**: Always define `request_options` schema.

```python
# ✅ CORRECT: Define request schema
from pydantic import BaseModel

class CustomRequest(BaseModel):
    param: str

config = EndpointConfig(
    provider="custom",
    name="custom_api",
    base_url="https://api.custom.com",
    endpoint="query",
    request_options=CustomRequest,
)
endpoint = Endpoint(config=config)
```

## See Also

- **Core**:
  - [`iModel`](imodel.md): Unified service interface
  - [`ServiceRegistry`](registry.md): Service management
- **Resilience**:
  - [`CircuitBreaker`](resilience.md#circuitbreaker): Fail-fast pattern
  - [`RetryConfig`](resilience.md#retryconfig): Retry with backoff
  - [`TokenBucket`](rate_limiting.md#tokenbucket): Rate limiting
- **Types**:
  - [NormalizedResponse](../types/base.md): Standardized response format
  - [Element](../base/element.md): UUID-based identity
