# Tool

> Wraps Python callables as service backends with automatic JSON Schema generation for LLM tool calling

## Overview

`Tool` transforms Python functions into LLM-compatible tool definitions with auto-generated JSON Schema from signatures or Pydantic models. Supports sync/async callables and TypeScript rendering for LLMs.

**Use for**: LLM tool calling, ReAct action handlers.
**Skip for**: HTTP APIs (use Endpoint), serialization scenarios (callables not serializable).

## Class Signature

```python
from lionpride import Tool

class Tool(ServiceBackend):
    """Wraps Python callables as service backends for LLM tool calling."""

    # Constructor signature
    def __init__(
        self,
        *,
        func_callable: Callable[..., Any],
        config: ToolConfig | dict | None = None,
        tool_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func_callable` | `Callable` | Required. Python function with `__name__` attribute. |
| `config` | `ToolConfig or dict` | Optional. Auto-generated from function if not provided. |
| `tool_schema` | `dict` | Optional. Priority: `request_options` > `tool_schema` > auto-generated. |

## Attributes

| Attribute | Type | Frozen | Description |
|-----------|------|--------|-------------|
| `func_callable` | `Callable[..., Any]` | Yes | The wrapped Python callable |
| `tool_schema` | `dict[str, Any]` | No | JSON Schema for parameters (auto-generated or provided) |
| `config` | `ToolConfig` | No | Service configuration (name, provider, request_options) |

### Inherited from ServiceBackend

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Unique identifier (from Element) |
| `created_at` | `datetime` | Creation timestamp (from Element) |
| `metadata` | `dict[str, Any]` | Arbitrary metadata (from Element) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Tool name from config (defaults to function name) |
| `provider` | `str` | Provider identifier (always `"tool"` for Tool) |
| `function_name` | `str` | Original function `__name__` |
| `rendered` | `str` | TypeScript-formatted schema for LLM consumption |
| `required_fields` | `frozenset[str]` | Set of required parameter names |
| `request_options` | `type[BaseModel] \| None` | Pydantic model for validation (from config) |
| `event_type` | `type[ToolCalling]` | Event class for tool execution |

## Methods

### `call(arguments) -> NormalizedResponse` (async)

Execute the wrapped callable. Auto-detects sync/async; sync functions use `run_sync()`.

```python
result = await tool.call({"name": "World"})
result.data  # Return value
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `function_name` | `str` | Original `__name__` of callable |
| `rendered` | `str` | TypeScript-formatted schema for LLMs |
| `required_fields` | `frozenset[str]` | Parameter names without defaults |
| `request_options` | `type[BaseModel] or None` | Pydantic validation model |

### Serialization

**Not supported** - callables cannot be serialized. Use `Endpoint` for persistence.

## Protocol Implementations

Inherits **Observable**, **Hashable** from Element. **Serializable/Deserializable** not implemented (callables).

## Usage Patterns

### Basic Usage

```python
from lionpride import Tool

def search(query: str, limit: int = 10) -> list[str]:
    return [f"Result {i} for {query}" for i in range(limit)]

tool = Tool(func_callable=search)
print(tool.required_fields)  # frozenset({'query'})

result = await tool.call({"query": "python", "limit": 5})
```

### With Pydantic Validation

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)

tool = Tool(
    func_callable=search,
    config={"name": "search", "request_options": SearchParams}
)
print(tool.rendered)  # TypeScript-formatted schema
```

### With ServiceRegistry

```python
from lionpride import Tool, ServiceRegistry

registry = ServiceRegistry()
registry.register(Tool(func_callable=search))

tool = registry.get("search")
result = await tool.call({"query": "example"})
```

## Common Pitfalls

- **Serialization fails**: Tool raises `NotImplementedError`. Use `Endpoint` for persistence.

- **Schema priority**: `request_options` always overrides `tool_schema`.

- **Prefer named functions**: Lambdas work but named functions provide clearer tool identification.

## Design Rationale

**Auto-generated Schema**: Extracts from signatures/Pydantic to prevent drift between code and schema.

**TypeScript Rendering**: LLMs perform better with TypeScript-style types than verbose JSON Schema.

**Not Serializable**: Callables cannot serialize reliably; explicit error guides users to Endpoint.

## See Also

- [ServiceRegistry](registry.md), [iModel](imodel.md): Service management
- [Tool Calling Guide](../../user_guide/tool_calling.md): Usage patterns

## Example: Tool in ReAct Pattern

```python
from lionpride import Tool, Session, iModel
from lionpride.operations.operate import operate, OperateParams, GenerateParams

def search(query: str) -> list[str]:
    return [f"Result for: {query}"]

def calculate(expr: str) -> str:
    return str(eval(expr))

model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)

session.services.register(Tool(func_callable=search))
session.services.register(Tool(func_callable=calculate))

branch = session.create_branch(resources={"search", "calculate"})

params = OperateParams(
    generate=GenerateParams(instruction="Search for Python then calculate 2+2"),
    actions=True,
)
result = await session.conduct("operate", branch, params=params)
```
