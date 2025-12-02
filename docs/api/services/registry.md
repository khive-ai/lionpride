# ServiceRegistry

> Unified service boundary managing iModel instances with O(1) name-based lookup

---

## Overview

`ServiceRegistry` provides Pile-backed storage with O(1) name lookup, tag-based filtering, tool schema extraction, and MCP integration. Central service management for Sessions.

**Use for**: Multiple LLM backends, tool registration, MCP servers.
**Skip for**: Single-model workflows (pass iModel directly).

## Class Signature

```python
from lionpride.services import ServiceRegistry

class ServiceRegistry:
    """Service registry managing iModel instances with O(1) name-based lookup."""

    # Constructor signature
    def __init__(self) -> None: ...
```

## Parameters

**None** - ServiceRegistry is initialized empty. Services are added via `register()`.

## Attributes

| Attribute | Type | Mutable | Description |
|-----------|------|---------|-------------|
| `_pile` | `Pile[iModel]` | Yes (internal) | Internal storage for iModel instances |
| `_name_index` | `dict[str, UUID]` | Yes (internal) | Name to UUID mapping for O(1) lookup |

**Note**: Internal attributes are prefixed with `_` and should not be accessed directly. Use the public methods for all operations.

## Methods

### Core Operations

#### `register(model, update=False) -> UUID`

Register iModel by name. Raises `ValueError` if name exists and `update=False`. O(1).

```python
registry.register(iModel(provider="openai", model="gpt-4o-mini"))
registry.register(new_model, update=True)  # Replace existing
```

#### `unregister(name) -> iModel`

Remove and return service by name. Raises `KeyError` if not found. O(n).

#### `get(name, default=...) -> iModel`

Get by name, UUID, or iModel instance. Raises `KeyError` if not found and no default. O(1).

```python
model = registry.get("gpt-4o-mini")
model = registry.get("missing", default=None)
```

#### `has(name) -> bool`

Check if service exists. O(1).

---

### Enumeration Methods

#### `list_names() -> list[str]`

All registered service names. O(n).

#### `list_by_tag(tag) -> Pile[iModel]`

Filter services by tag (e.g., "llm", "tool"). Returns new Pile. O(n).

#### `count() -> int` / `clear() -> None`

Count services or remove all. O(1).

---

### Tool Schema Methods

#### `get_tool_schemas(tool_names=None) -> list[dict]`

Get JSON schemas for LLM tool calling. Returns `{"type": "function", "function": {...}}` format.

```python
schemas = registry.get_tool_schemas()  # All tools
schemas = registry.get_tool_schemas(tool_names=["multiply"])  # Specific
```

Raises `KeyError` if tool not found, `TypeError` if not Tool-backed.

---

### MCP Integration Methods

#### `register_mcp_server(server_config, tool_names=None, update=False) -> list[str]` (async)

Register tools from MCP server. Returns list of registered tool names.

```python
tools = await registry.register_mcp_server({
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
    "env": {"ALLOWED_DIRS": "/tmp"},
})
# Returns: ['read_file', 'write_file', 'list_directory', ...]
```

#### `load_mcp_config(config_path, server_names=None, update=False) -> dict[str, list[str]]` (async)

Load multiple MCP servers from `.mcp.json`. Returns `{server_name: [tool_names]}`.

---

### Collection Protocol Methods

Supports `len(registry)`, `"name" in registry`, and `repr(registry)`.

## Protocol Implementations

ServiceRegistry is a **pure Python class** (not an Element). Supports Python collection protocols (`__len__`, `__contains__`, `__repr__`). Managed iModel instances ARE Elements with full protocol support.

## Design Rationale

**Dual Index**: Pile[iModel] for type-safe storage + dict[str, UUID] for O(1) name lookup.

**Name Uniqueness**: Duplicates raise ValueError; `update=True` for explicit replacement.

**MCP First-Class**: Native async methods match MCP's subprocess model.

---

## Usage Patterns

### Basic Usage

```python
from lionpride.services import ServiceRegistry, iModel

registry = ServiceRegistry()
registry.register(iModel(provider="openai", model="gpt-4o-mini"))
registry.register(iModel(provider="anthropic", model="claude-3-5-sonnet"))

model = registry.get("gpt-4o-mini")
print(registry.list_names())
```

### Tool Registration

```python
from lionpride.services import ServiceRegistry, Tool, iModel

def calculate(expression: str) -> float:
    return eval(expression)

registry = ServiceRegistry()
registry.register(iModel(backend=Tool(func_callable=calculate)))
schemas = registry.get_tool_schemas()  # For LLM tool calling
```

### MCP Integration

```python
registry = ServiceRegistry()
await registry.register_mcp_server({
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
}, tool_names=["read_file", "write_file"])
```

---

## Common Pitfalls

- **Duplicate registration**: Use `update=True` or check `has()` first.

- **Non-tool schemas**: `get_tool_schemas()` only works for Tool-backed services. Use `list_by_tag("tool")` to filter.

- **Missing await**: MCP methods are async. Always `await registry.register_mcp_server(...)`.

---

## See Also

- [iModel](imodel.md), [Tool](tool.md), [Endpoint](endpoint.md): Service types
- [Session](../session/session.md): Using ServiceRegistry in sessions

## Example: Complete Setup

```python
from lionpride import Session
from lionpride.services import ServiceRegistry, iModel, Tool

registry = ServiceRegistry()

# LLM models
registry.register(iModel(provider="openai", model="gpt-4o-mini"))
registry.register(iModel(provider="anthropic", model="claude-3-5-sonnet"))

# Function tools
def search(query: str) -> str:
    return f"Results for: {query}"

registry.register(iModel(backend=Tool(func_callable=search)))

# Get tool schemas and create session
schemas = registry.get_tool_schemas()
session = Session(services=registry, default_generate_model="gpt-4o-mini")
```
