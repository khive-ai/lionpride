# ServiceRegistry

> Name-indexed service discovery and management for iModel instances

## Overview

`ServiceRegistry` provides centralized service discovery and management for `iModel` instances. It maintains a name-indexed collection of models using `Pile[iModel]` for O(1) UUID lookups and a separate name index for human-readable access.

**Core Features:**

- **Name-Indexed Storage**: Register models by name for easy lookup
- **UUID-Based Identity**: Each model has unique UUID (inherited from Element)
- **Tag-Based Filtering**: List services by tags (e.g., "llm", "anthropic")
- **Update Support**: Update existing registrations or enforce uniqueness
- **MCP Integration**: Load tools from MCP servers directly into registry

**Key Use Cases:**

- Multi-provider LLM orchestration (OpenAI, Anthropic, Groq, etc.)
- Tool-based agent systems with centralized tool registry
- Service discovery in microservices architectures
- Configuration-driven model selection

**When to Use:**

- Building applications with multiple LLM providers
- Implementing agent systems with dynamic tool discovery
- Managing service configurations centrally
- Hot-swapping services without code changes

## Class Signature

```python
class ServiceRegistry:
    """Service discovery and management for iModel instances."""

    def __init__(self): ...

    def register(self, model: iModel, update: bool = False) -> UUID: ...

    def unregister(self, name: str) -> iModel: ...

    def get(self, name: str | UUID | iModel) -> iModel: ...

    def has(self, name: str | UUID | iModel) -> bool: ...

    def list_names(self) -> list[str]: ...

    def list_by_tag(self, tag: str) -> list[iModel]: ...

    def clear(self) -> None: ...

    def count(self) -> int: ...

    async def register_mcp_server(
        self,
        server_config: dict,
        tool_names: list[str] | None = None,
        request_options: dict[str, type] | None = None,
        update: bool = False,
    ) -> list[str]: ...

    async def load_mcp_config(
        self,
        config_path: str,
        server_names: list[str] | None = None,
        update: bool = False,
    ) -> dict[str, list[str]]: ...
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_pile` | `Pile[iModel]` | Internal storage for models (UUID-indexed) |
| `_name_index` | `dict[str, UUID]` | Name → UUID mapping for fast lookup |

## Methods

### Core Operations

#### `__init__()`

Initialize empty registry with Pile storage and name index.

**Signature:**

```python
def __init__(self): ...
```

**Example:**

```python
from lionpride.services import ServiceRegistry

registry = ServiceRegistry()
```

#### `register()`

Register model by name.

**Signature:**

```python
def register(self, model: iModel, update: bool = False) -> UUID: ...
```

**Parameters:**

- `model` (iModel): Model instance to register
- `update` (bool): If True, update existing registration. If False, raise error on duplicate name (default: False)

**Returns**: `UUID` - Model UUID

**Raises**:

- `ValueError`: If `model.name` already registered and `update=False`

**Examples:**

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Register model
model = iModel(provider="anthropic", api_key="ANTHROPIC_API_KEY")
uid = registry.register(model)

# Duplicate name (error)
model2 = iModel(provider="anthropic", api_key="ANTHROPIC_API_KEY")
registry.register(model2)  # ValueError: already registered

# Update existing
model3 = iModel(provider="anthropic", api_key="NEW_KEY")
registry.register(model3, update=True)  # Updates registration
```

#### `unregister()`

Remove and return service by name.

**Signature:**

```python
def unregister(self, name: str) -> iModel: ...
```

**Parameters:**

- `name` (str): Service name

**Returns**: `iModel` - Removed model instance

**Raises**:

- `KeyError`: If service not found

**Example:**

```python
removed_model = registry.unregister("anthropic_messages")
```

#### `get()`

Get service by name, UUID, or iModel instance.

**Signature:**

```python
def get(self, name: str | UUID | iModel) -> iModel: ...
```

**Parameters:**

- `name` (str | UUID | iModel): Service identifier (name string, UUID, or iModel instance)

**Returns**: `iModel` - Model instance

**Raises**:

- `KeyError`: If service not found

**Examples:**

```python
# Get by name
model = registry.get("anthropic_messages")

# Get by UUID
from uuid import UUID
model = registry.get(UUID("..."))

# Get by iModel (returns same instance)
model = registry.get(existing_model)
```

#### `has()` / `__contains__()`

Check if service is registered.

**Signature:**

```python
def has(self, name: str | UUID | iModel) -> bool: ...
def __contains__(self, name: str | UUID | iModel) -> bool: ...
```

**Parameters:**

- `name` (str | UUID | iModel): Service identifier

**Returns**: `bool` - True if registered

**Examples:**

```python
# Check by name
if registry.has("anthropic_messages"):
    print("Registered")

# Check with 'in' operator
if "anthropic_messages" in registry:
    print("Registered")

# Check by UUID
if some_uuid in registry:
    print("Registered")
```

#### `list_names()`

List all registered service names.

**Signature:**

```python
def list_names(self) -> list[str]: ...
```

**Returns**: `list[str]` - List of service names

**Example:**

```python
names = registry.list_names()
# ["anthropic_messages", "openai_chat", "groq_chat"]
```

#### `list_by_tag()`

List services that have the given tag.

**Signature:**

```python
def list_by_tag(self, tag: str) -> list[iModel]: ...
```

**Parameters:**

- `tag` (str): Tag to filter by

**Returns**: `list[iModel]` - Models with tag

**Examples:**

```python
# Get all LLM models
llm_models = registry.list_by_tag("llm")

# Get all Anthropic models
anthropic_models = registry.list_by_tag("anthropic")
```

#### `clear()`

Remove all registered services.

**Signature:**

```python
def clear(self) -> None: ...
```

**Example:**

```python
registry.clear()
print(len(registry))  # 0
```

#### `count()` / `__len__()`

Return number of registered services.

**Signature:**

```python
def count(self) -> int: ...
def __len__(self) -> int: ...
```

**Returns**: `int` - Number of registered services

**Examples:**

```python
# Count method
count = registry.count()

# len() operator
count = len(registry)
```

### MCP Integration

#### `register_mcp_server()`

Register tools from an MCP server.

**Signature:**

```python
async def register_mcp_server(
    self,
    server_config: dict,
    tool_names: list[str] | None = None,
    request_options: dict[str, type] | None = None,
    update: bool = False,
) -> list[str]: ...
```

**Parameters:**

- `server_config` (dict): MCP server configuration (command, args, env)
- `tool_names` (list[str] | None): Specific tools to load (None = all tools)
- `request_options` (dict[str, type] | None): Custom request schemas per tool name
- `update` (bool): Update existing registrations (default: False)

**Returns**: `list[str]` - List of registered tool names

**Example:**

```python
from lionpride.services import ServiceRegistry

registry = ServiceRegistry()

# Load all tools from server
tool_names = await registry.register_mcp_server(
    server_config={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        "env": {"ALLOWED_DIRS": "/tmp"},
    }
)

# Load specific tools
tool_names = await registry.register_mcp_server(
    server_config={...},
    tool_names=["read_file", "write_file"],
)

# Use registered tools
model = registry.get("read_file")
calling = await model.invoke(path="/tmp/test.txt")
```

#### `load_mcp_config()`

Load tools from MCP configuration file.

**Signature:**

```python
async def load_mcp_config(
    self,
    config_path: str,
    server_names: list[str] | None = None,
    update: bool = False,
) -> dict[str, list[str]]: ...
```

**Parameters:**

- `config_path` (str): Path to MCP config file (JSON)
- `server_names` (list[str] | None): Specific servers to load (None = all servers)
- `update` (bool): Update existing registrations (default: False)

**Returns**: `dict[str, list[str]]` - Mapping of server name to list of tool names

**Example:**

```python
from lionpride.services import ServiceRegistry

registry = ServiceRegistry()

# Load from config file
tool_map = await registry.load_mcp_config(
    config_path="/path/to/mcp-config.json"
)
# {"filesystem": ["read_file", "write_file"], "github": ["create_pr", ...]}

# Load specific servers
tool_map = await registry.load_mcp_config(
    config_path="/path/to/mcp-config.json",
    server_names=["filesystem"],
)
```

## Usage Patterns

### Basic Usage: Multi-Provider Setup

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Register multiple providers
providers = ["anthropic", "openai", "groq"]
for provider in providers:
    model = iModel(
        provider=provider,
        api_key=f"{provider.upper()}_API_KEY",
        limit_requests=50,
    )
    registry.register(model)

# Use by name
model = registry.get("anthropic_messages")
calling = await model.invoke(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Advanced Usage: Tag-Based Selection

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Register models with tags
anthropic = iModel(provider="anthropic", api_key="...")
anthropic.backend.config.tags = ["llm", "anthropic", "reasoning"]
registry.register(anthropic)

openai = iModel(provider="openai", api_key="...")
openai.backend.config.tags = ["llm", "openai", "fast"]
registry.register(openai)

# Get all LLM models
llm_models = registry.list_by_tag("llm")

# Get reasoning models
reasoning_models = registry.list_by_tag("reasoning")

# Get fast models
fast_models = registry.list_by_tag("fast")

# Use based on requirements
calling = await reasoning_models[0].invoke(...)
```

### Advanced Usage: MCP Tool Integration

```python
from lionpride.services import ServiceRegistry
import asyncio

registry = ServiceRegistry()

# Load filesystem tools
fs_tools = await registry.register_mcp_server(
    server_config={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        "env": {"ALLOWED_DIRS": "/tmp,/home/user/docs"},
    }
)

# Load GitHub tools
gh_tools = await registry.register_mcp_server(
    server_config={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": "ghp_..."},
    }
)

# Use tools
read_file = registry.get("read_file")
content = await read_file.invoke(path="/tmp/test.txt")

create_pr = registry.get("create_pr")
pr = await create_pr.invoke(
    repo="owner/repo",
    title="Fix bug",
    body="Description",
)
```

### Advanced Usage: Configuration-Driven Registry

```python
from lionpride.services import iModel, ServiceRegistry
import json

# Load config
with open("services.json", "r") as f:
    config = json.load(f)

registry = ServiceRegistry()

# Register from config
for service_config in config["services"]:
    model = iModel(**service_config)
    registry.register(model)

# Use by name
model = registry.get(config["default_service"])
calling = await model.invoke(...)
```

**services.json**:

```json
{
  "services": [
    {
      "provider": "anthropic",
      "api_key": "ANTHROPIC_API_KEY",
      "limit_requests": 50,
      "limit_tokens": 100000
    },
    {
      "provider": "openai",
      "api_key": "OPENAI_API_KEY",
      "limit_requests": 50
    }
  ],
  "default_service": "anthropic_messages"
}
```

### Advanced Usage: Hot-Swap Services

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Initial registration
registry.register(iModel(provider="openai", api_key="KEY1"))

# Hot-swap with updated config (update=True)
registry.register(
    iModel(provider="openai", api_key="KEY2", limit_requests=100),
    update=True,
)

# Same name, different backend
model = registry.get("openai_chat")  # Now uses KEY2 with limit_requests=100
```

## Common Pitfalls

### Issue: Duplicate name without update flag

**Problem**: Registering model with existing name raises error.

```python
# ❌ WRONG: Duplicate name without update
registry.register(iModel(provider="anthropic", api_key="KEY1"))
registry.register(iModel(provider="anthropic", api_key="KEY2"))
# ValueError: Service 'anthropic_messages' already registered
```

**Solution**: Use `update=True` to replace existing registration.

```python
# ✅ CORRECT: Update existing registration
registry.register(iModel(provider="anthropic", api_key="KEY1"))
registry.register(iModel(provider="anthropic", api_key="KEY2"), update=True)
```

### Issue: Confusing get() failure modes

**Problem**: `get()` raises `KeyError` instead of returning None.

```python
# ❌ WRONG: Assuming get() returns None
model = registry.get("nonexistent")  # KeyError
```

**Solution**: Use `has()` to check existence or catch `KeyError`.

```python
# ✅ CORRECT: Check before get
if registry.has("nonexistent"):
    model = registry.get("nonexistent")

# ✅ CORRECT: Catch exception
try:
    model = registry.get("nonexistent")
except KeyError:
    print("Service not found")
```

### Issue: MCP tools overwrite existing registrations

**Problem**: MCP tool names collide with existing registrations.

```python
# ❌ WRONG: Collision without update
registry.register(my_custom_tool_named_read_file)
await registry.register_mcp_server(
    server_config={...},  # Has tool named "read_file"
    update=False,  # Default
)
# ValueError: Service 'read_file' already registered
```

**Solution**: Use `update=True` or choose non-colliding names.

```python
# ✅ CORRECT: Update flag
await registry.register_mcp_server(
    server_config={...},
    update=True,  # Replaces existing
)

# ✅ CORRECT: Rename custom tool
my_tool.backend.config.name = "custom_read_file"
registry.register(my_tool)
```

### Issue: Assuming registry persists state

**Problem**: Registry is in-memory only, not persistent.

```python
# ❌ WRONG: Assuming persistence across restarts
registry.register(model)
# ... process restart
model = registry.get("model_name")  # KeyError: registry is empty
```

**Solution**: Rebuild registry on startup from config or database.

```python
# ✅ CORRECT: Load config on startup
def initialize_registry():
    registry = ServiceRegistry()
    config = load_config()
    for service in config["services"]:
        model = iModel(**service)
        registry.register(model)
    return registry

registry = initialize_registry()
```

## Design Rationale

### Why Pile[iModel] for storage?

`Pile[iModel]` provides:

1. **O(1) UUID Lookup**: Direct access via `pile[uuid]`
2. **Type Safety**: Enforces `iModel` instances only
3. **Iteration**: Supports iteration over all models
4. **Thread-Safe**: Pile operations are thread-safe

### Why separate name index?

Separate `_name_index` enables:

1. **Human-Readable Access**: `get("anthropic_messages")` instead of `get(uuid)`
2. **Name Uniqueness**: Enforce unique names across models
3. **Fast Lookup**: O(1) name → UUID → model
4. **Update Support**: Atomic name index updates

### Why not persist state?

In-memory registry simplifies:

1. **Simplicity**: No database dependencies
2. **Flexibility**: Use any persistence mechanism (file, DB, Redis)
3. **Composability**: Registry is value, not side effect

For persistence, wrap registry in custom persistence layer.

### Why MCP integration?

MCP integration enables:

1. **Tool Discovery**: Automatically load tools from servers
2. **Unified Interface**: Tools and LLMs use same registry
3. **Dynamic Loading**: Load tools at runtime without code changes
4. **Schema Validation**: MCP tools include JSON schemas

## See Also

- **Core**:
  - [`iModel`](imodel.md): Unified service interface
  - [Pile](../base/pile.md): O(1) UUID-indexed collection
- **Providers**:
  - [Providers Overview](providers.md): All provider endpoints
- **Integration**:
  - [MCP Integration](../session/mcp.md): MCP server integration

## Examples

### Example 1: Multi-Model Orchestration

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Register models with different capabilities
registry.register(
    iModel(
        provider="anthropic",
        api_key="ANTHROPIC_API_KEY",
        limit_requests=50,
    )
)  # Best for reasoning

registry.register(
    iModel(
        provider="groq",
        api_key="GROQ_API_KEY",
        limit_requests=100,
    )
)  # Best for speed

registry.register(
    iModel(
        provider="openai",
        api_key="OPENAI_API_KEY",
        limit_requests=50,
    )
)  # Best for cost

# Route based on requirements
async def process_task(task):
    if task["priority"] == "reasoning":
        model = registry.get("anthropic_messages")
    elif task["priority"] == "speed":
        model = registry.get("groq_chat")
    else:
        model = registry.get("openai_chat")

    return await model.invoke(
        model=task["model"],
        messages=task["messages"]
    )
```

### Example 2: Dynamic Tool Loading

```python
from lionpride.services import ServiceRegistry
import asyncio

async def main():
    registry = ServiceRegistry()

    # Load tools from multiple MCP servers
    servers = [
        {
            "name": "filesystem",
            "config": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "env": {"ALLOWED_DIRS": "/tmp"},
            }
        },
        {
            "name": "github",
            "config": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "ghp_..."},
            }
        },
    ]

    for server in servers:
        tools = await registry.register_mcp_server(server["config"])
        print(f"Loaded {len(tools)} tools from {server['name']}")

    # List all tools
    print(f"Total tools: {len(registry)}")
    print(f"Tool names: {registry.list_names()}")

    # Execute tools
    read_file = registry.get("read_file")
    content = await read_file.invoke(path="/tmp/test.txt")

asyncio.run(main())
```

### Example 3: Fallback Chain

```python
from lionpride.services import iModel, ServiceRegistry

registry = ServiceRegistry()

# Register models in priority order
providers = ["anthropic", "openai", "groq"]
for provider in providers:
    model = iModel(provider=provider, api_key=f"{provider.upper()}_API_KEY")
    registry.register(model)

async def call_with_fallback(messages, model_name="gpt-4o-mini"):
    """Try models in priority order until success."""
    for name in registry.list_names():
        try:
            model = registry.get(name)
            calling = await model.invoke(
                model=model_name,
                messages=messages,
                timeout=30.0,
            )
            return calling
        except Exception as e:
            print(f"{name} failed: {e}")
            continue

    raise RuntimeError("All models failed")

# Use
result = await call_with_fallback([{"role": "user", "content": "Hello"}])
```
