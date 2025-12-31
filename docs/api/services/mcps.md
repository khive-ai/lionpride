# MCP Integration

> Model Context Protocol (MCP) connection pooling, security, and tool loading.

## Overview

The `mcps` module provides secure, session-scoped integration with MCP servers. It
handles connection pooling, command allowlisting, environment filtering, and tool
registration into lionpride's `ServiceRegistry`.

**Key Design Principles:**

- **Session-scoped isolation**: Each session gets its own connection pool via
  `create_mcp_pool()`, preventing cross-session state leakage
- **Defense-in-depth security**: Command allowlisting and environment variable filtering
  prevent arbitrary command execution and credential leakage
- **Unified tool interface**: MCP tools are wrapped as `Tool` instances and registered
  via `iModel` for consistent invocation patterns

## Architecture

```text
+------------------+     +----------------------+     +------------------+
|  ServiceRegistry |<----|  load_mcp_tools()    |---->|  MCP Server      |
|                  |     |  load_mcp_config()   |     |  (subprocess)    |
+------------------+     +----------------------+     +------------------+
        |                         |                          ^
        v                         v                          |
+------------------+     +----------------------+     +------------------+
|  iModel(Tool)    |     | MCPConnectionPool-   |---->|  FastMCP Client  |
|                  |     | Instance             |     |                  |
+------------------+     +----------------------+     +------------------+
                                  |
                                  v
                         +----------------------+
                         |  MCPSecurityConfig   |
                         |  - allowed_commands  |
                         |  - strict_mode       |
                         +----------------------+
```

---

## Security Model

### Command Allowlisting

MCP servers execute as subprocesses. To prevent arbitrary command execution, lionpride
enforces a strict command allowlist:

```python
DEFAULT_ALLOWED_COMMANDS = frozenset({
    # Python
    "python", "python3", "python3.10", "python3.11", "python3.12", "python3.13",
    # Node.js
    "node", "npx", "npm",
    # Package managers / runners
    "uv", "uvx", "pipx", "pdm", "poetry", "rye",
    # Other common MCP server runners
    "deno", "bun",
})
```

**Strict Mode Enforcement:**

- Commands with path separators (`/`, `\`) are rejected to prevent allowlist bypass via
  paths like `./python` or `/tmp/malicious`
- Only bare command names resolved via `PATH` are permitted
- Raises `CommandNotAllowedError` for non-allowlisted commands

### Environment Variable Filtering

To prevent credential leakage to MCP subprocesses, environment variables are filtered:

```python
MCP_ENV_ALLOWLIST = frozenset({
    # System essentials
    "PATH", "HOME", "USER", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP",
    # Locale settings
    "LANG", "LANGUAGE",
    # Python environment
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    # Node.js environment
    "NODE_PATH", "NODE_ENV", "NPM_CONFIG_PREFIX",
})
```

**Pattern Matching:**

Variables matching these patterns are also allowed:

- `LC_*` (locale settings)
- `MCP_*` (MCP-specific)
- `FASTMCP_*` (FastMCP-specific)

**Result:** Sensitive variables like `OPENAI_API_KEY`, `AWS_SECRET_ACCESS_KEY`, etc. are
NOT passed to MCP subprocesses.

---

## Connection Pool

### create_mcp_pool()

Factory function to create session-scoped MCP connection pools. **This is the
recommended API.**

```python
def create_mcp_pool(
    allowed_commands: set[str] | None = None,
    strict_mode: bool = True,
    extend_defaults: bool = True,
    configs: dict[str, dict] | None = None,
) -> MCPConnectionPoolInstance: ...
```

**Parameters:**

| Parameter          | Type                      | Default | Description                                                       |
| ------------------ | ------------------------- | ------- | ----------------------------------------------------------------- |
| `allowed_commands` | `set[str] or None`        | `None`  | Additional commands to allow. If None, uses defaults only.        |
| `strict_mode`      | `bool`                    | `True`  | If True, only allowlisted commands can execute.                   |
| `extend_defaults`  | `bool`                    | `True`  | If True, `allowed_commands` extends defaults. If False, replaces. |
| `configs`          | `dict[str, dict] or None` | `None`  | Pre-loaded server configurations (from `.mcp.json`).              |

**Returns:**

- `MCPConnectionPoolInstance`: New session-scoped connection pool.

**Examples:**

```python
from lionpride.services.mcps import create_mcp_pool

# Create pool with default security
pool = create_mcp_pool()

# Add custom command to allowlist
pool = create_mcp_pool(allowed_commands={"my-custom-runner"})

# Replace defaults entirely (only allow specific commands)
pool = create_mcp_pool(
    allowed_commands={"python", "node"},
    extend_defaults=False,
)

# Disable strict mode (not recommended)
pool = create_mcp_pool(strict_mode=False)
```

---

### MCPConnectionPoolInstance

Session-scoped connection pool for MCP clients. Unlike the deprecated
`MCPConnectionPool`, this class maintains instance-level state for safe concurrent use.

```python
class MCPConnectionPoolInstance:
    def __init__(
        self,
        security_config: MCPSecurityConfig | None = None,
        configs: dict[str, dict] | None = None,
    ) -> None: ...
```

**Parameters:**

| Parameter         | Type                        | Default | Description                                                |
| ----------------- | --------------------------- | ------- | ---------------------------------------------------------- |
| `security_config` | `MCPSecurityConfig or None` | `None`  | Immutable security configuration. Defaults to strict mode. |
| `configs`         | `dict[str, dict] or None`   | `None`  | Pre-loaded server configurations.                          |

**Attributes:**

| Attribute         | Type                | Description                                 |
| ----------------- | ------------------- | ------------------------------------------- |
| `security_config` | `MCPSecurityConfig` | Immutable security configuration (property) |

**Methods:**

#### `load_config()`

Load MCP server configurations from file.

```python
def load_config(self, path: str = ".mcp.json") -> None: ...
```

**Raises:**

- `FileNotFoundError`: If config file does not exist.
- `ValueError`: If config file has invalid JSON or structure.

---

#### `get_client()` (async)

Get or create a pooled MCP client.

```python
async def get_client(self, server_config: dict[str, Any]) -> Any: ...
```

**Parameters:**

- `server_config`: Either `{"server": "name"}` to reference loaded config, or full
  inline config with `command`/`args` or `url`.

**Returns:**

- Connected FastMCP `Client` instance.

**Raises:**

- `ValueError`: If server reference not found or config invalid.
- `CommandNotAllowedError`: If command not in allowlist (strict mode).

---

#### `cleanup()` (async)

Clean up all pooled connections. Safe to call multiple times.

```python
async def cleanup(self) -> None: ...
```

---

**Context Manager Support:**

```python
async with create_mcp_pool() as pool:
    pool.load_config(".mcp.json")
    client = await pool.get_client({"server": "search"})
    result = await client.call_tool("search", {"query": "AI"})
# Connections automatically cleaned up
```

---

### MCPSecurityConfig

Immutable security configuration for MCP connection pools.

```python
@dataclass(frozen=True)
class MCPSecurityConfig:
    allowed_commands: frozenset[str] = DEFAULT_ALLOWED_COMMANDS
    strict_mode: bool = True
```

**Attributes:**

| Attribute          | Type             | Default                    | Description                                     |
| ------------------ | ---------------- | -------------------------- | ----------------------------------------------- |
| `allowed_commands` | `frozenset[str]` | `DEFAULT_ALLOWED_COMMANDS` | Set of command names allowed to execute.        |
| `strict_mode`      | `bool`           | `True`                     | If True, only allowlisted commands can execute. |

**Methods:**

#### `with_commands()`

Create a new config with additional allowed commands.

```python
def with_commands(self, additional_commands: set[str]) -> MCPSecurityConfig: ...
```

**Example:**

```python
from lionpride.services.mcps import MCPSecurityConfig, MCPConnectionPoolInstance

# Start with defaults, add custom command
security = MCPSecurityConfig().with_commands({"my-runner"})

# Create pool with custom security
pool = MCPConnectionPoolInstance(security_config=security)
```

---

### MCPConnectionPool (Deprecated)

Global connection pool for MCP clients. **Deprecated: uses class-level state shared
across all sessions.**

```python
class MCPConnectionPool:
    """
    .. deprecated::
        Use MCPConnectionPoolInstance or create_mcp_pool() for session-scoped isolation.
    """
```

**Why Deprecated:**

- Class-level state causes cross-session interference
- Security settings affect all sessions globally
- Not safe for concurrent multi-session use

**Migration:**

```python
# DEPRECATED
MCPConnectionPool.configure_security(allowed_commands={"my-runner"})
client = await MCPConnectionPool.get_client({"server": "search"})
await MCPConnectionPool.cleanup()

# RECOMMENDED
pool = create_mcp_pool(allowed_commands={"my-runner"})
pool.load_config(".mcp.json")
client = await pool.get_client({"server": "search"})
await pool.cleanup()
```

---

## Tool Loading

### load_mcp_tools()

Load MCP tools into a `ServiceRegistry`.

```python
async def load_mcp_tools(
    registry: Any,  # ServiceRegistry
    server_config: dict[str, Any],
    tool_names: list[str] | None = None,
    request_options: dict[str, type] | None = None,
    update: bool = False,
) -> list[str]: ...
```

**Parameters:**

| Parameter         | Type                      | Default | Description                                                                   |
| ----------------- | ------------------------- | ------- | ----------------------------------------------------------------------------- |
| `registry`        | `ServiceRegistry`         | -       | Registry instance to register tools into.                                     |
| `server_config`   | `dict[str, Any]`          | -       | MCP server config: `{"server": "name"}` or full config with `command`/`args`. |
| `tool_names`      | `list[str] or None`       | `None`  | Specific tools to register. If None, auto-discovers all tools.                |
| `request_options` | `dict[str, type] or None` | `None`  | Pydantic models for request validation, keyed by tool name.                   |
| `update`          | `bool`                    | `False` | If True, allow updating existing tool registrations.                          |

**Returns:**

- `list[str]`: List of registered tool names (qualified with server prefix).

**Examples:**

```python
from lionpride import Session
from lionpride.services.mcps import load_mcp_tools

session = Session()

# Auto-discover all tools from server
tools = await load_mcp_tools(session.services, {"server": "search"})
print(f"Registered: {tools}")  # ["search_exa_search", "search_web_search", ...]

# Register specific tools with validation
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

tools = await load_mcp_tools(
    session.services,
    {"command": "python", "args": ["-m", "my_mcp_server"]},
    tool_names=["search"],
    request_options={"search": SearchRequest},
)
```

---

### load_mcp_config()

Load MCP configurations from a `.mcp.json` file with auto-discovery.

```python
async def load_mcp_config(
    registry: Any,  # ServiceRegistry
    config_path: str,
    server_names: list[str] | None = None,
    update: bool = False,
) -> dict[str, list[str]]: ...
```

**Parameters:**

| Parameter      | Type                | Default | Description                                     |
| -------------- | ------------------- | ------- | ----------------------------------------------- |
| `registry`     | `ServiceRegistry`   | -       | Registry instance to register tools into.       |
| `config_path`  | `str`               | -       | Path to `.mcp.json` configuration file.         |
| `server_names` | `list[str] or None` | `None`  | Specific servers to load. If None, loads all.   |
| `update`       | `bool`              | `False` | If True, allow updating existing registrations. |

**Returns:**

- `dict[str, list[str]]`: Mapping of server names to lists of registered tool names.

**Example:**

```python
from lionpride import Session
from lionpride.services.mcps import load_mcp_config

session = Session()

# Load all servers from config
tools = await load_mcp_config(session.services, ".mcp.json")
# {"search": ["search_exa_search", ...], "memory": ["memory_store", ...]}

# Load specific servers only
tools = await load_mcp_config(
    session.services,
    ".mcp.json",
    server_names=["search", "memory"],
)
```

---

### create_mcp_callable()

Create an async callable that wraps MCP tool execution.

```python
def create_mcp_callable(
    server_config: dict[str, Any],
    tool_name: str,
) -> Callable: ...
```

**Parameters:**

| Parameter       | Type             | Description               |
| --------------- | ---------------- | ------------------------- |
| `server_config` | `dict[str, Any]` | MCP server configuration. |
| `tool_name`     | `str`            | Original MCP tool name.   |

**Returns:**

- `Callable`: Async callable that executes MCP tool via connection pool.

**Example:**

```python
from lionpride.services.mcps import create_mcp_callable

# Create callable for a specific tool
search = create_mcp_callable({"server": "search"}, "exa_search")

# Use directly
result = await search(query="AI research", limit=10)
```

---

## Helper Functions

### filter_mcp_environment()

Filter environment variables for MCP subprocess security.

```python
def filter_mcp_environment(
    env: dict[str, str] | None = None,
    allowlist: frozenset[str] | set[str] | None = None,
    patterns: tuple[re.Pattern, ...] | None = None,
    debug: bool = False,
) -> dict[str, str]: ...
```

**Parameters:**

| Parameter   | Type                                 | Default | Description                                                     |
| ----------- | ------------------------------------ | ------- | --------------------------------------------------------------- |
| `env`       | `dict[str, str] or None`             | `None`  | Source environment. If None, uses `os.environ`.                 |
| `allowlist` | `frozenset[str] or set[str] or None` | `None`  | Exact variable names to allow. Defaults to `MCP_ENV_ALLOWLIST`. |
| `patterns`  | `tuple[re.Pattern, ...] or None`     | `None`  | Regex patterns to match (LC__, MCP__, FASTMCP_*).               |
| `debug`     | `bool`                               | `False` | If True, logs filtered-out variables.                           |

**Returns:**

- `dict[str, str]`: Filtered environment containing only allowed variables.

**Example:**

```python
from lionpride.services.mcps import filter_mcp_environment

# Get filtered environment with defaults
env = filter_mcp_environment()
assert "PATH" in env           # Allowed
assert "OPENAI_API_KEY" not in env  # Filtered out

# Custom allowlist
env = filter_mcp_environment(allowlist={"PATH", "HOME", "MY_SAFE_VAR"})
```

---

## Exceptions

### CommandNotAllowedError

Raised when a command is not in the allowlist (strict mode).

```python
class CommandNotAllowedError(Exception):
    """Raised when a command is not in the allowlist."""
    pass
```

**When Raised:**

- `strict_mode=True` (default) and command not in `allowed_commands`
- Command contains path separators (`/`, `\`)

**Example:**

```python
from lionpride.services.mcps import create_mcp_pool, CommandNotAllowedError

pool = create_mcp_pool()  # strict_mode=True by default

try:
    await pool.get_client({"command": "malicious-binary", "args": []})
except CommandNotAllowedError as e:
    print(f"Blocked: {e}")
    # Blocked: Command 'malicious-binary' is not in the allowlist...
```

---

## Configuration File Format

MCP servers are configured via `.mcp.json`:

```json
{
  "mcpServers": {
    "search": {
      "command": "uvx",
      "args": ["--from", "exa-mcp-server", "exa-mcp-server"],
      "env": {
        "EXA_API_KEY": "${EXA_API_KEY}"
      }
    },
    "memory": {
      "command": "python",
      "args": ["-m", "memory_server"],
      "debug": true
    },
    "remote": {
      "url": "https://mcp.example.com/api"
    }
  }
}
```

**Server Configuration Fields:**

| Field     | Type             | Description                                                  |
| --------- | ---------------- | ------------------------------------------------------------ |
| `command` | `str`            | Command to execute (must be in allowlist).                   |
| `args`    | `list[str]`      | Command arguments.                                           |
| `env`     | `dict[str, str]` | Additional environment variables (merged with filtered env). |
| `debug`   | `bool`           | Enable debug logging for this server.                        |
| `url`     | `str`            | Alternative: direct URL connection (no subprocess).          |

---

## Usage Patterns

### Session-Scoped Pool (Recommended)

```python
from lionpride import Session
from lionpride.services.mcps import create_mcp_pool, load_mcp_tools

async def create_session_with_mcp():
    session = Session()

    # Create session-scoped pool
    pool = create_mcp_pool(allowed_commands={"my-runner"})
    pool.load_config(".mcp.json")

    # Load tools into session registry
    tools = await load_mcp_tools(session.services, {"server": "search"})

    # Use tools via registry
    model = session.services.get("search_exa_search")
    calling = await model.invoke(query="AI research", limit=10)

    # Cleanup when session ends
    await pool.cleanup()

    return calling.response
```

### Context Manager Pattern

```python
from lionpride.services.mcps import create_mcp_pool

async def mcp_operation():
    async with create_mcp_pool() as pool:
        pool.load_config(".mcp.json")
        client = await pool.get_client({"server": "search"})
        result = await client.call_tool("exa_search", {"query": "AI"})
        return result
    # Automatic cleanup on exit
```

### Multi-Server Configuration

```python
from lionpride import Session
from lionpride.services.mcps import load_mcp_config

async def load_all_servers():
    session = Session()

    # Load all servers from config
    all_tools = await load_mcp_config(session.services, ".mcp.json")

    for server, tools in all_tools.items():
        print(f"Server '{server}': {len(tools)} tools registered")

    return session
```

---

## Common Pitfalls

- **Command not allowed**: Ensure custom commands are added to allowlist via
  `create_mcp_pool(allowed_commands={"my-cmd"})`.

- **Path in command**: Use bare command names (`python`), not paths (`./python`,
  `/usr/bin/python`).

- **Environment leakage**: API keys in `os.environ` are NOT passed to MCP subprocesses.
  Use explicit `env` in config if needed.

- **Global state (deprecated)**: Avoid `MCPConnectionPool` class methods. Use
  `create_mcp_pool()` for session isolation.

- **Missing fastmcp**: Install dependency: `pip install fastmcp`.

---

## Design Rationale

**Session Isolation**: `MCPConnectionPoolInstance` was introduced to replace global
`MCPConnectionPool` state, enabling safe concurrent multi-session operation.

**Immutable Security**: `MCPSecurityConfig` is frozen at creation to prevent runtime
security weakening.

**Defense-in-Depth**: Both command allowlisting and environment filtering provide
layered security against arbitrary command execution and credential leakage.

**Unified Tool Interface**: MCP tools are wrapped as lionpride `Tool` instances via
`iModel`, providing consistent `invoke()` patterns across local and remote tools.

---

## See Also

- [`iModel`](./imodel.md): Unified service interface
- [`ServiceRegistry`](./registry.md): Service management
- [`Tool`](./tool.md): Local tool wrapper
- [FastMCP Documentation](https://github.com/jlowin/fastmcp): Underlying MCP client
