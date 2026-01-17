# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from lionpride.libs.concurrency import Lock

__all__ = (
    "MCP_ENV_ALLOWLIST",
    "CommandNotAllowedError",
    "MCPConnectionPool",
    "MCPConnectionPoolInstance",
    "MCPSecurityConfig",
    "create_mcp_pool",
    "filter_mcp_environment",
)


# Default environment variable allowlist for MCP subprocesses
# Only these variables (or patterns) are inherited from the parent environment
# This prevents accidental leakage of sensitive environment variables (API keys, tokens, etc.)
MCP_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # System essentials
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        # Locale settings (LC_* handled via pattern)
        "LANG",
        "LANGUAGE",
        # Python environment
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        # Node.js environment
        "NODE_PATH",
        "NODE_ENV",
        "NPM_CONFIG_PREFIX",
        # MCP-specific variables (MCP_*, FASTMCP_* handled via pattern)
    }
)

# Patterns for environment variables that should be allowed
# These are checked via regex if the exact name is not in MCP_ENV_ALLOWLIST
_MCP_ENV_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^LC_"),  # Locale settings: LC_ALL, LC_CTYPE, etc.
    re.compile(r"^MCP_"),  # MCP-specific: MCP_DEBUG, MCP_QUIET, etc.
    re.compile(r"^FASTMCP_"),  # FastMCP-specific: FASTMCP_QUIET, etc.
)


def filter_mcp_environment(
    env: dict[str, str] | None = None,
    allowlist: frozenset[str] | set[str] | None = None,
    patterns: tuple[re.Pattern, ...] | None = None,
    debug: bool = False,
) -> dict[str, str]:
    """Filter environment variables to only include allowed ones for MCP subprocesses.

    This function filters environment variables to prevent accidental leakage of
    sensitive data (API keys, tokens, credentials) to MCP subprocess environments.

    Args:
        env: Source environment dict. If None, uses os.environ.
        allowlist: Set of exact variable names to allow. Defaults to MCP_ENV_ALLOWLIST.
        patterns: Tuple of compiled regex patterns to match. Defaults to _MCP_ENV_PATTERNS
            (LC_*, MCP_*, FASTMCP_*).
        debug: If True, logs variables that were filtered out.

    Returns:
        Filtered environment dictionary containing only allowed variables.

    Example:
        >>> # Get filtered environment with defaults
        >>> env = filter_mcp_environment()
        >>> "PATH" in env  # Allowed
        True
        >>> "OPENAI_API_KEY" in env  # Filtered out (not in allowlist)
        False
        >>>
        >>> # Custom allowlist
        >>> env = filter_mcp_environment(allowlist={"PATH", "HOME", "MY_SAFE_VAR"})
    """
    if env is None:
        env = dict(os.environ)
    if allowlist is None:
        allowlist = MCP_ENV_ALLOWLIST
    if patterns is None:
        patterns = _MCP_ENV_PATTERNS

    filtered = {}
    excluded = []

    for key, value in env.items():
        # Check exact match first
        if key in allowlist:
            filtered[key] = value
            continue

        # Check pattern match
        if any(pattern.match(key) for pattern in patterns):
            filtered[key] = value
            continue

        # Not allowed
        excluded.append(key)

    if debug and excluded:
        logging.debug(
            "MCP subprocess environment filtered. Excluded %d variables: %s",
            len(excluded),
            ", ".join(sorted(excluded)[:10]) + ("..." if len(excluded) > 10 else ""),
        )

    return filtered


class CommandNotAllowedError(Exception):
    """Raised when a command is not in the allowlist.

    This exception is raised when strict_mode is enabled (default) and
    a command is attempted that is not in the configured allowlist.
    """

    pass


# Default safe commands for MCP servers
# These are commonly used interpreters/runners that MCP servers typically use
DEFAULT_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        # Python
        "python",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
        "python3.13",
        # Node.js
        "node",
        "npx",
        "npm",
        # Package managers / runners
        "uv",
        "uvx",
        "pipx",
        "pdm",
        "poetry",
        "rye",
        # Other common MCP server runners
        "deno",
        "bun",
    }
)

# Suppress MCP server logging by default
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)


@dataclass(frozen=True)
class MCPSecurityConfig:
    """Immutable security configuration for MCP connection pools.

    This configuration is frozen at creation time and cannot be modified afterward,
    preventing runtime security weakening.

    Attributes:
        allowed_commands: Set of command names allowed to execute.
        strict_mode: If True, only allowlisted commands can execute.
    """

    allowed_commands: frozenset[str] = field(default_factory=lambda: DEFAULT_ALLOWED_COMMANDS)
    strict_mode: bool = True

    def __post_init__(self):
        """Ensure allowed_commands is a frozenset."""
        if not isinstance(self.allowed_commands, frozenset):
            object.__setattr__(self, "allowed_commands", frozenset(self.allowed_commands))

    def with_commands(self, additional_commands: set[str]) -> "MCPSecurityConfig":
        """Create a new config with additional allowed commands.

        Args:
            additional_commands: Commands to add to the allowlist.

        Returns:
            New MCPSecurityConfig with extended allowlist.
        """
        return MCPSecurityConfig(
            allowed_commands=self.allowed_commands | frozenset(additional_commands),
            strict_mode=self.strict_mode,
        )


class MCPConnectionPoolInstance:
    """Session-scoped connection pool for MCP clients.

    Unlike the global MCPConnectionPool, this class maintains instance-level state,
    making it safe for concurrent use across multiple sessions without state leakage.

    Args:
        security_config: Immutable security configuration. Defaults to strict mode
            with standard allowed commands.
        configs: Pre-loaded server configurations (from .mcp.json).

    Example:
        >>> # Create session-scoped pool
        >>> security = MCPSecurityConfig(
        ...     allowed_commands=frozenset({"python", "node", "my-runner"}),
        ...     strict_mode=True,
        ... )
        >>> pool = MCPConnectionPoolInstance(security_config=security)
        >>>
        >>> # Load config and get client
        >>> pool.load_config(".mcp.json")
        >>> client = await pool.get_client({"server": "search"})
        >>>
        >>> # Cleanup when done
        >>> await pool.cleanup()
    """

    def __init__(
        self,
        security_config: MCPSecurityConfig | None = None,
        configs: dict[str, dict] | None = None,
    ):
        """Initialize session-scoped connection pool.

        Args:
            security_config: Immutable security config. If None, uses default strict mode.
            configs: Pre-loaded server configurations.
        """
        self._security = security_config or MCPSecurityConfig()
        self._clients: dict[str, Any] = {}
        self._configs: dict[str, dict] = configs.copy() if configs else {}
        self._lock = Lock()

    @property
    def security_config(self) -> MCPSecurityConfig:
        """Get the immutable security configuration."""
        return self._security

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, *_):
        """Context manager exit - cleanup connections."""
        await self.cleanup()

    def _validate_command(self, command: str) -> None:
        """Validate a command against the security config.

        Args:
            command: The command to validate.

        Raises:
            CommandNotAllowedError: If strict_mode and command not allowed.
        """
        if not self._security.strict_mode:
            return

        if "/" in command or "\\" in command:
            raise CommandNotAllowedError(
                f"Command '{command}' contains path separators which are not allowed "
                f"in strict mode. Use bare command names (e.g., 'python' not './python')."
            )

        if command not in self._security.allowed_commands:
            allowed_list = ", ".join(sorted(self._security.allowed_commands))
            raise CommandNotAllowedError(
                f"Command '{command}' is not in the allowlist. Allowed commands: [{allowed_list}]."
            )

    def load_config(self, path: str = ".mcp.json") -> None:
        """Load MCP server configurations from file.

        Args:
            path: Path to .mcp.json configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file has invalid JSON or structure.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"MCP config file not found: {path}")

        try:
            content = config_path.read_text(encoding="utf-8")
            data = orjson.loads(content)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid JSON in MCP config file: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("MCP config must be a JSON object")

        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise ValueError("mcpServers must be a dictionary")

        self._configs.update(servers)

    async def get_client(self, server_config: dict[str, Any]) -> Any:
        """Get or create a pooled MCP client.

        Args:
            server_config: Either {"server": "name"} or full config with command/args.

        Returns:
            FastMCP Client instance (connected).

        Raises:
            ValueError: If server reference not found or config invalid.
        """
        if server_config.get("server") is not None:
            server_name = server_config["server"]
            if server_name not in self._configs:
                self.load_config()
                if server_name not in self._configs:
                    raise ValueError(f"Unknown MCP server: {server_name}")

            config = self._configs[server_name]
            cache_key = f"server:{server_name}"
        else:
            config = server_config
            cache_key = f"inline:{config.get('command')}:{id(config)}"

        async with self._lock:
            if cache_key in self._clients:
                client = self._clients[cache_key]
                if hasattr(client, "is_connected") and client.is_connected():
                    return client
                else:
                    del self._clients[cache_key]

            client = await self._create_client(config)
            self._clients[cache_key] = client
            return client

    async def _create_client(self, config: dict[str, Any]) -> Any:
        """Create a new MCP client from config."""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        if not any(config.get(k) is not None for k in ["url", "command"]):
            raise ValueError("Config must have either 'url' or 'command' with non-None value")

        try:
            from fastmcp import Client as FastMCPClient
        except ImportError as e:
            raise ImportError("FastMCP not installed. Run: pip install fastmcp") from e

        if config.get("url") is not None:
            client = FastMCPClient(config["url"])
        elif config.get("command") is not None:
            command = config["command"]
            self._validate_command(command)

            args = config.get("args", [])
            if not isinstance(args, list):
                raise ValueError("Config 'args' must be a list")

            debug_mode = (
                config.get("debug", False) or os.environ.get("MCP_DEBUG", "").lower() == "true"
            )

            env = filter_mcp_environment(debug=debug_mode)
            env.update(config.get("env", {}))

            if not debug_mode:
                env.setdefault("LOG_LEVEL", "ERROR")
                env.setdefault("PYTHONWARNINGS", "ignore")
                env.setdefault("FASTMCP_QUIET", "true")
                env.setdefault("MCP_QUIET", "true")

            from fastmcp.client.transports import StdioTransport

            transport = StdioTransport(command=command, args=args, env=env)
            client = FastMCPClient(transport)
        else:
            raise ValueError("Config must have 'url' or 'command' with non-None value")

        await client.__aenter__()
        return client

    async def cleanup(self):
        """Clean up all pooled connections."""
        async with self._lock:
            for cache_key, client in self._clients.items():
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    logging.debug(f"Error cleaning up MCP client {cache_key}: {e}")
            self._clients.clear()


def create_mcp_pool(
    allowed_commands: set[str] | None = None,
    strict_mode: bool = True,
    extend_defaults: bool = True,
    configs: dict[str, dict] | None = None,
) -> MCPConnectionPoolInstance:
    """Factory function to create a session-scoped MCP connection pool.

    Args:
        allowed_commands: Additional commands to allow. If None, uses defaults only.
        strict_mode: If True, only allowlisted commands can execute.
        extend_defaults: If True, allowed_commands extends defaults. If False, replaces.
        configs: Pre-loaded server configurations.

    Returns:
        New MCPConnectionPoolInstance with the specified security settings.

    Example:
        >>> # Create pool with custom commands
        >>> pool = create_mcp_pool(allowed_commands={"my-runner"})
        >>>
        >>> # Create pool with only specific commands (no defaults)
        >>> pool = create_mcp_pool(
        ...     allowed_commands={"python", "node"},
        ...     extend_defaults=False,
        ... )
    """
    if allowed_commands is None:
        base_commands = DEFAULT_ALLOWED_COMMANDS
    elif extend_defaults:
        base_commands = DEFAULT_ALLOWED_COMMANDS | frozenset(allowed_commands)
    else:
        base_commands = frozenset(allowed_commands)

    security = MCPSecurityConfig(allowed_commands=base_commands, strict_mode=strict_mode)
    return MCPConnectionPoolInstance(security_config=security, configs=configs)


class MCPConnectionPool:
    """Global connection pool for MCP clients.

    .. deprecated::
        This class uses global state shared across all sessions. Use
        :class:`MCPConnectionPoolInstance` or :func:`create_mcp_pool` for
        session-scoped isolation.

    Manages FastMCP client instances with connection pooling and lifecycle management.
    Clients are cached by config and reused across calls for efficiency.

    Warning:
        This class uses class-level state that is shared globally. For session
        isolation, use MCPConnectionPoolInstance instead:

        >>> pool = create_mcp_pool(allowed_commands={"my-runner"})
        >>> client = await pool.get_client({"server": "search"})
        >>> await pool.cleanup()

    Security:
        By default, only commands in the allowlist can be executed (strict_mode=True).
        Use configure_security() to customize the allowlist or disable strict mode.

    Example:
        >>> # Load config
        >>> MCPConnectionPool.load_config(".mcp.json")
        >>>
        >>> # Get client (auto-connects)
        >>> client = await MCPConnectionPool.get_client({"server": "search"})
        >>> result = await client.call_tool("exa_search", {"query": "AI"})
        >>>
        >>> # Cleanup on shutdown
        >>> await MCPConnectionPool.cleanup()
    """

    _clients: dict[str, Any] = {}
    _configs: dict[str, dict] = {}
    _lock = Lock()

    # Security: Command allowlist
    _allowed_commands: set[str] = set(DEFAULT_ALLOWED_COMMANDS)
    _strict_mode: bool = True

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, *_):
        """Context manager exit - cleanup connections."""
        await self.cleanup()

    @classmethod
    def configure_security(
        cls,
        allowed_commands: set[str] | None = None,
        strict_mode: bool | None = None,
        extend_defaults: bool = True,
    ) -> None:
        """Configure command execution security settings.

        .. deprecated::
            This method modifies global state affecting all sessions. Use
            :func:`create_mcp_pool` with security options for session isolation.

        Args:
            allowed_commands: Set of allowed command names. If extend_defaults=True,
                these are added to the default allowlist. If extend_defaults=False,
                these replace the allowlist entirely.
            strict_mode: If True (default), only allowlisted commands can execute.
                If False, all commands are allowed (use with caution).
            extend_defaults: If True (default), allowed_commands extends the default
                allowlist. If False, allowed_commands replaces it entirely.

        Example:
            >>> # Preferred: use create_mcp_pool for session isolation
            >>> pool = create_mcp_pool(allowed_commands={"my-runner"})
            >>>
            >>> # Legacy: global configuration (deprecated)
            >>> MCPConnectionPool.configure_security(allowed_commands={"my-custom-runner"})
        """
        warnings.warn(
            "MCPConnectionPool.configure_security() modifies global state. "
            "Use create_mcp_pool() for session-scoped isolation.",
            DeprecationWarning,
            stacklevel=2,
        )
        if strict_mode is not None:
            cls._strict_mode = strict_mode

        if allowed_commands is not None:
            if extend_defaults:
                cls._allowed_commands = set(DEFAULT_ALLOWED_COMMANDS) | allowed_commands
            else:
                cls._allowed_commands = set(allowed_commands)

    @classmethod
    def reset_security(cls) -> None:
        """Reset security settings to defaults.

        Restores:
        - strict_mode to True
        - allowed_commands to DEFAULT_ALLOWED_COMMANDS
        """
        cls._strict_mode = True
        cls._allowed_commands = set(DEFAULT_ALLOWED_COMMANDS)

    @classmethod
    def _validate_command(cls, command: str) -> None:
        """Validate a command against the allowlist.

        In strict mode, commands with path separators are rejected to prevent
        attackers from bypassing the allowlist with paths like ./python or
        /tmp/python. Only bare command names that will be resolved via PATH
        are allowed.

        Args:
            command: The command to validate (must be bare name in strict mode)

        Raises:
            CommandNotAllowedError: If strict_mode is True and:
                - command contains path separators (/, \\)
                - command not in allowlist
        """
        if not cls._strict_mode:
            return

        # In strict mode, reject any command with path separators
        # This prevents bypass via ./python, /tmp/python, etc.
        if "/" in command or "\\" in command:
            raise CommandNotAllowedError(
                f"Command '{command}' contains path separators which are not allowed "
                f"in strict mode. Use bare command names (e.g., 'python' not './python'). "
                f"This prevents allowlist bypass via malicious binaries in writable paths."
            )

        if command not in cls._allowed_commands:
            allowed_list = ", ".join(sorted(cls._allowed_commands))
            raise CommandNotAllowedError(
                f"Command '{command}' is not in the allowlist. "
                f"Allowed commands: [{allowed_list}]. "
                f"Use MCPConnectionPool.configure_security() to add custom commands "
                f"or set strict_mode=False (not recommended)."
            )

    @classmethod
    def load_config(cls, path: str = ".mcp.json") -> None:
        """Load MCP server configurations from file.

        Args:
            path: Path to .mcp.json configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file has invalid JSON or structure is invalid

        Example:
            >>> MCPConnectionPool.load_config(".mcp.json")
            >>> # Now can reference servers: {"server": "name"}
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"MCP config file not found: {path}")

        try:
            content = config_path.read_text(encoding="utf-8")
            data = orjson.loads(content)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid JSON in MCP config file: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("MCP config must be a JSON object")

        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise ValueError("mcpServers must be a dictionary")

        cls._configs.update(servers)

    @classmethod
    async def get_client(cls, server_config: dict[str, Any]) -> Any:
        """Get or create a pooled MCP client.

        Args:
            server_config: Either {"server": "name"} or full config with command/args

        Returns:
            FastMCP Client instance (connected)

        Raises:
            ValueError: If server reference not found or config invalid

        Example:
            >>> # Via server reference
            >>> client = await MCPConnectionPool.get_client({"server": "search"})
            >>>
            >>> # Via inline config
            >>> client = await MCPConnectionPool.get_client(
            ...     {
            ...         "command": "python",
            ...         "args": ["-m", "server"],
            ...     }
            ... )
        """
        # Generate unique key for this config
        if server_config.get("server") is not None:
            # Server reference from .mcp.json
            server_name = server_config["server"]
            if server_name not in cls._configs:
                # Try loading config
                cls.load_config()
                if server_name not in cls._configs:
                    raise ValueError(f"Unknown MCP server: {server_name}")

            config = cls._configs[server_name]
            cache_key = f"server:{server_name}"
        else:
            # Inline config - use command as key
            config = server_config
            cache_key = f"inline:{config.get('command')}:{id(config)}"

        # Check if client exists and is connected
        async with cls._lock:
            if cache_key in cls._clients:
                client = cls._clients[cache_key]
                # Simple connectivity check
                if hasattr(client, "is_connected") and client.is_connected():
                    return client
                else:
                    # Remove stale client
                    del cls._clients[cache_key]

            # Create new client
            client = await cls._create_client(config)
            cls._clients[cache_key] = client
            return client

    @classmethod
    async def _create_client(cls, config: dict[str, Any]) -> Any:
        """Create a new MCP client from config.

        Args:
            config: Server configuration with 'url' or 'command' + optional 'args' and 'env'

        Raises:
            ValueError: If config format is invalid
            ImportError: If fastmcp not installed
        """
        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        # Check that at least one of url or command has a non-None value
        if not any(config.get(k) is not None for k in ["url", "command"]):
            raise ValueError("Config must have either 'url' or 'command' with non-None value")

        try:
            from fastmcp import Client as FastMCPClient
        except ImportError as e:
            raise ImportError("FastMCP not installed. Run: pip install fastmcp") from e

        # Handle different config formats
        if config.get("url") is not None:
            # Direct URL connection
            client = FastMCPClient(config["url"])
        elif config.get("command") is not None:
            # Command-based connection
            command = config["command"]

            # SECURITY: Validate command against allowlist
            cls._validate_command(command)

            # Validate args if provided
            args = config.get("args", [])
            if not isinstance(args, list):
                raise ValueError("Config 'args' must be a list")

            # Check debug mode
            debug_mode = (
                config.get("debug", False) or os.environ.get("MCP_DEBUG", "").lower() == "true"
            )

            # SECURITY: Filter environment variables to prevent leaking secrets
            # Only allowlisted variables are passed to the subprocess
            env = filter_mcp_environment(debug=debug_mode)

            # Merge user-specified environment variables (these take precedence)
            env.update(config.get("env", {}))

            # Suppress server logging unless debug mode is enabled
            if not debug_mode:
                # Common environment variables to suppress logging
                env.setdefault("LOG_LEVEL", "ERROR")
                env.setdefault("PYTHONWARNINGS", "ignore")
                # Suppress FastMCP server logs
                env.setdefault("FASTMCP_QUIET", "true")
                env.setdefault("MCP_QUIET", "true")

            # Create client with command
            from fastmcp.client.transports import StdioTransport

            transport = StdioTransport(
                command=command,
                args=args,
                env=env,
            )
            client = FastMCPClient(transport)
        else:
            # Defense-in-depth: should never reach here due to validation at line 160
            raise ValueError("Config must have 'url' or 'command' with non-None value")

        # Initialize connection
        await client.__aenter__()
        return client

    @classmethod
    async def cleanup(cls):
        """Clean up all pooled connections.

        Safe to call multiple times. Errors are logged but don't raise.

        Example:
            >>> await MCPConnectionPool.cleanup()
        """
        async with cls._lock:
            for cache_key, client in cls._clients.items():
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    # Log cleanup errors for debugging while continuing cleanup
                    logging.debug(f"Error cleaning up MCP client {cache_key}: {e}")
            cls._clients.clear()
