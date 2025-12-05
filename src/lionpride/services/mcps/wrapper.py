# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import Any

import orjson

from lionpride.libs.concurrency import Lock

__all__ = ("CommandNotAllowedError", "MCPConnectionPool")


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


class MCPConnectionPool:
    """Singleton connection pool for MCP clients.

    Manages FastMCP client instances with connection pooling and lifecycle management.
    Clients are cached by config and reused across calls for efficiency.

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
        >>>
        >>> # Security configuration
        >>> MCPConnectionPool.configure_security(
        ...     allowed_commands={"python", "node", "custom-runner"},
        ...     strict_mode=True,  # Default
        ... )
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

        Args:
            allowed_commands: Set of allowed command names. If extend_defaults=True,
                these are added to the default allowlist. If extend_defaults=False,
                these replace the allowlist entirely.
            strict_mode: If True (default), only allowlisted commands can execute.
                If False, all commands are allowed (use with caution).
            extend_defaults: If True (default), allowed_commands extends the default
                allowlist. If False, allowed_commands replaces it entirely.

        Example:
            >>> # Add custom commands to defaults
            >>> MCPConnectionPool.configure_security(allowed_commands={"my-custom-runner"})
            >>>
            >>> # Replace allowlist entirely
            >>> MCPConnectionPool.configure_security(
            ...     allowed_commands={"python", "node"}, extend_defaults=False
            ... )
            >>>
            >>> # Disable strict mode (DANGEROUS - allows any command)
            >>> MCPConnectionPool.configure_security(strict_mode=False)
        """
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

            # Merge environment variables - user config takes precedence
            env = os.environ.copy()
            env.update(config.get("env", {}))

            # Suppress server logging unless debug mode is enabled
            if not (
                config.get("debug", False) or os.environ.get("MCP_DEBUG", "").lower() == "true"
            ):
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
