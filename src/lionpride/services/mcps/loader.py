# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

__all__ = ("create_mcp_callable", "load_mcp_config", "load_mcp_tools")

logger = logging.getLogger(__name__)


async def load_mcp_tools(
    registry: Any,  # ServiceRegistry
    server_config: dict[str, Any],
    tool_names: list[str] | None = None,
    request_options: dict[str, type] | None = None,
    update: bool = False,
) -> list[str]:
    """Load MCP tools into ServiceRegistry.

    Converts MCP tools to standard Tool instances wrapped in iModel
    and registers them in the provided ServiceRegistry.

    Args:
        registry: ServiceRegistry instance to register tools into
        server_config: MCP server configuration (command, args, etc.)
                      Can be {"server": "name"} to reference loaded config
                      or full config dict with command/args
        tool_names: Optional list of specific tool names to register.
                   If None, will discover and register all available tools.
        request_options: Optional dict mapping tool names to Pydantic model classes
                        for request validation. E.g., {"exa_search": ExaSearchRequest}
        update: If True, allow updating existing tools.

    Returns:
        List of registered tool names (qualified with server prefix)

    Example:
        >>> # Auto-discover all tools
        >>> tools = await load_mcp_tools(session.registry, {"server": "search"})
        >>>
        >>> # Register specific tools with validation
        >>> from pydantic import BaseModel
        >>> class SearchRequest(BaseModel):
        ...     query: str
        ...     limit: int = 10
        >>> tools = await load_mcp_tools(
        ...     session.registry,
        ...     {"command": "python", "args": ["-m", "server"]},
        ...     tool_names=["search"],
        ...     request_options={"search": SearchRequest},
        ... )
    """
    from lionpride.services import Tool, iModel
    from lionpride.services.types.tool import ToolConfig

    from .wrapper import MCPConnectionPool

    registered_tools = []

    # Extract server name for qualified naming
    server_name = None
    if isinstance(server_config, dict) and "server" in server_config:
        server_name = server_config["server"]

    if tool_names:
        # Register specific tools
        for tool_name in tool_names:
            # Qualified name to avoid collisions
            qualified_name = f"{server_name}_{tool_name}" if server_name else tool_name

            # Check for existing registration
            if registry.has(qualified_name) and not update:
                raise ValueError(
                    f"Tool '{qualified_name}' already registered. Use update=True to replace."
                )

            # Get request_options for this tool if provided
            tool_request_options = None
            if request_options and tool_name in request_options:
                tool_request_options = request_options[tool_name]

            # Create MCP wrapper callable
            mcp_callable = create_mcp_callable(
                server_config=server_config,
                tool_name=tool_name,
            )

            # Create and register Tool
            try:
                tool = Tool(
                    func_callable=mcp_callable,
                    config=ToolConfig(
                        name=qualified_name,
                        provider=server_name or "mcp",
                        request_options=tool_request_options,
                    ),
                )
                model = iModel(backend=tool)
                registry.register(model, update=update)
                registered_tools.append(qualified_name)
            except Exception as e:
                logger.warning(f"Failed to register tool {tool_name}: {e}")
    else:
        # Auto-discover tools from the server
        client = await MCPConnectionPool.get_client(server_config)
        tools = await client.list_tools()

        # Register each discovered tool
        for tool in tools:
            # Qualified name to avoid collisions
            qualified_name = f"{server_name}_{tool.name}" if server_name else tool.name

            # Get request_options for this tool if provided
            tool_request_options = None
            if request_options and tool.name in request_options:
                tool_request_options = request_options[tool.name]

            # Extract schema from FastMCP tool directly
            tool_schema = None
            try:
                if (
                    hasattr(tool, "inputSchema")
                    and tool.inputSchema is not None
                    and isinstance(tool.inputSchema, dict)
                ):
                    # Format as OpenAI function calling schema
                    tool_schema = {
                        "type": "function",
                        "function": {
                            "name": qualified_name,  # Use qualified name
                            "description": (
                                tool.description
                                if hasattr(tool, "description") and tool.description
                                else f"MCP tool: {tool.name}"
                            ),
                            "parameters": tool.inputSchema,
                        },
                    }
            except Exception as schema_error:
                # If schema extraction fails, Tool will auto-generate from callable signature
                logger.warning(f"Could not extract schema for {tool.name}: {schema_error}")
                tool_schema = None

            try:
                # Create MCP wrapper callable
                mcp_callable = create_mcp_callable(
                    server_config=server_config,
                    tool_name=tool.name,
                )

                # Register as regular Tool with MCP-discovered schema
                if registry.has(qualified_name) and not update:
                    logger.warning(f"Tool '{qualified_name}' already registered. Skipping.")
                    continue

                tool_obj = Tool(
                    func_callable=mcp_callable,
                    config=ToolConfig(
                        name=qualified_name,
                        provider=server_name or "mcp",
                        request_options=tool_request_options,
                    ),
                    tool_schema=tool_schema,
                )
                model = iModel(backend=tool_obj)
                registry.register(model, update=update)
                registered_tools.append(qualified_name)
            except Exception as e:
                logger.warning(f"Failed to register tool {tool.name}: {e}")

    return registered_tools


def create_mcp_callable(
    server_config: dict[str, Any],
    tool_name: str,
) -> Callable:
    """Create async callable that wraps MCP tool execution.

    Args:
        server_config: MCP server configuration
        tool_name: Original MCP tool name

    Returns:
        Async callable that executes MCP tool via connection pool
    """
    from .wrapper import MCPConnectionPool

    async def mcp_wrapper(**kwargs: Any) -> Any:
        """Execute MCP tool call via connection pool.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool execution result (normalized)
        """
        # Get pooled client
        client = await MCPConnectionPool.get_client(server_config)

        # Call the tool
        result = await client.call_tool(tool_name, kwargs)

        # Extract content from FastMCP response
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list) and len(content) == 1:
                item = content[0]
                if hasattr(item, "text"):
                    return item.text
                elif isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return content
        elif isinstance(result, list) and len(result) == 1:
            item = result[0]
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")

        return result

    # Set function name for introspection
    mcp_wrapper.__name__ = tool_name

    return mcp_wrapper


async def load_mcp_config(
    registry: Any,  # ServiceRegistry
    config_path: str,
    server_names: list[str] | None = None,
    update: bool = False,
) -> dict[str, list[str]]:
    """Load MCP configurations from a .mcp.json file with auto-discovery.

    Args:
        registry: ServiceRegistry instance to register tools into
        config_path: Path to .mcp.json configuration file
        server_names: Optional list of server names to load.
                     If None, loads all servers.
        update: If True, allow updating existing tools.

    Returns:
        Dict mapping server names to lists of registered tool names

    Example:
        >>> # Load all servers and auto-discover their tools
        >>> tools = await load_mcp_config(session.registry, ".mcp.json")
        >>> print(f"Loaded {sum(len(t) for t in tools.values())} tools")
        >>>
        >>> # Load specific servers only
        >>> tools = await load_mcp_config(
        ...     session.registry, ".mcp.json", server_names=["search", "memory"]
        ... )
    """
    from .wrapper import MCPConnectionPool

    # Load the config file into the connection pool
    MCPConnectionPool.load_config(config_path)

    # Get server list to process
    if server_names is None:
        # Get all server names from loaded config
        server_names = list(MCPConnectionPool._configs.keys())

    # Register tools from each server
    all_tools = {}
    for server_name in server_names:
        try:
            # Register using server reference
            tools = await load_mcp_tools(registry, {"server": server_name}, update=update)
            all_tools[server_name] = tools
            logger.info(f"✅ Registered {len(tools)} tools from server '{server_name}'")
        except Exception as e:
            logger.error(f"⚠️  Failed to register server '{server_name}': {e}")
            all_tools[server_name] = []

    return all_tools
