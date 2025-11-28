# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from uuid import UUID

from lionpride import Pile

from .imodel import iModel

__all__ = ("ServiceRegistry",)


class ServiceRegistry:
    """Service registry managing iModel instances with O(1) name-based lookup.

    ServiceRegistry is the unified service boundary for all iModel instances.
    iModel is the interface that wraps ServiceBackend (Endpoint for LLMs, Tool
    for functions). It provides:
    - Type-safe storage via Pile[iModel]
    - O(1) name-based lookup via internal index
    - Tag-based filtering
    - MCP server integration
    - Tool schema extraction for LLM tool calling

    Architecture:
        - Storage: Pile[iModel] (UUID-based, type-safe)
        - Index: dict[str, UUID] (name → UUID mapping)
        - iModel wraps ServiceBackend:
            - Endpoint: LLM API backends (OpenAI, Anthropic, etc.)
            - Tool: Function wrappers (calculators, APIs, MCP tools)

    Registration patterns:
        1. Direct registration: registry.register(iModel(backend=...))
        2. Provider shorthand: registry.register(iModel(provider="openai"))
        3. MCP server: await registry.register_mcp_server(config)
        4. MCP config file: await registry.load_mcp_config(path)

    Attributes:
        _pile: Internal Pile[iModel] storage
        _name_index: Internal name → UUID mapping

    Example:
        Basic registration:
            >>> from lionpride.services import ServiceRegistry, iModel
            >>> registry = ServiceRegistry()
            >>> model = iModel(provider="openai", model="gpt-4")
            >>> registry.register(model)
            >>> model = registry.get("gpt-4")

        Tool registration (wrapped in iModel):
            >>> from lionpride.services import Tool, iModel
            >>> tool = Tool(func_callable=my_calculator)
            >>> model = iModel(backend=tool)
            >>> registry.register(model)

        MCP integration:
            >>> await registry.register_mcp_server(
            ...     {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem"]},
            ...     tool_names=["read_file", "write_file"],
            ... )
            >>> tool_model = registry.get("read_file")

        Tag-based filtering:
            >>> llms = registry.list_by_tag("llm")
            >>> tools = registry.list_by_tag("tool")
    """

    def __init__(self):
        """Initialize empty registry with Pile storage and name index."""
        from .imodel import iModel

        self._pile: Pile[iModel] = Pile(item_type=iModel)
        self._name_index: dict[str, UUID] = {}

    def register(self, model: iModel, update: bool = False) -> UUID:
        """Register iModel instance by name with O(1) lookup.

        Services must have unique names. Duplicate registration raises
        ValueError unless update=True.

        Args:
            model: iModel instance (wraps Endpoint or Tool backend).
            update: If True, replaces existing service with same name.

        Returns:
            UUID: Service ID for reference.

        Raises:
            ValueError: If service with same name exists and update=False.

        Example:
            Register LLM:
                >>> model = iModel(provider="openai", model="gpt-4")
                >>> registry.register(model)

            Update existing:
                >>> new_model = iModel(provider="openai", model="gpt-4-turbo")
                >>> registry.register(new_model, update=True)

            Register tool (wrapped in iModel):
                >>> tool = Tool(func_callable=my_calc_fn)
                >>> model = iModel(backend=tool)
                >>> registry.register(model)

            Check before registering:
                >>> if not registry.has("gpt-4"):
                ...     registry.register(model)
        """
        if model.name in self._name_index:
            if not update:
                raise ValueError(f"Service '{model.name}' already registered")
            # Update: remove old, add new
            old_uid = self._name_index[model.name]
            self._pile.remove(old_uid)

        self._pile.add(model)
        self._name_index[model.name] = model.id

        return model.id

    def unregister(self, name: str) -> iModel:
        """Remove and return service by name."""
        if name not in self._name_index:
            raise KeyError(f"Service '{name}' not found")

        uid = self._name_index.pop(name)
        return self._pile.remove(uid)

    def get(self, name: str | UUID | iModel, default: Any = ...) -> iModel:
        """Get service by name."""
        if isinstance(name, UUID):
            return self._pile[name]
        if isinstance(name, iModel):
            return name
        if name not in self._name_index:
            if default is not ...:
                return default
            raise KeyError(f"Service '{name}' not found")

        uid = self._name_index[name]
        return self._pile[uid]

    def has(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._name_index

    def list_names(self) -> list[str]:
        """List all registered service names."""
        return list(self._name_index.keys())

    def list_by_tag(self, tag: str) -> Pile[iModel]:
        """Get services with specific tag."""
        return self._pile[lambda m: tag in m.tags]

    def count(self) -> int:
        """Count registered services."""
        return len(self._pile)

    def clear(self) -> None:
        """Remove all registered services."""
        self._pile.clear()
        self._name_index.clear()

    def __len__(self) -> int:
        """Return number of registered services."""
        return len(self._pile)

    def __contains__(self, name: str) -> bool:
        """Check if service exists (supports `name in registry`)."""
        return name in self._name_index

    def __repr__(self) -> str:
        """String representation."""
        return f"ServiceRegistry(count={len(self)})"

    def get_tool_schemas(
        self,
        tool_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tool schemas for LLM tool calling.

        Extracts JSON schemas from iModel instances with Tool backends
        for injection into LLM prompts. Used by operate() to enable tool calling.

        Args:
            tool_names: Optional list of specific tool names to get schemas for.
                If None, returns schemas for all registered tool-backed services.

        Returns:
            List of tool schema dicts in format:
            {"type": "function", "function": {"name": "...", "parameters": {...}}}

        Raises:
            KeyError: If a specified tool name is not registered
            TypeError: If a specified name refers to non-Tool-backed service

        Example:
            >>> tool = Tool(func_callable=multiply_fn)
            >>> registry.register(iModel(backend=tool))
            >>> schemas = registry.get_tool_schemas()
            >>> # Pass schemas to LLM for tool calling
        """
        from .tool import Tool

        # Get all tool-backed services or specific ones
        if tool_names is None:
            # Filter for iModel instances with Tool backends
            tool_models = list(self._pile[lambda m: isinstance(m.backend, Tool)])
        else:
            # Get specific services and validate they have Tool backends
            tool_models = []
            for name in tool_names:
                if not self.has(name):
                    raise KeyError(f"Tool '{name}' not found in registry")
                service = self.get(name)
                if not isinstance(service.backend, Tool):
                    raise TypeError(
                        f"Service '{name}' does not have Tool backend "
                        f"(backend type: {type(service.backend).__name__})"
                    )
                tool_models.append(service)

        # Extract schemas from Tool backends (backend is verified as Tool above)
        schemas = []
        for model in tool_models:
            backend = model.backend
            if isinstance(backend, Tool) and backend.tool_schema is not None:
                schemas.append(backend.tool_schema)
        return schemas

    # =========================================================================
    # MCP Integration Methods
    # =========================================================================

    async def register_mcp_server(
        self,
        server_config: dict,
        tool_names: list[str] | None = None,
        request_options: dict[str, type] | None = None,
        update: bool = False,
    ) -> list[str]:
        """Register tools from Model Context Protocol (MCP) server.

        MCP servers expose tools (functions) that can be called by models. This
        method spawns an MCP server process, discovers available tools, and
        registers them as iModel instances.

        Tool discovery:
        - If tool_names=None, discovers and registers ALL available tools
        - If tool_names provided, registers only those specific tools

        Args:
            server_config: MCP server configuration dict:
                - Direct config: {"command": "npx", "args": [...], "env": {...}}
                - Reference: {"server": "filesystem"} (requires .mcp.json)
            tool_names: Optional list of specific tool names to register.
                If None, discovers and registers all available tools.
            request_options: Optional dict mapping tool names to Pydantic models
                for request validation.
            update: If True, replaces existing tools with same names.

        Returns:
            List of registered tool names (subset if tool_names provided).

        Raises:
            RuntimeError: If MCP server fails to start or tool discovery fails.
            ValueError: If tool already registered and update=False.

        Example:
            Register all tools from MCP server:
                >>> tools = await registry.register_mcp_server(
                ...     {
                ...         "command": "npx",
                ...         "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                ...         "env": {"ALLOWED_DIRS": "/tmp"},
                ...     }
                ... )
                >>> # tools = ["read_file", "write_file", "list_directory", ...]

            Register specific tools only:
                >>> tools = await registry.register_mcp_server(
                ...     {"command": "npx", "args": ["-y", "..."]},
                ...     tool_names=["read_file", "write_file"],
                ... )

            Reference .mcp.json config:
                >>> tools = await registry.register_mcp_server({"server": "filesystem"})
                >>> # Loads config from ~/.config/mcp/.mcp.json

            Use registered tools in operations:
                >>> result = await branch.operate(
                ...     session,
                ...     "Read the file and summarize",
                ...     response_model=Summary,
                ...     model=model,
                ...     tools=True,  # Inject all registered tools
                ...     actions=True,  # Auto-execute tool calls
                ... )

        Common patterns:
            Selective tool registration:
                >>> # Register only safe filesystem tools
                >>> await registry.register_mcp_server(
                ...     {"server": "filesystem"}, tool_names=["read_file", "list_directory"]
                ... )

            Update tools on server restart:
                >>> await registry.register_mcp_server(
                ...     {"server": "filesystem"},
                ...     update=True,  # Replace existing tools
                ... )
        """
        from lionpride.services.mcps.loader import load_mcp_tools

        return await load_mcp_tools(
            registry=self,
            server_config=server_config,
            tool_names=tool_names,
            request_options=request_options,
            update=update,
        )

    async def load_mcp_config(
        self,
        config_path: str,
        server_names: list[str] | None = None,
        update: bool = False,
    ) -> dict[str, list[str]]:
        """Load MCP configurations from a .mcp.json file.

        Delegates to load_mcp_config from the loader module.

        Args:
            config_path: Path to .mcp.json configuration file
            server_names: Optional list of server names to load.
                         If None, loads all servers.
            update: If True, allow updating existing tools.

        Returns:
            Dict mapping server names to lists of registered tool names
        """
        from lionpride.services.mcps.loader import load_mcp_config

        return await load_mcp_config(
            registry=self,
            config_path=config_path,
            server_names=server_names,
            update=update,
        )
