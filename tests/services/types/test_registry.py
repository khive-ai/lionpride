# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage tests for ServiceRegistry.

Target: 100% branch coverage for src/lionpride/services/types/registry.py

Coverage areas:
- Basic CRUD: register, get, unregister, has
- Duplicate handling: register with update=True/False
- Discovery: list_names, list_by_tag, count, clear
- Operators: __contains__, __len__, __repr__
- MCP Integration: register_mcp_server, load_mcp_config (delegates to loader)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from lionpride.services import Calling, ServiceBackend, ServiceRegistry, iModel
from lionpride.services.mcps.loader import create_mcp_callable

# =============================================================================
# Mock Components
# =============================================================================


class MockCalling(Calling):
    """Mock Calling for testing."""

    async def _invoke(self) -> Any:
        """Execute mock invocation."""
        return "mock_response"


class MockBackend(ServiceBackend):
    """Mock ServiceBackend implementation."""

    @property
    def event_type(self) -> type[Calling]:
        """Return MockCalling type."""
        return MockCalling

    async def call(self, *args, **kwargs) -> Any:
        """Mock call implementation."""
        return {"status": "success"}


# =============================================================================
# Basic CRUD Tests
# =============================================================================


class TestRegistryBasicOperations:
    """Test basic registry operations."""

    def test_creation(self):
        """Test empty registry creation."""
        registry = ServiceRegistry()
        assert registry.count() == 0
        assert len(registry) == 0
        assert registry.list_names() == []

    def test_register(self):
        """Test registering an iModel."""
        registry = ServiceRegistry()
        backend = MockBackend(config={"provider": "test", "name": "service1"})
        model = iModel(backend=backend)

        uid = registry.register(model)

        assert uid == model.id
        assert registry.count() == 1
        assert "service1" in registry

    def test_register_duplicate_raises(self):
        """Test registering duplicate name raises ValueError."""
        registry = ServiceRegistry()
        backend1 = MockBackend(config={"provider": "test", "name": "duplicate"})
        backend2 = MockBackend(config={"provider": "test", "name": "duplicate"})

        registry.register(iModel(backend=backend1))

        with pytest.raises(ValueError, match="already registered"):
            registry.register(iModel(backend=backend2))

    def test_register_update_replaces(self):
        """Test update=True replaces existing service."""
        registry = ServiceRegistry()
        backend1 = MockBackend(config={"provider": "test", "name": "service", "version": "1.0.0"})
        backend2 = MockBackend(config={"provider": "test", "name": "service", "version": "2.0.0"})

        model1 = iModel(backend=backend1)
        model2 = iModel(backend=backend2)

        uid1 = registry.register(model1)
        assert registry.get("service").version == "1.0.0"

        uid2 = registry.register(model2, update=True)
        assert registry.get("service").version == "2.0.0"
        assert uid2 != uid1

    def test_register_update_removes_old_from_pile(self):
        """Test update=True removes old service from pile."""
        registry = ServiceRegistry()
        backend1 = MockBackend(config={"provider": "test", "name": "service"})
        backend2 = MockBackend(config={"provider": "test", "name": "service"})

        model1 = iModel(backend=backend1)
        model2 = iModel(backend=backend2)

        uid1 = registry.register(model1)
        assert registry.count() == 1

        registry.register(model2, update=True)

        # Should still be 1 service
        assert registry.count() == 1

        # Old UUID should not exist
        assert uid1 not in registry._pile

    def test_get(self):
        """Test get by name."""
        registry = ServiceRegistry()
        backend = MockBackend(config={"provider": "test", "name": "my_service"})
        model = iModel(backend=backend)
        registry.register(model)

        retrieved = registry.get("my_service")

        assert retrieved is model
        assert retrieved.name == "my_service"

    def test_get_not_found_raises(self):
        """Test get raises KeyError for missing service."""
        registry = ServiceRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_get_with_default(self):
        """Test get returns default when service not found (line 140)."""
        registry = ServiceRegistry()

        result = registry.get("nonexistent", default=None)

        assert result is None

    def test_get_with_default_custom_value(self):
        """Test get returns custom default value when service not found."""
        registry = ServiceRegistry()
        sentinel = object()

        result = registry.get("nonexistent", default=sentinel)

        assert result is sentinel

    def test_get_by_uuid(self):
        """Test get by UUID directly (line 135)."""
        registry = ServiceRegistry()
        backend = MockBackend(config={"provider": "test", "name": "my_service"})
        model = iModel(backend=backend)
        registry.register(model)

        # Get by UUID directly
        retrieved = registry.get(model.id)

        assert retrieved is model
        assert retrieved.name == "my_service"

    def test_get_with_imodel_passthrough(self):
        """Test get with iModel returns same instance (line 136-137)."""
        registry = ServiceRegistry()
        backend = MockBackend(config={"provider": "test", "name": "my_service"})
        model = iModel(backend=backend)
        registry.register(model)

        # Pass iModel directly - should return same instance
        retrieved = registry.get(model)

        assert retrieved is model

    def test_unregister(self):
        """Test unregister removes service."""
        registry = ServiceRegistry()
        backend = MockBackend(config={"provider": "test", "name": "temp"})
        model = iModel(backend=backend)
        registry.register(model)

        removed = registry.unregister("temp")

        assert removed is model
        assert registry.count() == 0
        assert "temp" not in registry

    def test_unregister_not_found_raises(self):
        """Test unregister raises KeyError for missing service."""
        registry = ServiceRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.unregister("nonexistent")

    def test_has(self):
        """Test has() existence check."""
        registry = ServiceRegistry()
        backend = MockBackend(config={"provider": "test", "name": "service"})
        registry.register(iModel(backend=backend))

        assert registry.has("service")
        assert not registry.has("missing")

    def test_contains_operator(self):
        """Test __contains__ operator."""
        registry = ServiceRegistry()
        backend = MockBackend(config={"provider": "test", "name": "service"})
        registry.register(iModel(backend=backend))

        assert "service" in registry
        assert "missing" not in registry


# =============================================================================
# Discovery Tests
# =============================================================================


class TestRegistryDiscovery:
    """Test discovery operations."""

    def test_list_names(self):
        """Test list_names returns all registered names."""
        registry = ServiceRegistry()
        for i in range(3):
            backend = MockBackend(config={"provider": "test", "name": f"service{i}"})
            registry.register(iModel(backend=backend))

        names = registry.list_names()

        assert len(names) == 3
        assert set(names) == {"service0", "service1", "service2"}

    def test_list_by_tag(self):
        """Test list_by_tag filters by tag."""
        registry = ServiceRegistry()

        backend1 = MockBackend(
            config={"provider": "test", "name": "prod1", "tags": ["production", "api"]}
        )
        backend2 = MockBackend(
            config={"provider": "test", "name": "prod2", "tags": ["production", "worker"]}
        )
        backend3 = MockBackend(config={"provider": "test", "name": "dev1", "tags": ["development"]})

        registry.register(iModel(backend=backend1))
        registry.register(iModel(backend=backend2))
        registry.register(iModel(backend=backend3))

        prod_services = registry.list_by_tag("production")
        api_services = registry.list_by_tag("api")
        dev_services = registry.list_by_tag("development")

        assert {m.name for m in prod_services} == {"prod1", "prod2"}
        assert [m.name for m in api_services] == ["prod1"]
        assert [m.name for m in dev_services] == ["dev1"]

    def test_count(self):
        """Test count() returns number of services."""
        registry = ServiceRegistry()
        assert registry.count() == 0

        backend = MockBackend(config={"provider": "test", "name": "service"})
        registry.register(iModel(backend=backend))

        assert registry.count() == 1

    def test_len_operator(self):
        """Test __len__() operator."""
        registry = ServiceRegistry()
        assert len(registry) == 0

        backend = MockBackend(config={"provider": "test", "name": "service"})
        registry.register(iModel(backend=backend))

        assert len(registry) == 1

    def test_clear(self):
        """Test clear() removes all services."""
        registry = ServiceRegistry()

        for i in range(5):
            backend = MockBackend(config={"provider": "test", "name": f"service{i}"})
            registry.register(iModel(backend=backend))

        assert registry.count() == 5

        registry.clear()

        assert registry.count() == 0
        assert registry.list_names() == []

    def test_repr(self):
        """Test __repr__() string representation."""
        registry = ServiceRegistry()

        for i in range(3):
            backend = MockBackend(config={"provider": "test", "name": f"service{i}"})
            registry.register(iModel(backend=backend))

        repr_str = repr(registry)

        assert "ServiceRegistry" in repr_str
        assert "3" in repr_str


# =============================================================================
# MCP Integration Tests
# =============================================================================


class TestRegistryMCPIntegration:
    """Test MCP integration methods."""

    async def test_register_mcp_server_specific_tools(self):
        """Test register_mcp_server with specific tool_names."""
        registry = ServiceRegistry()

        # Mock MCP components
        mock_client = Mock()
        mock_client.call_tool = AsyncMock(return_value=Mock(content=[Mock(text="result")]))

        server_config = {"server": "test_server"}
        tool_names = ["tool1", "tool2"]

        # Mock create_mcp_callable from loader
        with patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create:
            mock_callable = AsyncMock(return_value="mock_result")
            mock_create.return_value = mock_callable

            # Mock Tool class
            with patch("lionpride.services.types.tool.Tool") as MockTool:
                mock_tool_instance = Mock()
                MockTool.return_value = mock_tool_instance

                registered = await registry.register_mcp_server(
                    server_config=server_config, tool_names=tool_names
                )

                assert len(registered) == 2
                assert "test_server_tool1" in registered
                assert "test_server_tool2" in registered

    async def test_register_mcp_server_duplicate_without_update_raises(self):
        """Test register_mcp_server raises when tool exists and update=False."""
        registry = ServiceRegistry()

        # Pre-register a tool
        backend = MockBackend(config={"provider": "test", "name": "server_tool1"})
        registry.register(iModel(backend=backend))

        server_config = {"server": "server"}
        tool_names = ["tool1"]

        with pytest.raises(ValueError, match="already registered"):
            await registry.register_mcp_server(
                server_config=server_config, tool_names=tool_names, update=False
            )

    async def test_register_mcp_server_duplicate_with_update_replaces(self):
        """Test register_mcp_server with update=True replaces existing tool."""
        registry = ServiceRegistry()

        # Pre-register a tool
        backend = MockBackend(config={"provider": "test", "name": "server_tool1"})
        registry.register(iModel(backend=backend))

        server_config = {"server": "server"}
        tool_names = ["tool1"]

        # Mock _create_mcp_callable and Tool
        with patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create:
            mock_callable = AsyncMock(return_value="mock_result")
            mock_create.return_value = mock_callable

            with patch("lionpride.services.types.tool.Tool") as MockTool:
                mock_tool_instance = Mock()
                MockTool.return_value = mock_tool_instance

                registered = await registry.register_mcp_server(
                    server_config=server_config, tool_names=tool_names, update=True
                )

                assert "server_tool1" in registered

    async def test_register_mcp_server_with_request_options(self):
        """Test register_mcp_server with request_options."""
        from pydantic import BaseModel

        class CustomOptions(BaseModel):
            param: str

        registry = ServiceRegistry()

        server_config = {"server": "server"}
        tool_names = ["tool1"]
        request_options = {"tool1": CustomOptions}

        with patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create:
            mock_callable = AsyncMock(return_value="mock_result")
            mock_create.return_value = mock_callable

            with patch("lionpride.services.types.tool.ToolConfig") as MockToolConfig:
                mock_config = Mock()
                mock_config.request_options = CustomOptions
                MockToolConfig.return_value = mock_config

                with patch("lionpride.services.types.tool.Tool") as MockTool:
                    mock_tool_instance = Mock()
                    MockTool.return_value = mock_tool_instance

                    await registry.register_mcp_server(
                        server_config=server_config,
                        tool_names=tool_names,
                        request_options=request_options,
                    )

                    # Verify ToolConfig was called with request_options
                    assert MockToolConfig.call_args.kwargs.get("request_options") == CustomOptions

    async def test_register_mcp_server_tool_creation_fails(self):
        """Test register_mcp_server handles tool creation failure gracefully."""
        registry = ServiceRegistry()

        server_config = {"server": "server"}
        tool_names = ["tool1", "tool2"]

        with (
            patch("lionpride.services.mcps.loader.create_mcp_callable"),
            patch("lionpride.services.types.tool.Tool", side_effect=Exception("Tool error")),
        ):
            # Should not raise, but skip failed tools
            registered = await registry.register_mcp_server(
                server_config=server_config, tool_names=tool_names
            )

            assert len(registered) == 0

    async def test_register_mcp_server_auto_discover(self):
        """Test register_mcp_server auto-discovers tools when tool_names=None."""
        registry = ServiceRegistry()

        # Mock MCP client and tools
        mock_tool1 = Mock()
        mock_tool1.name = "discovered_tool1"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = Mock()
        mock_tool2.name = "discovered_tool2"
        mock_tool2.inputSchema = None

        mock_client = Mock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        server_config = {"command": "test", "args": []}

        with (
            patch("lionpride.services.mcps.MCPConnectionPool.get_client", return_value=mock_client),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create,
            patch("lionpride.services.types.tool.Tool") as MockTool,
            patch("lionpride.schema_handlers.typescript_schema"),
        ):
            mock_callable = AsyncMock(return_value="mock_result")
            mock_create.return_value = mock_callable

            mock_tool_instance = Mock()
            MockTool.return_value = mock_tool_instance

            registered = await registry.register_mcp_server(
                server_config=server_config, tool_names=None
            )

            # Should have discovered and registered both tools
            assert len(registered) == 2

    async def test_register_mcp_server_auto_discover_duplicate_skips(self):
        """Test auto-discover skips duplicates when update=False."""
        registry = ServiceRegistry()

        # Pre-register a tool
        backend = MockBackend(config={"provider": "test", "name": "discovered_tool1"})
        registry.register(iModel(backend=backend))

        mock_tool1 = Mock()
        mock_tool1.name = "discovered_tool1"
        mock_tool1.inputSchema = None

        mock_client = Mock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1])

        server_config = {"command": "test", "args": []}

        with (
            patch("lionpride.services.mcps.MCPConnectionPool.get_client", return_value=mock_client),
            patch("lionpride.services.mcps.loader.create_mcp_callable"),
            patch("lionpride.services.types.tool.Tool"),
        ):
            registered = await registry.register_mcp_server(
                server_config=server_config, tool_names=None, update=False
            )

            # Should skip the duplicate
            assert len(registered) == 0

    async def test_register_mcp_server_schema_extraction_fails(self):
        """Test schema extraction failure is handled gracefully."""
        registry = ServiceRegistry()

        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool1.inputSchema = {"invalid": "schema"}

        mock_client = Mock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1])

        server_config = {"command": "test", "args": []}

        with (
            patch("lionpride.services.mcps.MCPConnectionPool.get_client", return_value=mock_client),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create,
            patch("lionpride.services.types.tool.Tool") as MockTool,
            patch(
                "lionpride.schema_handlers.typescript_schema",
                side_effect=Exception("Schema error"),
            ),
        ):
            mock_callable = AsyncMock()
            mock_create.return_value = mock_callable

            mock_tool_instance = Mock()
            MockTool.return_value = mock_tool_instance

            registered = await registry.register_mcp_server(
                server_config=server_config, tool_names=None
            )

            # Should still register despite schema error
            assert len(registered) == 1

    async def test_create_mcp_callable_wrapper(self):
        """Test create_mcp_callable creates functional wrapper."""
        # Mock client
        mock_result = Mock()
        mock_result.content = [Mock(text="test_result")]

        mock_client = Mock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        server_config = {"server": "test"}

        with patch(
            "lionpride.services.mcps.MCPConnectionPool.get_client", return_value=mock_client
        ):
            wrapper = create_mcp_callable(server_config, "tool_name")

            result = await wrapper(param1="value1", param2="value2")

            assert result == "test_result"
            mock_client.call_tool.assert_called_once_with(
                "tool_name", {"param1": "value1", "param2": "value2"}
            )

    async def test_create_mcp_callable_dict_response(self):
        """Test create_mcp_callable handles dict response."""
        mock_result = [{"type": "text", "text": "dict_result"}]

        mock_client = Mock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        server_config = {"server": "test"}

        with patch(
            "lionpride.services.mcps.MCPConnectionPool.get_client", return_value=mock_client
        ):
            wrapper = create_mcp_callable(server_config, "tool_name")

            result = await wrapper(param="value")

            assert result == "dict_result"

    async def test_create_mcp_callable_fallback_response(self):
        """Test create_mcp_callable fallback for unknown response format."""
        mock_result = "raw_string_result"

        mock_client = Mock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        server_config = {"server": "test"}

        with patch(
            "lionpride.services.mcps.MCPConnectionPool.get_client", return_value=mock_client
        ):
            wrapper = create_mcp_callable(server_config, "tool_name")

            result = await wrapper()

            assert result == "raw_string_result"

    async def test_load_mcp_config(self):
        """Test load_mcp_config delegates to loader module."""
        registry = ServiceRegistry()

        mock_result = {
            "server1": ["tool1", "tool2"],
            "server2": ["tool1", "tool2"],
        }

        with patch(
            "lionpride.services.mcps.loader.load_mcp_config", return_value=mock_result
        ) as mock_load:
            result = await registry.load_mcp_config("/path/to/.mcp.json")

            # Verify delegation occurred with correct parameters
            mock_load.assert_called_once_with(
                registry=registry,
                config_path="/path/to/.mcp.json",
                server_names=None,
                update=False,
            )

            assert "server1" in result
            assert "server2" in result
            assert result["server1"] == ["tool1", "tool2"]
            assert result["server2"] == ["tool1", "tool2"]

    async def test_load_mcp_config_specific_servers(self):
        """Test load_mcp_config delegates with specific server_names."""
        registry = ServiceRegistry()

        mock_result = {
            "server1": ["tool1"],
            "server3": ["tool1"],
        }

        with patch(
            "lionpride.services.mcps.loader.load_mcp_config", return_value=mock_result
        ) as mock_load:
            result = await registry.load_mcp_config(
                "/path/to/.mcp.json", server_names=["server1", "server3"]
            )

            # Verify delegation with server_names parameter
            mock_load.assert_called_once_with(
                registry=registry,
                config_path="/path/to/.mcp.json",
                server_names=["server1", "server3"],
                update=False,
            )

            assert "server1" in result
            assert "server3" in result
            assert "server2" not in result

    async def test_load_mcp_config_server_failure(self):
        """Test load_mcp_config delegates and handles server registration failure."""
        registry = ServiceRegistry()

        # Mock result with one server success, one server failure
        mock_result = {
            "server1": ["tool1"],
            "server2": [],  # Failed, empty list
        }

        with patch(
            "lionpride.services.mcps.loader.load_mcp_config", return_value=mock_result
        ) as mock_load:
            result = await registry.load_mcp_config("/path/to/.mcp.json")

            # Verify delegation occurred
            mock_load.assert_called_once_with(
                registry=registry,
                config_path="/path/to/.mcp.json",
                server_names=None,
                update=False,
            )

            assert result["server1"] == ["tool1"]
            assert result["server2"] == []  # Failed, empty list


# =============================================================================
# Tool Schema Tests (lines 206-232)
# =============================================================================


class TestRegistryToolSchemas:
    """Test get_tool_schemas method (lines 206-232)."""

    def test_get_tool_schemas_all_tools(self):
        """Test get_tool_schemas with no tool_names returns all Tool schemas."""
        from lionpride.services.types.tool import Tool

        registry = ServiceRegistry()

        # Create real Tool backends
        def my_func(x: str) -> str:
            return x

        tool = Tool(func_callable=my_func)
        model = iModel(backend=tool)
        registry.register(model)

        schemas = registry.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "object"
        assert "properties" in schemas[0]

    def test_get_tool_schemas_specific_tools(self):
        """Test get_tool_schemas with specific tool_names."""
        from lionpride.services.types.tool import Tool

        registry = ServiceRegistry()

        def func1(x: str) -> str:
            return x

        def func2(y: int) -> int:
            return y

        tool1 = Tool(func_callable=func1)
        tool2 = Tool(func_callable=func2)
        registry.register(iModel(backend=tool1))
        registry.register(iModel(backend=tool2))

        # Get only func1
        schemas = registry.get_tool_schemas(tool_names=["func1"])

        assert len(schemas) == 1

    def test_get_tool_schemas_missing_tool_raises(self):
        """Test get_tool_schemas raises KeyError for missing tool."""
        registry = ServiceRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_tool_schemas(tool_names=["nonexistent"])

    def test_get_tool_schemas_non_tool_backend_raises(self):
        """Test get_tool_schemas raises TypeError for non-Tool backend."""
        registry = ServiceRegistry()

        # Register a non-Tool backend (MockBackend)
        backend = MockBackend(config={"provider": "test", "name": "not_a_tool"})
        registry.register(iModel(backend=backend))

        with pytest.raises(TypeError, match="does not have Tool backend"):
            registry.get_tool_schemas(tool_names=["not_a_tool"])

    def test_get_tool_schemas_empty_when_no_tools(self):
        """Test get_tool_schemas returns empty list when no Tool backends."""
        registry = ServiceRegistry()

        # Register non-Tool backend only
        backend = MockBackend(config={"provider": "test", "name": "service"})
        registry.register(iModel(backend=backend))

        schemas = registry.get_tool_schemas()

        assert schemas == []

    def test_get_tool_schemas_skips_none_schema(self):
        """Test get_tool_schemas skips tools with None schema."""
        from lionpride.services.types.tool import Tool

        registry = ServiceRegistry()

        def my_func(x: str) -> str:
            return x

        tool = Tool(func_callable=my_func)
        # Manually set tool_schema to None (edge case)
        object.__setattr__(tool, "tool_schema", None)

        model = iModel(backend=tool)
        registry.register(model)

        schemas = registry.get_tool_schemas()

        # Should return empty - tool_schema is None
        assert schemas == []
