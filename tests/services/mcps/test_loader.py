# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage tests for MCP loader functions.

Target: 85%+ branch coverage for src/lionpride/services/mcps/loader.py

Coverage areas:
- load_mcp_tools: tool registration, auto-discovery, error handling
- create_mcp_callable: response parsing, content extraction
- load_mcp_config: config loading, server discovery, error handling
"""

import json
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

from lionpride.services.mcps.loader import create_mcp_callable, load_mcp_config, load_mcp_tools


# Helper function for creating mock results
def _create_text_content_mock(text):
    """Create mock result with content.text pattern."""
    mock_result = Mock()
    mock_item = Mock()
    mock_item.text = text
    mock_result.content = [mock_item]
    return mock_result


@pytest.fixture
def mock_registry():
    """Create mock ServiceRegistry."""
    registry = Mock()
    registry.has = Mock(return_value=False)
    registry.register = Mock()
    return registry


@pytest.fixture
def mock_tool_config():
    """Standard tool configuration for testing."""
    return {"server": "test_server", "command": "python", "args": ["-m", "test"]}


@pytest.fixture(autouse=True)
def reset_connection_pool():
    """Reset MCPConnectionPool state before each test."""
    from lionpride.services.mcps.wrapper import MCPConnectionPool

    MCPConnectionPool._clients.clear()
    MCPConnectionPool._configs.clear()
    yield
    MCPConnectionPool._clients.clear()
    MCPConnectionPool._configs.clear()


# =============================================================================
# load_mcp_tools Tests
# =============================================================================


class TestLoadMCPToolsSpecificTools:
    """Test load_mcp_tools with specific tool names."""

    async def test_load_specific_tools_success(self, mock_registry):
        """Test loading specific tools successfully."""
        server_config = {"server": "test_server"}
        tool_names = ["tool1", "tool2"]

        with (
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool = Mock()
            MockTool.return_value = mock_tool

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config, tool_names)

            assert result == ["test_server_tool1", "test_server_tool2"]
            assert mock_registry.register.call_count == 2

    async def test_load_specific_tools_with_request_options(self, mock_registry):
        """Test loading tools with request_options provided."""
        from pydantic import BaseModel

        server_config = {"server": "test_server"}
        tool_names = ["tool1"]

        # Create valid Pydantic model for request_options
        class MockRequestModel(BaseModel):
            query: str

        request_options = {"tool1": MockRequestModel}

        with (
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool = Mock()
            MockTool.return_value = mock_tool

            mock_model = Mock()
            MockiModel.return_value = mock_model

            _result = await load_mcp_tools(
                mock_registry, server_config, tool_names, request_options
            )

            # Verify request_options was passed to ToolConfig
            assert MockTool.called
            tool_call_kwargs = MockTool.call_args.kwargs
            assert tool_call_kwargs["config"].request_options == MockRequestModel

    async def test_load_specific_tools_no_server_name(self, mock_registry):
        """Test loading tools without server name in config."""
        server_config = {"command": "python", "args": ["-m", "test"]}
        tool_names = ["tool1"]

        with (
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool = Mock()
            MockTool.return_value = mock_tool

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config, tool_names)

            # Should use unqualified name when no server name
            assert result == ["tool1"]

    async def test_load_specific_tools_already_registered(self, mock_registry):
        """Test loading tool that's already registered raises ValueError."""
        mock_registry.has = Mock(return_value=True)
        server_config = {"server": "test_server"}
        tool_names = ["tool1"]

        with pytest.raises(ValueError, match="already registered"):
            await load_mcp_tools(mock_registry, server_config, tool_names)

    async def test_load_specific_tools_with_update(self, mock_registry):
        """Test loading tool with update=True allows replacement."""
        mock_registry.has = Mock(return_value=True)
        server_config = {"server": "test_server"}
        tool_names = ["tool1"]

        with (
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool = Mock()
            MockTool.return_value = mock_tool

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config, tool_names, update=True)

            assert result == ["test_server_tool1"]
            mock_registry.register.assert_called_once()

    async def test_load_specific_tools_registration_fails(self, mock_registry):
        """Test loading tool handles registration exceptions (line 106)."""
        server_config = {"server": "test_server"}
        tool_names = ["tool1", "tool2"]

        with (
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
            patch("lionpride.services.mcps.loader.logger") as mock_logger,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            # First tool succeeds, second tool fails
            mock_tool1 = Mock()
            MockTool.side_effect = [mock_tool1, Exception("Tool creation failed")]

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config, tool_names)

            # Should return only successfully registered tools
            assert "test_server_tool1" in result
            assert "test_server_tool2" not in result
            mock_logger.warning.assert_called_once()


class TestLoadMCPToolsAutoDiscovery:
    """Test load_mcp_tools with auto-discovery (tool_names=None)."""

    async def test_auto_discovery_success(self, mock_registry):
        """Test auto-discovery of tools from server."""
        server_config = {"server": "test_server"}

        # Mock discovered tools
        mock_tool1 = Mock()
        mock_tool1.name = "discovered_tool1"
        mock_tool1.description = "Tool 1 description"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = Mock()
        mock_tool2.name = "discovered_tool2"
        mock_tool2.description = None  # Test missing description
        mock_tool2.inputSchema = {"type": "object"}

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        with (
            patch(
                "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
                return_value=mock_client,
            ),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool_obj = Mock()
            MockTool.return_value = mock_tool_obj

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config)

            assert result == [
                "test_server_discovered_tool1",
                "test_server_discovered_tool2",
            ]
            assert mock_registry.register.call_count == 2

    async def test_auto_discovery_with_request_options(self, mock_registry):
        """Test auto-discovery respects request_options (line 120)."""
        from pydantic import BaseModel

        server_config = {"server": "test_server"}

        mock_tool = Mock()
        mock_tool.name = "tool1"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        # Create valid Pydantic model
        class MockRequestModel(BaseModel):
            query: str

        request_options = {"tool1": MockRequestModel}

        with (
            patch(
                "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
                return_value=mock_client,
            ),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool_obj = Mock()
            MockTool.return_value = mock_tool_obj

            mock_model = Mock()
            MockiModel.return_value = mock_model

            _result = await load_mcp_tools(
                mock_registry, server_config, request_options=request_options
            )

            # Verify request_options was used (line 120 covered)
            tool_call_kwargs = MockTool.call_args.kwargs
            assert tool_call_kwargs["config"].request_options == MockRequestModel

    async def test_auto_discovery_schema_extraction_fails(self, mock_registry):
        """Test auto-discovery handles schema extraction errors (lines 143-146)."""
        server_config = {"server": "test_server"}

        # Create a mock tool where accessing inputSchema raises an exception
        mock_tool = Mock()
        mock_tool.name = "tool1"
        mock_tool.description = "Test tool"
        type(mock_tool).inputSchema = PropertyMock(side_effect=Exception("Schema error"))

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        with (
            patch(
                "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
                return_value=mock_client,
            ),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
            patch("lionpride.services.mcps.loader.logger") as mock_logger,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool_obj = Mock()
            MockTool.return_value = mock_tool_obj

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config)

            # Should still register tool with None schema
            assert "test_server_tool1" in result
            # Verify warning was logged (lines 143-146)
            mock_logger.warning.assert_called()

    async def test_auto_discovery_tool_has_no_input_schema(self, mock_registry):
        """Test auto-discovery when tool has no inputSchema attribute."""
        server_config = {"server": "test_server"}

        mock_tool = Mock(spec=["name", "description"])  # No inputSchema
        mock_tool.name = "tool1"
        mock_tool.description = "Test tool"

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        with (
            patch(
                "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
                return_value=mock_client,
            ),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool_obj = Mock()
            MockTool.return_value = mock_tool_obj

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config)

            # Should register with None schema
            assert "test_server_tool1" in result
            # tool_schema should be None
            tool_call_kwargs = MockTool.call_args.kwargs
            assert tool_call_kwargs.get("tool_schema") is None

    async def test_auto_discovery_tool_input_schema_not_dict(self, mock_registry):
        """Test auto-discovery when inputSchema is not a dict."""
        server_config = {"server": "test_server"}

        mock_tool = Mock()
        mock_tool.name = "tool1"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = "not a dict"  # Invalid schema type

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        with (
            patch(
                "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
                return_value=mock_client,
            ),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            mock_tool_obj = Mock()
            MockTool.return_value = mock_tool_obj

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config)

            # Should register with None schema (schema validation failed)
            assert "test_server_tool1" in result

    async def test_auto_discovery_already_registered_skips(self, mock_registry):
        """Test auto-discovery skips already registered tools."""
        mock_registry.has = Mock(return_value=True)
        server_config = {"server": "test_server"}

        mock_tool = Mock()
        mock_tool.name = "tool1"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        with (
            patch(
                "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
                return_value=mock_client,
            ),
            patch("lionpride.services.mcps.loader.logger") as mock_logger,
        ):
            result = await load_mcp_tools(mock_registry, server_config)

            # Should skip registration
            assert result == []
            mock_logger.warning.assert_called()

    async def test_auto_discovery_tool_registration_fails(self, mock_registry):
        """Test auto-discovery handles tool registration failures (lines 172-173)."""
        server_config = {"server": "test_server"}

        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1"
        mock_tool1.inputSchema = {"type": "object"}

        mock_tool2 = Mock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Tool 2"
        mock_tool2.inputSchema = {"type": "object"}

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        with (
            patch(
                "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
                return_value=mock_client,
            ),
            patch("lionpride.services.mcps.loader.create_mcp_callable") as mock_create_callable,
            patch("lionpride.services.Tool") as MockTool,
            patch("lionpride.services.iModel") as MockiModel,
            patch("lionpride.services.mcps.loader.logger") as mock_logger,
        ):
            mock_callable = Mock()
            mock_create_callable.return_value = mock_callable

            # First tool succeeds, second fails
            mock_tool_obj = Mock()
            MockTool.side_effect = [mock_tool_obj, Exception("Registration failed")]

            mock_model = Mock()
            MockiModel.return_value = mock_model

            result = await load_mcp_tools(mock_registry, server_config)

            # Should register only successful tools (lines 172-173 covered)
            assert "test_server_tool1" in result
            assert "test_server_tool2" not in result
            mock_logger.warning.assert_called()


# =============================================================================
# create_mcp_callable Tests
# =============================================================================


class TestCreateMCPCallable:
    """Test create_mcp_callable function."""

    async def test_create_callable_basic(self):
        """Test creating basic MCP callable."""
        server_config = {"server": "test_server"}
        tool_name = "test_tool"

        callable_func = create_mcp_callable(server_config, tool_name)

        assert callable_func.__name__ == tool_name
        assert callable(callable_func)

    @pytest.mark.parametrize(
        "result_type,call_kwargs",
        [
            ("text_content", {"arg1": "value1"}),
            ("dict_text", {}),
            ("list_dict", {}),
            ("no_match", {}),
            ("no_content", {}),
            ("empty_content", {}),
            ("multiple_items", {}),
        ],
        ids=[
            "text_content",
            "dict_text_content",
            "list_dict_result",
            "content_list_no_match",
            "no_content_attribute",
            "empty_content_list",
            "multiple_content_items",
        ],
    )
    async def test_callable_execution_content_patterns(
        self, mock_mcp_client, patch_mcp_pool, result_type, call_kwargs
    ):
        """Test callable execution with various result content patterns.

        Parametrized test covering 7 content extraction scenarios:
        - text_content: content[0].text
        - dict_text: content[0]["text"]
        - list_dict: result[0]["text"] (no content attr)
        - no_match: return full content list
        - no_content: return raw result
        - empty_content: return empty list
        - multiple_items: return full content list
        """
        server_config = {"server": "test_server"}
        tool_name = "test_tool"

        # Create mock result and compute expected value based on type
        if result_type == "text_content":
            mock_result = _create_text_content_mock("result text")
            expected = "result text"
        elif result_type == "dict_text":
            mock_result = Mock()
            mock_result.content = [{"type": "text", "text": "dict text result"}]
            expected = "dict text result"
        elif result_type == "list_dict":
            mock_result = [{"type": "text", "text": "list dict text"}]
            expected = "list dict text"
        elif result_type == "no_match":
            mock_result = Mock()
            mock_result.content = [{"type": "other", "data": "something"}]
            expected = mock_result.content
        elif result_type == "no_content":
            mock_result = {"data": "raw result"}
            expected = mock_result
        elif result_type == "empty_content":
            mock_result = Mock()
            mock_result.content = []
            expected = []
        elif result_type == "multiple_items":
            mock_result = Mock()
            mock_item1 = Mock()
            mock_item1.text = "item1"
            mock_item2 = Mock()
            mock_item2.text = "item2"
            mock_result.content = [mock_item1, mock_item2]
            expected = mock_result.content

        mock_mcp_client.call_tool = AsyncMock(return_value=mock_result)

        with patch_mcp_pool:
            callable_func = create_mcp_callable(server_config, tool_name)
            result = await callable_func(**call_kwargs)

            # Verify result matches expected
            assert result == expected
            # Verify mock was called correctly (all cases)
            mock_mcp_client.call_tool.assert_called_once_with(tool_name, call_kwargs)


# =============================================================================
# load_mcp_config Tests
# =============================================================================


class TestLoadMCPConfig:
    """Test load_mcp_config function."""

    async def test_load_config_all_servers(self, mock_registry, tmp_path):
        """Test loading all servers from config file."""
        config_file = tmp_path / ".mcp.json"
        config_data = {
            "mcpServers": {
                "server1": {"command": "python", "args": ["-m", "server1"]},
                "server2": {"command": "python", "args": ["-m", "server2"]},
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Mock successful tool loading
        with patch(
            "lionpride.services.mcps.loader.load_mcp_tools",
            return_value=["tool1", "tool2"],
        ) as mock_load_tools:
            result = await load_mcp_config(mock_registry, str(config_file))

            assert "server1" in result
            assert "server2" in result
            assert result["server1"] == ["tool1", "tool2"]
            assert result["server2"] == ["tool1", "tool2"]
            assert mock_load_tools.call_count == 2

    async def test_load_config_specific_servers(self, mock_registry, tmp_path):
        """Test loading specific servers only."""
        config_file = tmp_path / ".mcp.json"
        config_data = {
            "mcpServers": {
                "server1": {"command": "python"},
                "server2": {"command": "python"},
                "server3": {"command": "python"},
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch(
            "lionpride.services.mcps.loader.load_mcp_tools",
            return_value=["tool1"],
        ) as mock_load_tools:
            result = await load_mcp_config(
                mock_registry, str(config_file), server_names=["server1", "server3"]
            )

            assert "server1" in result
            assert "server3" in result
            assert "server2" not in result
            assert mock_load_tools.call_count == 2

    async def test_load_config_server_registration_fails(self, mock_registry, tmp_path):
        """Test load_config handles server registration failures (lines 277-279)."""
        config_file = tmp_path / ".mcp.json"
        config_data = {
            "mcpServers": {
                "server1": {"command": "python"},
                "server2": {"command": "python"},
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Mock first server succeeds, second fails
        async def mock_load_side_effect(registry, config, update=False):
            if config["server"] == "server1":
                return ["tool1"]
            else:
                raise Exception("Server 2 failed")

        with (
            patch(
                "lionpride.services.mcps.loader.load_mcp_tools",
                side_effect=mock_load_side_effect,
            ),
            patch("lionpride.services.mcps.loader.logger") as mock_logger,
        ):
            result = await load_mcp_config(mock_registry, str(config_file))

            # Lines 277-279 covered
            assert result["server1"] == ["tool1"]
            assert result["server2"] == []  # Failed server returns empty list
            mock_logger.error.assert_called()

    async def test_load_config_with_update_flag(self, mock_registry, tmp_path):
        """Test load_config passes update flag to load_mcp_tools."""
        config_file = tmp_path / ".mcp.json"
        config_data = {"mcpServers": {"server1": {"command": "python"}}}
        config_file.write_text(json.dumps(config_data))

        with patch(
            "lionpride.services.mcps.loader.load_mcp_tools",
            return_value=["tool1"],
        ) as mock_load_tools:
            _result = await load_mcp_config(mock_registry, str(config_file), update=True)

            # Verify update=True was passed
            mock_load_tools.assert_called_once()
            call_kwargs = mock_load_tools.call_args.kwargs
            assert call_kwargs["update"] is True

    async def test_load_config_empty_servers(self, mock_registry, tmp_path):
        """Test load_config with no servers in config."""
        config_file = tmp_path / ".mcp.json"
        config_data = {"mcpServers": {}}
        config_file.write_text(json.dumps(config_data))

        result = await load_mcp_config(mock_registry, str(config_file))

        # Should return empty dict
        assert result == {}
