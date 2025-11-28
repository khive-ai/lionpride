# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage tests for MCPConnectionPool.

Target: 100% branch coverage for src/lionpride/services/mcps/wrapper.py

Coverage areas:
- load_config: file loading, validation, error handling
- get_client: server reference, inline config, caching, reconnection
- _create_client: URL connection, command connection, env handling
- cleanup: connection cleanup with error handling
- Context manager: __aenter__/__aexit__
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, mock_open, patch

import pytest

from lionpride.services.mcps.wrapper import MCPConnectionPool


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset MCPConnectionPool state before each test."""
    MCPConnectionPool._clients.clear()
    MCPConnectionPool._configs.clear()
    yield
    MCPConnectionPool._clients.clear()
    MCPConnectionPool._configs.clear()


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestMCPConnectionPoolContextManager:
    """Test context manager protocol."""

    async def test_context_manager_entry(self):
        """Test __aenter__ returns self."""
        pool = MCPConnectionPool()
        result = await pool.__aenter__()
        assert result is pool

    async def test_context_manager_exit_calls_cleanup(self):
        """Test __aexit__ calls cleanup."""
        pool = MCPConnectionPool()

        with patch.object(MCPConnectionPool, "cleanup", new_callable=AsyncMock) as mock_cleanup:
            await pool.__aexit__(None, None, None)
            mock_cleanup.assert_called_once()


# =============================================================================
# load_config Tests
# =============================================================================


class TestLoadConfig:
    """Test load_config method."""

    def test_load_config_file_not_found(self, tmp_path):
        """Test load_config raises FileNotFoundError when file doesn't exist."""
        non_existent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="MCP config file not found"):
            MCPConnectionPool.load_config(str(non_existent))

    def test_load_config_invalid_json(self, tmp_path):
        """Test load_config raises ValueError for invalid JSON."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text("{invalid json")

        with pytest.raises(ValueError, match="Invalid JSON in MCP config file"):
            MCPConnectionPool.load_config(str(config_file))

    def test_load_config_not_a_dict(self, tmp_path):
        """Test load_config raises ValueError when root is not a dict."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text('["not", "a", "dict"]')

        with pytest.raises(ValueError, match="MCP config must be a JSON object"):
            MCPConnectionPool.load_config(str(config_file))

    def test_load_config_mcp_servers_not_dict(self, tmp_path):
        """Test load_config raises ValueError when mcpServers is not a dict."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text('{"mcpServers": ["not", "a", "dict"]}')

        with pytest.raises(ValueError, match="mcpServers must be a dictionary"):
            MCPConnectionPool.load_config(str(config_file))

    def test_load_config_success(self, tmp_path):
        """Test load_config successfully loads valid config."""
        config_file = tmp_path / ".mcp.json"
        config_data = {
            "mcpServers": {
                "server1": {"command": "python", "args": ["-m", "server1"]},
                "server2": {"url": "http://localhost:8000"},
            }
        }
        config_file.write_text(json.dumps(config_data))

        MCPConnectionPool.load_config(str(config_file))

        assert "server1" in MCPConnectionPool._configs
        assert "server2" in MCPConnectionPool._configs
        assert MCPConnectionPool._configs["server1"]["command"] == "python"
        assert MCPConnectionPool._configs["server2"]["url"] == "http://localhost:8000"

    def test_load_config_updates_existing(self, tmp_path):
        """Test load_config updates existing configs."""
        # Load first config
        config_file1 = tmp_path / ".mcp1.json"
        config_file1.write_text('{"mcpServers": {"server1": {"command": "test1"}}}')
        MCPConnectionPool.load_config(str(config_file1))

        # Load second config
        config_file2 = tmp_path / ".mcp2.json"
        config_file2.write_text('{"mcpServers": {"server2": {"command": "test2"}}}')
        MCPConnectionPool.load_config(str(config_file2))

        # Both should be present
        assert "server1" in MCPConnectionPool._configs
        assert "server2" in MCPConnectionPool._configs


# =============================================================================
# get_client Tests
# =============================================================================


class TestGetClient:
    """Test get_client method."""

    async def test_get_client_server_reference_not_found(self):
        """Test get_client raises ValueError for unknown server reference."""
        MCPConnectionPool._configs = {}

        # Mock load_config to do nothing (simulating no config file)
        with (
            patch.object(MCPConnectionPool, "load_config"),
            pytest.raises(ValueError, match="Unknown MCP server"),
        ):
            await MCPConnectionPool.get_client({"server": "nonexistent"})

    async def test_get_client_server_reference_auto_load(self, tmp_path):
        """Test get_client auto-loads config if server not found initially."""
        # Create config file
        config_file = tmp_path / ".mcp.json"
        config_data = {"mcpServers": {"testserver": {"command": "python"}}}
        config_file.write_text(json.dumps(config_data))

        mock_client = Mock()
        mock_client.is_connected = Mock(return_value=True)

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(config_data)),
            patch.object(MCPConnectionPool, "_create_client", return_value=mock_client),
        ):
            client = await MCPConnectionPool.get_client({"server": "testserver"})
            assert client is mock_client

    async def test_get_client_cached_client_still_connected(self):
        """Test get_client returns cached client if still connected."""
        MCPConnectionPool._configs = {"testserver": {"command": "python"}}

        mock_client = Mock()
        mock_client.is_connected = Mock(return_value=True)

        # Pre-populate cache
        MCPConnectionPool._clients["server:testserver"] = mock_client

        client = await MCPConnectionPool.get_client({"server": "testserver"})
        assert client is mock_client

    async def test_get_client_cached_client_disconnected(self):
        """Test get_client recreates client if cached client is disconnected."""
        MCPConnectionPool._configs = {"testserver": {"command": "python"}}

        old_client = Mock()
        old_client.is_connected = Mock(return_value=False)

        new_client = Mock()
        new_client.is_connected = Mock(return_value=True)

        # Pre-populate cache with disconnected client
        MCPConnectionPool._clients["server:testserver"] = old_client

        with patch.object(MCPConnectionPool, "_create_client", return_value=new_client):
            client = await MCPConnectionPool.get_client({"server": "testserver"})
            assert client is new_client
            assert "server:testserver" in MCPConnectionPool._clients

    async def test_get_client_cached_client_no_is_connected_method(self):
        """Test get_client handles client without is_connected method."""
        MCPConnectionPool._configs = {"testserver": {"command": "python"}}

        old_client = Mock(spec=[])  # No is_connected method

        new_client = Mock()
        new_client.is_connected = Mock(return_value=True)

        # Pre-populate cache
        MCPConnectionPool._clients["server:testserver"] = old_client

        with patch.object(MCPConnectionPool, "_create_client", return_value=new_client):
            client = await MCPConnectionPool.get_client({"server": "testserver"})
            assert client is new_client

    async def test_get_client_inline_config(self):
        """Test get_client with inline config (no server reference)."""
        inline_config = {"command": "python", "args": ["-m", "server"]}

        mock_client = Mock()
        mock_client.is_connected = Mock(return_value=True)

        with patch.object(MCPConnectionPool, "_create_client", return_value=mock_client):
            client = await MCPConnectionPool.get_client(inline_config)
            assert client is mock_client


# =============================================================================
# _create_client Tests
# =============================================================================

# Check if fastmcp is available for patching
try:
    import fastmcp

    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False


class TestCreateClient:
    """Test _create_client method."""

    async def test_create_client_not_a_dict(self):
        """Test _create_client raises ValueError for non-dict config."""
        with pytest.raises(ValueError, match="Config must be a dictionary"):
            await MCPConnectionPool._create_client("not a dict")

    async def test_create_client_no_url_or_command(self):
        """Test _create_client raises ValueError when neither url nor command present."""
        with pytest.raises(
            ValueError, match="Config must have either 'url' or 'command' with non-None value"
        ):
            await MCPConnectionPool._create_client({"invalid": "config"})

    async def test_create_client_fastmcp_not_installed(self):
        """Test _create_client raises ImportError when fastmcp not installed."""
        with (
            patch("builtins.__import__", side_effect=ImportError("No module named 'fastmcp'")),
            pytest.raises(ImportError, match="FastMCP not installed"),
        ):
            await MCPConnectionPool._create_client({"url": "http://localhost:8000"})

    @pytest.mark.skipif(not HAS_FASTMCP, reason="fastmcp not installed")
    async def test_create_client_url_connection(self):
        """Test _create_client creates URL-based connection."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with patch("fastmcp.Client", return_value=mock_client) as MockClient:
            client = await MCPConnectionPool._create_client({"url": "http://localhost:8000"})

            MockClient.assert_called_once_with("http://localhost:8000")
            mock_client.__aenter__.assert_called_once()
            assert client is mock_client

    @pytest.mark.skipif(not HAS_FASTMCP, reason="fastmcp not installed")
    async def test_create_client_command_connection_basic(self):
        """Test _create_client creates command-based connection."""
        mock_transport = Mock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with (
            patch(
                "fastmcp.client.transports.StdioTransport", return_value=mock_transport
            ) as MockTransport,
            patch("fastmcp.Client", return_value=mock_client) as MockClient,
        ):
            config = {"command": "python", "args": ["-m", "server"]}
            await MCPConnectionPool._create_client(config)

            # Verify transport creation
            MockTransport.assert_called_once()
            call_kwargs = MockTransport.call_args.kwargs
            assert call_kwargs["command"] == "python"
            assert call_kwargs["args"] == ["-m", "server"]
            assert "env" in call_kwargs

            # Verify client creation
            MockClient.assert_called_once_with(mock_transport)
            mock_client.__aenter__.assert_called_once()

    @pytest.mark.skipif(not HAS_FASTMCP, reason="fastmcp not installed")
    async def test_create_client_command_invalid_args(self):
        """Test _create_client raises ValueError for non-list args."""
        with pytest.raises(ValueError, match="Config 'args' must be a list"):
            await MCPConnectionPool._create_client({"command": "python", "args": "not a list"})

    @pytest.mark.skipif(not HAS_FASTMCP, reason="fastmcp not installed")
    async def test_create_client_command_with_env_vars(self):
        """Test _create_client merges environment variables."""
        mock_transport = Mock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with (
            patch(
                "fastmcp.client.transports.StdioTransport", return_value=mock_transport
            ) as MockTransport,
            patch("fastmcp.Client", return_value=mock_client),
            patch.dict(os.environ, {"EXISTING_VAR": "existing"}),
        ):
            config = {"command": "python", "env": {"CUSTOM_VAR": "custom"}}
            await MCPConnectionPool._create_client(config)

            call_kwargs = MockTransport.call_args.kwargs
            env = call_kwargs["env"]
            assert env["EXISTING_VAR"] == "existing"
            assert env["CUSTOM_VAR"] == "custom"
            assert env["LOG_LEVEL"] == "ERROR"  # Default suppression

    @pytest.mark.skipif(not HAS_FASTMCP, reason="fastmcp not installed")
    async def test_create_client_command_debug_mode(self):
        """Test _create_client doesn't suppress logs in debug mode."""
        mock_transport = Mock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with (
            patch(
                "fastmcp.client.transports.StdioTransport", return_value=mock_transport
            ) as MockTransport,
            patch("fastmcp.Client", return_value=mock_client),
        ):
            config = {"command": "python", "debug": True}
            await MCPConnectionPool._create_client(config)

            call_kwargs = MockTransport.call_args.kwargs
            env = call_kwargs["env"]
            # Should not have default suppression vars when debug=True
            assert "LOG_LEVEL" not in env or env.get("LOG_LEVEL") != "ERROR"

    @pytest.mark.skipif(not HAS_FASTMCP, reason="fastmcp not installed")
    async def test_create_client_command_mcp_debug_env(self):
        """Test _create_client respects MCP_DEBUG environment variable."""
        mock_transport = Mock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with (
            patch(
                "fastmcp.client.transports.StdioTransport", return_value=mock_transport
            ) as MockTransport,
            patch("fastmcp.Client", return_value=mock_client),
            patch.dict(os.environ, {"MCP_DEBUG": "true"}),
        ):
            config = {"command": "python"}
            await MCPConnectionPool._create_client(config)

            call_kwargs = MockTransport.call_args.kwargs
            env = call_kwargs["env"]
            # Should not have default suppression vars when MCP_DEBUG=true
            assert "LOG_LEVEL" not in env or env.get("LOG_LEVEL") != "ERROR"


# =============================================================================
# cleanup Tests
# =============================================================================


class TestCleanup:
    """Test cleanup method."""

    async def test_cleanup_empty_pool(self):
        """Test cleanup works with no clients."""
        await MCPConnectionPool.cleanup()
        assert len(MCPConnectionPool._clients) == 0

    async def test_cleanup_closes_all_clients(self):
        """Test cleanup calls __aexit__ on all clients."""
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()

        MCPConnectionPool._clients = {
            "client1": mock_client1,
            "client2": mock_client2,
        }

        await MCPConnectionPool.cleanup()

        mock_client1.__aexit__.assert_called_once_with(None, None, None)
        mock_client2.__aexit__.assert_called_once_with(None, None, None)
        assert len(MCPConnectionPool._clients) == 0

    async def test_cleanup_handles_client_errors(self):
        """Test cleanup continues on client errors."""
        mock_client1 = AsyncMock()
        mock_client1.__aexit__ = AsyncMock(side_effect=Exception("Client 1 error"))

        mock_client2 = AsyncMock()

        MCPConnectionPool._clients = {
            "client1": mock_client1,
            "client2": mock_client2,
        }

        with patch("logging.debug") as mock_log:
            await MCPConnectionPool.cleanup()

            # Should log error but continue cleanup
            mock_log.assert_called()
            assert "Client 1 error" in str(mock_log.call_args)

            # Both clients should be attempted
            mock_client1.__aexit__.assert_called_once()
            mock_client2.__aexit__.assert_called_once()

            # Pool should be cleared
            assert len(MCPConnectionPool._clients) == 0
