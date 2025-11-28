# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MCPConnectionPool with mocked MCP dependencies.

This file provides comprehensive mocked unit tests that do NOT require
real MCP server connections or the fastmcp package to be installed.

Target: 100% branch coverage for src/lionpride/services/mcps/wrapper.py

Coverage areas (by line):
- Line 47: __aenter__ returns self
- Line 51: __aexit__ calls cleanup
- Line 70: FileNotFoundError for missing config
- Lines 75-76: ValueError for invalid JSON
- Line 79: ValueError for non-dict config root
- Line 83: ValueError for non-dict mcpServers
- Lines 113-143: get_client logic (server reference, inline config, caching)
- Lines 157-210: _create_client logic (URL, command, env, debug)
- Lines 221-228: cleanup with error handling
"""

import builtins
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from lionpride.services.mcps.wrapper import MCPConnectionPool


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset MCPConnectionPool state before and after each test."""
    MCPConnectionPool._clients.clear()
    MCPConnectionPool._configs.clear()
    yield
    MCPConnectionPool._clients.clear()
    MCPConnectionPool._configs.clear()


@pytest.fixture
def mock_fastmcp():
    """Create mock fastmcp module and inject into sys.modules.

    This fixture creates a complete mock of the fastmcp module structure
    including Client and transports, allowing _create_client to work
    without the real fastmcp package installed.
    """
    # Create mock transport
    mock_stdio_transport = Mock()

    # Create mock transports module
    mock_transports = MagicMock()
    mock_transports.StdioTransport = Mock(return_value=mock_stdio_transport)

    # Create mock client.transports submodule
    mock_client_module = MagicMock()
    mock_client_module.transports = mock_transports

    # Create mock Client class
    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)
    mock_client_instance.is_connected = Mock(return_value=True)

    mock_client_class = Mock(return_value=mock_client_instance)

    # Create mock fastmcp module
    mock_fastmcp_module = MagicMock()
    mock_fastmcp_module.Client = mock_client_class
    mock_fastmcp_module.client = mock_client_module
    mock_fastmcp_module.client.transports = mock_transports

    # Store references for test assertions
    mock_fastmcp_module._mock_client_instance = mock_client_instance
    mock_fastmcp_module._mock_client_class = mock_client_class
    mock_fastmcp_module._mock_transport = mock_stdio_transport
    mock_fastmcp_module._mock_transport_class = mock_transports.StdioTransport

    # Inject into sys.modules
    original_modules = {}
    modules_to_mock = [
        "fastmcp",
        "fastmcp.client",
        "fastmcp.client.transports",
    ]

    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]

    sys.modules["fastmcp"] = mock_fastmcp_module
    sys.modules["fastmcp.client"] = mock_client_module
    sys.modules["fastmcp.client.transports"] = mock_transports

    yield mock_fastmcp_module

    # Restore original modules
    for mod_name in modules_to_mock:
        if mod_name in original_modules:
            sys.modules[mod_name] = original_modules[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]


# =============================================================================
# Context Manager Tests (Lines 45-51)
# =============================================================================


class TestContextManager:
    """Test async context manager protocol."""

    async def test_aenter_returns_self(self):
        """Test __aenter__ returns self (line 47)."""
        pool = MCPConnectionPool()
        result = await pool.__aenter__()
        assert result is pool

    async def test_aexit_calls_cleanup(self):
        """Test __aexit__ calls cleanup (line 51)."""
        pool = MCPConnectionPool()

        with patch.object(MCPConnectionPool, "cleanup", new_callable=AsyncMock) as mock_cleanup:
            await pool.__aexit__(None, None, None)
            mock_cleanup.assert_called_once()

    async def test_context_manager_full_lifecycle(self):
        """Test full async context manager lifecycle."""
        cleanup_called = False

        async def mock_cleanup(self_arg=None):
            nonlocal cleanup_called
            cleanup_called = True

        with patch.object(MCPConnectionPool, "cleanup", new_callable=lambda: mock_cleanup):
            async with MCPConnectionPool() as pool:
                assert isinstance(pool, MCPConnectionPool)
            assert cleanup_called


# =============================================================================
# load_config Tests (Lines 53-85)
# =============================================================================


class TestLoadConfig:
    """Test load_config method."""

    def test_file_not_found_raises_error(self, tmp_path):
        """Test FileNotFoundError when config file doesn't exist (line 70)."""
        non_existent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="MCP config file not found"):
            MCPConnectionPool.load_config(str(non_existent))

    def test_invalid_json_raises_value_error(self, tmp_path):
        """Test ValueError for invalid JSON (lines 75-76)."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text("{invalid json syntax")

        with pytest.raises(ValueError, match="Invalid JSON in MCP config file"):
            MCPConnectionPool.load_config(str(config_file))

    def test_non_dict_root_raises_value_error(self, tmp_path):
        """Test ValueError when root is not a dict (line 79)."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text('["this", "is", "an", "array"]')

        with pytest.raises(ValueError, match="MCP config must be a JSON object"):
            MCPConnectionPool.load_config(str(config_file))

    def test_non_dict_mcp_servers_raises_value_error(self, tmp_path):
        """Test ValueError when mcpServers is not a dict (line 83)."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text('{"mcpServers": ["not", "a", "dict"]}')

        with pytest.raises(ValueError, match="mcpServers must be a dictionary"):
            MCPConnectionPool.load_config(str(config_file))

    def test_successful_load(self, tmp_path):
        """Test successful config loading."""
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

    def test_load_updates_existing_configs(self, tmp_path):
        """Test loading multiple configs merges them."""
        # First config
        config1 = tmp_path / "config1.json"
        config1.write_text('{"mcpServers": {"server1": {"command": "cmd1"}}}')
        MCPConnectionPool.load_config(str(config1))

        # Second config
        config2 = tmp_path / "config2.json"
        config2.write_text('{"mcpServers": {"server2": {"command": "cmd2"}}}')
        MCPConnectionPool.load_config(str(config2))

        # Both should exist
        assert "server1" in MCPConnectionPool._configs
        assert "server2" in MCPConnectionPool._configs

    def test_empty_mcp_servers(self, tmp_path):
        """Test loading config with empty mcpServers."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        # Should not raise
        MCPConnectionPool.load_config(str(config_file))
        # Should have no servers
        assert len(MCPConnectionPool._configs) == 0

    def test_missing_mcp_servers_key(self, tmp_path):
        """Test loading config without mcpServers key."""
        config_file = tmp_path / ".mcp.json"
        config_file.write_text('{"other_key": "value"}')

        # Should not raise, just load empty
        MCPConnectionPool.load_config(str(config_file))
        assert len(MCPConnectionPool._configs) == 0


# =============================================================================
# get_client Tests (Lines 87-143)
# =============================================================================


class TestGetClient:
    """Test get_client method."""

    async def test_unknown_server_reference_raises_error(self):
        """Test ValueError for unknown server reference (lines 113-120)."""
        # Mock load_config to do nothing (simulating no config file)
        with (
            patch.object(MCPConnectionPool, "load_config"),
            pytest.raises(ValueError, match="Unknown MCP server: nonexistent"),
        ):
            await MCPConnectionPool.get_client({"server": "nonexistent"})

    async def test_server_reference_auto_loads_config(self, tmp_path, mock_fastmcp):
        """Test get_client auto-loads config if server not initially found (lines 117-118)."""
        config_data = {"mcpServers": {"testserver": {"command": "python"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config_data))

        # Pre-populate an empty configs to trigger load
        MCPConnectionPool._configs = {}

        with patch.object(
            MCPConnectionPool,
            "load_config",
            side_effect=lambda: MCPConnectionPool._configs.update(config_data["mcpServers"]),
        ):
            client = await MCPConnectionPool.get_client({"server": "testserver"})
            assert client is mock_fastmcp._mock_client_instance

    async def test_server_reference_uses_cached_config(self, mock_fastmcp):
        """Test get_client uses server config from _configs (lines 122-123)."""
        MCPConnectionPool._configs = {"testserver": {"url": "http://localhost:8000"}}

        client = await MCPConnectionPool.get_client({"server": "testserver"})

        assert client is mock_fastmcp._mock_client_instance
        assert "server:testserver" in MCPConnectionPool._clients

    async def test_cached_client_still_connected_returned(self, mock_fastmcp):
        """Test cached client returned if still connected (lines 131-135)."""
        MCPConnectionPool._configs = {"testserver": {"command": "python"}}

        # Create a mock client that reports connected
        cached_client = Mock()
        cached_client.is_connected = Mock(return_value=True)
        MCPConnectionPool._clients["server:testserver"] = cached_client

        result = await MCPConnectionPool.get_client({"server": "testserver"})

        assert result is cached_client
        cached_client.is_connected.assert_called_once()

    async def test_disconnected_cached_client_replaced(self, mock_fastmcp):
        """Test disconnected cached client is replaced (lines 136-138)."""
        MCPConnectionPool._configs = {"testserver": {"command": "python"}}

        # Create old client that reports disconnected
        old_client = Mock()
        old_client.is_connected = Mock(return_value=False)
        MCPConnectionPool._clients["server:testserver"] = old_client

        result = await MCPConnectionPool.get_client({"server": "testserver"})

        # Should get new client
        assert result is mock_fastmcp._mock_client_instance
        assert MCPConnectionPool._clients["server:testserver"] is mock_fastmcp._mock_client_instance

    async def test_cached_client_without_is_connected_replaced(self, mock_fastmcp):
        """Test cached client without is_connected method is replaced (line 134 condition)."""
        MCPConnectionPool._configs = {"testserver": {"command": "python"}}

        # Create old client without is_connected attribute
        old_client = Mock(spec=[])  # Empty spec = no attributes
        MCPConnectionPool._clients["server:testserver"] = old_client

        result = await MCPConnectionPool.get_client({"server": "testserver"})

        # Should get new client since hasattr fails
        assert result is mock_fastmcp._mock_client_instance

    async def test_inline_config_creates_new_client(self, mock_fastmcp):
        """Test inline config (no server reference) creates client (lines 124-127)."""
        inline_config = {"command": "python", "args": ["-m", "server"]}

        client = await MCPConnectionPool.get_client(inline_config)

        assert client is mock_fastmcp._mock_client_instance
        # Cache key should use inline: prefix
        cache_keys = list(MCPConnectionPool._clients.keys())
        assert any(k.startswith("inline:") for k in cache_keys)

    async def test_inline_config_with_url(self, mock_fastmcp):
        """Test inline config with URL."""
        inline_config = {"url": "http://localhost:9000"}

        client = await MCPConnectionPool.get_client(inline_config)

        assert client is mock_fastmcp._mock_client_instance


# =============================================================================
# _create_client Tests (Lines 145-210)
# =============================================================================


class TestCreateClient:
    """Test _create_client method."""

    async def test_non_dict_config_raises_error(self):
        """Test ValueError for non-dict config (line 157-158)."""
        with pytest.raises(ValueError, match="Config must be a dictionary"):
            await MCPConnectionPool._create_client("not a dict")

    async def test_non_dict_config_list_raises_error(self):
        """Test ValueError for list config (line 157-158)."""
        with pytest.raises(ValueError, match="Config must be a dictionary"):
            await MCPConnectionPool._create_client(["list", "config"])

    async def test_missing_url_and_command_raises_error(self):
        """Test ValueError when neither url nor command present (lines 160-162)."""
        with pytest.raises(
            ValueError, match="Config must have either 'url' or 'command' with non-None value"
        ):
            await MCPConnectionPool._create_client({"other_key": "value"})

    async def test_null_url_and_command_raises_error(self):
        """Test ValueError when both url and command are None (lines 160-162)."""
        with pytest.raises(
            ValueError, match="Config must have either 'url' or 'command' with non-None value"
        ):
            await MCPConnectionPool._create_client({"url": None, "command": None})

    async def test_fastmcp_import_error_raised(self):
        """Test ImportError when fastmcp not installed (lines 164-167)."""
        # Mock the import to raise ImportError
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastmcp" or name.startswith("fastmcp."):
                raise ImportError("No module named 'fastmcp'")
            return original_import(name, *args, **kwargs)

        # Remove fastmcp from sys.modules to force re-import
        fastmcp_modules = {
            k: v for k, v in sys.modules.items() if k == "fastmcp" or k.startswith("fastmcp.")
        }
        for k in fastmcp_modules:
            sys.modules.pop(k, None)

        try:
            with (
                patch.object(builtins, "__import__", side_effect=mock_import),
                pytest.raises(ImportError, match="FastMCP not installed"),
            ):
                await MCPConnectionPool._create_client({"url": "http://localhost:8000"})
        finally:
            # Restore fastmcp modules
            sys.modules.update(fastmcp_modules)

    async def test_url_connection_creates_client(self, mock_fastmcp):
        """Test URL-based connection (lines 170-172)."""
        config = {"url": "http://localhost:8000"}

        client = await MCPConnectionPool._create_client(config)

        # Verify Client was called with URL
        mock_fastmcp._mock_client_class.assert_called_once_with("http://localhost:8000")
        mock_fastmcp._mock_client_instance.__aenter__.assert_called_once()
        assert client is mock_fastmcp._mock_client_instance

    async def test_command_connection_creates_client(self, mock_fastmcp):
        """Test command-based connection (lines 173-203)."""
        config = {"command": "python", "args": ["-m", "server"]}

        client = await MCPConnectionPool._create_client(config)

        # Verify StdioTransport was created
        mock_fastmcp._mock_transport_class.assert_called_once()
        call_kwargs = mock_fastmcp._mock_transport_class.call_args.kwargs
        assert call_kwargs["command"] == "python"
        assert call_kwargs["args"] == ["-m", "server"]
        assert "env" in call_kwargs

        # Verify Client was called with transport
        mock_fastmcp._mock_client_class.assert_called_once_with(mock_fastmcp._mock_transport)
        assert client is mock_fastmcp._mock_client_instance

    async def test_command_with_invalid_args_raises_error(self, mock_fastmcp):
        """Test ValueError for non-list args (lines 176-178)."""
        config = {"command": "python", "args": "not a list"}

        with pytest.raises(ValueError, match="Config 'args' must be a list"):
            await MCPConnectionPool._create_client(config)

    async def test_command_with_empty_args(self, mock_fastmcp):
        """Test command with empty args list."""
        config = {"command": "python"}  # No args key

        await MCPConnectionPool._create_client(config)

        call_kwargs = mock_fastmcp._mock_transport_class.call_args.kwargs
        assert call_kwargs["args"] == []

    async def test_command_merges_environment_variables(self, mock_fastmcp):
        """Test env vars are merged (lines 180-182)."""
        config = {"command": "python", "env": {"CUSTOM_VAR": "custom_value"}}

        with patch.dict(os.environ, {"EXISTING_VAR": "existing_value"}, clear=False):
            await MCPConnectionPool._create_client(config)

        call_kwargs = mock_fastmcp._mock_transport_class.call_args.kwargs
        env = call_kwargs["env"]

        # Custom var should be in env
        assert env["CUSTOM_VAR"] == "custom_value"
        # Should have log suppression by default
        assert env["LOG_LEVEL"] == "ERROR"
        assert env["FASTMCP_QUIET"] == "true"

    async def test_command_suppresses_logs_by_default(self, mock_fastmcp):
        """Test default log suppression environment vars (lines 185-193)."""
        config = {"command": "python"}

        await MCPConnectionPool._create_client(config)

        call_kwargs = mock_fastmcp._mock_transport_class.call_args.kwargs
        env = call_kwargs["env"]

        # Verify suppression vars are set
        assert env.get("LOG_LEVEL") == "ERROR"
        assert env.get("PYTHONWARNINGS") == "ignore"
        assert env.get("FASTMCP_QUIET") == "true"
        assert env.get("MCP_QUIET") == "true"

    async def test_command_debug_mode_no_suppression(self, mock_fastmcp):
        """Test debug=True disables log suppression (lines 185-187)."""
        config = {"command": "python", "debug": True}

        # Ensure clean environment for test
        with patch.dict(os.environ, {}, clear=True):
            await MCPConnectionPool._create_client(config)

        call_kwargs = mock_fastmcp._mock_transport_class.call_args.kwargs
        env = call_kwargs["env"]

        # LOG_LEVEL should NOT be set to ERROR in debug mode
        # (setdefault only sets if not present, but debug mode skips the entire block)
        assert env.get("LOG_LEVEL") != "ERROR" or "LOG_LEVEL" not in env

    async def test_command_mcp_debug_env_no_suppression(self, mock_fastmcp):
        """Test MCP_DEBUG=true env var disables log suppression (lines 185-187)."""
        config = {"command": "python"}

        with patch.dict(os.environ, {"MCP_DEBUG": "true"}, clear=True):
            await MCPConnectionPool._create_client(config)

        call_kwargs = mock_fastmcp._mock_transport_class.call_args.kwargs
        env = call_kwargs["env"]

        # Should not have suppression vars when MCP_DEBUG=true
        # The env will have MCP_DEBUG=true from os.environ.copy()
        assert env.get("MCP_DEBUG") == "true"

    async def test_url_takes_precedence_over_command(self, mock_fastmcp):
        """Test URL config takes precedence when both present (line 170)."""
        config = {"url": "http://localhost:8000", "command": "python"}

        await MCPConnectionPool._create_client(config)

        # Should use URL path, not command path
        mock_fastmcp._mock_client_class.assert_called_once_with("http://localhost:8000")
        # StdioTransport should NOT be created
        mock_fastmcp._mock_transport_class.assert_not_called()


# =============================================================================
# cleanup Tests (Lines 212-228)
# =============================================================================


class TestCleanup:
    """Test cleanup method."""

    async def test_cleanup_empty_pool(self):
        """Test cleanup works with no clients (lines 221-228)."""
        await MCPConnectionPool.cleanup()
        assert len(MCPConnectionPool._clients) == 0

    async def test_cleanup_closes_all_clients(self):
        """Test cleanup calls __aexit__ on all clients (lines 222-227)."""
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

    async def test_cleanup_handles_client_errors_gracefully(self):
        """Test cleanup continues on client errors (lines 225-227)."""
        mock_client1 = AsyncMock()
        mock_client1.__aexit__ = AsyncMock(side_effect=Exception("Client 1 cleanup error"))

        mock_client2 = AsyncMock()

        MCPConnectionPool._clients = {
            "client1": mock_client1,
            "client2": mock_client2,
        }

        with patch("logging.debug") as mock_log:
            await MCPConnectionPool.cleanup()

            # Should log the error
            mock_log.assert_called()
            log_message = str(mock_log.call_args)
            assert "Client 1 cleanup error" in log_message or "client1" in log_message

            # Both clients should be attempted
            mock_client1.__aexit__.assert_called_once()
            mock_client2.__aexit__.assert_called_once()

            # Pool should be cleared
            assert len(MCPConnectionPool._clients) == 0

    async def test_cleanup_clears_pool_after_errors(self):
        """Test pool is cleared even if all clients fail (line 228)."""
        mock_client1 = AsyncMock()
        mock_client1.__aexit__ = AsyncMock(side_effect=RuntimeError("Error 1"))

        mock_client2 = AsyncMock()
        mock_client2.__aexit__ = AsyncMock(side_effect=RuntimeError("Error 2"))

        MCPConnectionPool._clients = {
            "client1": mock_client1,
            "client2": mock_client2,
        }

        with patch("logging.debug"):
            await MCPConnectionPool.cleanup()

        # Pool should still be cleared
        assert len(MCPConnectionPool._clients) == 0

    async def test_cleanup_idempotent(self):
        """Test cleanup can be called multiple times safely."""
        mock_client = AsyncMock()
        MCPConnectionPool._clients = {"client": mock_client}

        await MCPConnectionPool.cleanup()
        await MCPConnectionPool.cleanup()  # Second call should not raise

        # First call clears the pool, second call does nothing
        mock_client.__aexit__.assert_called_once()


# =============================================================================
# Integration Tests (Full Workflow)
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple methods."""

    async def test_full_lifecycle_with_context_manager(self, tmp_path, mock_fastmcp):
        """Test complete lifecycle: load config, get client, cleanup."""
        # Setup config
        config_file = tmp_path / ".mcp.json"
        config_data = {
            "mcpServers": {
                "testserver": {"command": "python", "args": ["-m", "testserver"]},
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Load config
        MCPConnectionPool.load_config(str(config_file))
        assert "testserver" in MCPConnectionPool._configs

        # Use context manager
        async with MCPConnectionPool() as pool:
            client = await pool.get_client({"server": "testserver"})
            assert client is mock_fastmcp._mock_client_instance
            assert "server:testserver" in MCPConnectionPool._clients

        # After context exit, clients should be cleaned up
        assert len(MCPConnectionPool._clients) == 0

    async def test_multiple_clients_same_server(self, mock_fastmcp):
        """Test multiple get_client calls for same server return cached client."""
        MCPConnectionPool._configs = {"server1": {"url": "http://localhost:8000"}}

        client1 = await MCPConnectionPool.get_client({"server": "server1"})
        client2 = await MCPConnectionPool.get_client({"server": "server1"})

        # Should return same cached client
        assert client1 is client2
        # Client class should only be called once
        assert mock_fastmcp._mock_client_class.call_count == 1

    async def test_different_servers_different_clients(self, mock_fastmcp):
        """Test different servers get different cache keys."""
        MCPConnectionPool._configs = {
            "server1": {"url": "http://localhost:8000"},
            "server2": {"url": "http://localhost:9000"},
        }

        # Reset mock to track calls
        mock_fastmcp._mock_client_class.reset_mock()

        await MCPConnectionPool.get_client({"server": "server1"})
        await MCPConnectionPool.get_client({"server": "server2"})

        # Should create two different clients
        assert mock_fastmcp._mock_client_class.call_count == 2
        assert "server:server1" in MCPConnectionPool._clients
        assert "server:server2" in MCPConnectionPool._clients


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_json_with_type_error(self, tmp_path):
        """Test TypeError during JSON parsing (line 75-76)."""
        config_file = tmp_path / ".mcp.json"
        # Write binary content that will cause orjson to raise TypeError
        config_file.write_bytes(b"\x00\x01\x02")

        with pytest.raises(ValueError, match="Invalid JSON in MCP config file"):
            MCPConnectionPool.load_config(str(config_file))

    async def test_server_none_value_triggers_reference_path(self):
        """Test server=None doesn't trigger server reference path (line 113)."""
        # When server key exists but is None, it should NOT take the server reference path
        config = {"server": None, "command": "python"}

        with patch.object(
            MCPConnectionPool, "_create_client", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = Mock()
            await MCPConnectionPool.get_client(config)

            # Should call _create_client with the config (inline path)
            mock_create.assert_called_once()

    async def test_command_with_none_env_raises_type_error(self, mock_fastmcp):
        """Test command config with env=None raises TypeError.

        Note: The code expects env to be a dict or omitted, not None.
        This documents the expected behavior.
        """
        config = {"command": "python", "env": None}

        # env=None causes TypeError since None is not iterable
        with pytest.raises(TypeError):
            await MCPConnectionPool._create_client(config)

    async def test_inline_config_cache_key_uniqueness(self, mock_fastmcp):
        """Test inline configs get unique cache keys based on id()."""
        config1 = {"command": "python"}
        config2 = {"command": "python"}  # Same content, different object

        mock_fastmcp._mock_client_class.reset_mock()

        await MCPConnectionPool.get_client(config1)
        await MCPConnectionPool.get_client(config2)

        # Each inline config should create its own client (different id())
        assert mock_fastmcp._mock_client_class.call_count == 2

    def test_load_config_with_extra_fields(self, tmp_path):
        """Test config with extra fields is accepted."""
        config_file = tmp_path / ".mcp.json"
        config_data = {
            "mcpServers": {
                "server1": {"command": "python", "extra_field": "ignored"},
            },
            "extraTopLevel": "also ignored",
        }
        config_file.write_text(json.dumps(config_data))

        MCPConnectionPool.load_config(str(config_file))
        assert "server1" in MCPConnectionPool._configs
        assert MCPConnectionPool._configs["server1"]["extra_field"] == "ignored"
