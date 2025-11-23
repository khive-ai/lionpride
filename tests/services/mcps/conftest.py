# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for MCP loader tests."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_mcp_client():
    """Create mock FastMCP client with common setup.

    Returns AsyncMock client that can be customized per-test:
    - list_tools: Mock for tool discovery
    - call_tool: Mock for tool execution

    Example:
        def test_something(mock_mcp_client):
            mock_mcp_client.list_tools = AsyncMock(return_value=[...])
            with patch_mcp_pool(mock_mcp_client):
                # test code
    """
    client = AsyncMock()
    # Default mocks (can be overridden per-test)
    client.list_tools = AsyncMock(return_value=[])
    client.call_tool = AsyncMock(return_value=None)
    return client


@pytest.fixture
def patch_mcp_pool(mock_mcp_client):
    """Patch MCPConnectionPool.get_client to return mock_mcp_client.

    Usage:
        def test_something(mock_mcp_client, patch_mcp_pool):
            # Customize mock_mcp_client for this test
            mock_mcp_client.list_tools = AsyncMock(return_value=[...])

            with patch_mcp_pool:
                # MCPConnectionPool.get_client now returns mock_mcp_client
                result = await load_mcp_tools(...)
    """
    return patch(
        "lionpride.services.mcps.wrapper.MCPConnectionPool.get_client",
        return_value=mock_mcp_client,
    )
