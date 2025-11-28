# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for MCP loader with real MCP fetch server.

Tests real MCP server integration using Anthropic's official fetch server:
- npx -y @modelcontextprotocol/server-fetch

These tests verify:
- Real MCP protocol communication
- Tool discovery from actual server
- Request/response handling with live server
- URL fetching capability
"""

import shutil

import pytest

from lionpride.services.mcps.loader import load_mcp_tools
from lionpride.services.types import ServiceRegistry

# Check if fastmcp is available
try:
    import fastmcp

    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False

# Skip integration tests if uvx not available or fastmcp not installed
pytestmark = pytest.mark.skipif(
    not shutil.which("uvx") or not HAS_FASTMCP,
    reason="uvx not available or fastmcp not installed - required for MCP tests",
)


@pytest.fixture
def registry():
    """Create a real ServiceRegistry for integration tests."""
    return ServiceRegistry()


@pytest.fixture
def fetch_server_config():
    """Configuration for Anthropic MCP fetch server.

    Uses the official mcp-server-fetch Python package via uvx.
    Server provides 'fetch' tool for URL content retrieval.
    """
    return {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
    }


@pytest.mark.timeout(30)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_load_mcp_tools_with_fetch_server_auto_discovery(registry, fetch_server_config):
    """Test auto-discovery of tools from real MCP fetch server.

    Verifies:
    - Connection to real MCP server via npx
    - Auto-discovery of 'fetch' tool
    - Tool registration in ServiceRegistry
    """
    # Auto-discover tools from fetch server (can take a few seconds)
    registered_tools = await load_mcp_tools(
        registry=registry,
        server_config=fetch_server_config,
        tool_names=None,  # Auto-discover all tools
    )

    # Verify fetch tool was discovered and registered
    assert len(registered_tools) > 0, "Should discover at least one tool from fetch server"
    assert "fetch" in registered_tools, "fetch tool should be auto-discovered"

    # Verify tool is in registry
    assert registry.has("fetch"), "fetch tool should be registered in ServiceRegistry"


@pytest.mark.timeout(30)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_load_mcp_tools_with_fetch_server_specific_tool(registry, fetch_server_config):
    """Test loading specific 'fetch' tool from MCP server.

    Verifies:
    - Specific tool registration (not auto-discovery)
    - Tool name qualification
    - Registry integration
    """
    # Load specific fetch tool
    registered_tools = await load_mcp_tools(
        registry=registry,
        server_config=fetch_server_config,
        tool_names=["fetch"],  # Request specific tool
    )

    assert len(registered_tools) == 1, "Should register exactly one tool"
    assert "fetch" in registered_tools, "fetch tool should be registered"
    assert registry.has("fetch"), "fetch tool should be in registry"


@pytest.mark.timeout(30)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_fetch_tool_invocation_real_url(registry, fetch_server_config):
    """Test invoking fetch tool with real URL.

    Fetches from: https://modelcontextprotocol.io/docs/getting-started/intro
    Verifies:
    - Real HTTP request execution
    - Content retrieval
    - Response format
    """
    # Load fetch tool
    await load_mcp_tools(
        registry=registry,
        server_config=fetch_server_config,
        tool_names=["fetch"],
    )

    # Get the fetch tool from registry
    fetch_model = registry.get("fetch")
    assert fetch_model is not None, "fetch tool should be retrievable from registry"

    # Invoke fetch tool with MCP docs URL
    result = await fetch_model.invoke(
        url="https://modelcontextprotocol.io/docs/getting-started/intro"
    )

    # Verify response structure - result is a ToolCalling object
    assert result is not None, "fetch should return a response"

    # The actual MCP response is in calling.execution.response
    assert hasattr(result, "execution"), "ToolCalling should have execution attribute"
    assert hasattr(result.execution, "response"), "Execution should have response attribute"

    # Verify we got content from the fetch
    response = result.execution.response
    assert response is not None, "Should have response data from MCP fetch server"

    # Verify actual content from MCP docs page
    response_text = str(response)
    assert "Model Context Protocol" in response_text, (
        "Response should contain 'Model Context Protocol' from MCP docs"
    )
    assert "open-source standard" in response_text, (
        "Response should contain 'open-source standard' from MCP docs"
    )


@pytest.mark.timeout(30)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_fetch_server_multiple_urls(registry, fetch_server_config):
    """Test fetch tool with multiple different URLs.

    Verifies:
    - Multiple invocations work correctly
    - Different URLs return different content
    """
    # Load fetch tool
    await load_mcp_tools(
        registry=registry,
        server_config=fetch_server_config,
        tool_names=["fetch"],
    )

    fetch_model = registry.get("fetch")
    assert fetch_model is not None, "fetch tool should be registered"

    # Fetch from two different MCP doc pages
    result1 = await fetch_model.invoke(
        url="https://modelcontextprotocol.io/docs/getting-started/intro"
    )
    result2 = await fetch_model.invoke(url="https://modelcontextprotocol.io/docs/core-features")

    assert result1 is not None and result2 is not None, "Both fetches should succeed"

    # Both should have execution.response with fetched content
    assert result1.execution.response is not None, "First fetch should have response"
    assert result2.execution.response is not None, "Second fetch should have response"

    # Verify actual content from fetched pages
    response1_text = str(result1.execution.response)
    response2_text = str(result2.execution.response)

    # First fetch from /intro should contain MCP documentation content
    assert "Model Context Protocol" in response1_text, (
        "First fetch should contain 'Model Context Protocol'"
    )

    # Second fetch from /core-features should contain different content
    assert len(response2_text) > 100, "Second fetch should contain substantial content"

    # Responses should be different (different pages)
    assert response1_text != response2_text, "Different URLs should return different content"


@pytest.mark.timeout(30)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_fetch_server_with_update_flag(registry, fetch_server_config):
    """Test updating already registered fetch tool.

    Verifies:
    - update=True allows re-registration
    - Tool replacement in registry
    """
    # Initial registration
    await load_mcp_tools(
        registry=registry,
        server_config=fetch_server_config,
        tool_names=["fetch"],
    )

    # Re-register with update=True (should succeed)
    registered_tools = await load_mcp_tools(
        registry=registry,
        server_config=fetch_server_config,
        tool_names=["fetch"],
        update=True,
    )

    assert "fetch" in registered_tools, "fetch tool should be re-registered with update=True"
    assert registry.has("fetch"), "fetch tool should still be in registry after update"
