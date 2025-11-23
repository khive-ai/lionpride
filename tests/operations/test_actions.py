# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for action execution (tool calling)."""

import asyncio

import pytest

from lionpride.operations.actions import act
from lionpride.operations.models import ActionRequestModel, ActionResponseModel
from lionpride.services import ServiceRegistry, iModel
from lionpride.services.types.backend import NormalizedResponse
from lionpride.services.types.tool import Tool, ToolConfig


# Test Tools
async def mock_multiply(a: int, b: int) -> int:
    """Mock multiply tool."""
    return a * b


async def mock_add_tool(a: int, b: int) -> int:
    """Mock add tool."""
    return a + b


async def mock_error_tool(message: str) -> str:
    """Mock tool that always raises error."""
    raise ValueError(f"Test error: {message}")


async def mock_slow_tool(delay: float) -> str:
    """Mock tool with delay."""
    await asyncio.sleep(delay)
    return "completed"


def sync_tool(value: str) -> str:
    """Mock synchronous tool."""
    return f"sync: {value}"


class TestAct:
    """Test act() function for tool execution."""

    @pytest.fixture
    def registry(self) -> ServiceRegistry:
        """Create registry with test tools."""
        registry = ServiceRegistry()

        # Register tools (wrapped in iModel)
        multiply_tool = Tool(
            func_callable=mock_multiply,
            config=ToolConfig(name="multiply", provider="tool"),
        )
        add_tool = Tool(
            func_callable=mock_add_tool,
            config=ToolConfig(name="add_tool", provider="tool"),
        )
        error_tool = Tool(
            func_callable=mock_error_tool,
            config=ToolConfig(name="error_tool", provider="tool"),
        )
        slow_tool = Tool(
            func_callable=mock_slow_tool,
            config=ToolConfig(name="slow_tool", provider="tool"),
        )
        sync_tool_obj = Tool(
            func_callable=sync_tool,
            config=ToolConfig(name="sync_tool", provider="tool"),
        )

        # Wrap tools in iModel before registering
        registry.register(iModel(backend=multiply_tool))
        registry.register(iModel(backend=add_tool))
        registry.register(iModel(backend=error_tool))
        registry.register(iModel(backend=slow_tool))
        registry.register(iModel(backend=sync_tool_obj))

        return registry

    async def test_empty_requests_returns_empty(self, registry: ServiceRegistry):
        """Test that empty request list returns empty response list."""
        requests = []
        responses = await act(requests, registry)

        assert len(responses) == 0
        assert isinstance(responses, list)

    async def test_missing_function_name_raises_error(self, registry: ServiceRegistry):
        """Test that missing function name raises ValueError."""
        requests = [
            ActionRequestModel(function=None, arguments={"a": 5, "b": 3}),  # type: ignore[arg-type]
        ]

        with pytest.raises(ValueError, match=r"Action request missing function name"):
            await act(requests, registry)

    async def test_tool_not_in_registry_raises_error(self, registry: ServiceRegistry):
        """Test that tool not in registry raises ValueError with available tools."""
        requests = [
            ActionRequestModel(function="nonexistent_tool", arguments={"a": 5}),
        ]

        with pytest.raises(ValueError, match=r"Tool 'nonexistent_tool' not found in registry"):
            await act(requests, registry)

        # Verify error message includes available tools
        with pytest.raises(ValueError, match=r"Available tools"):
            await act(requests, registry)

    async def test_concurrent_execution_success(self, registry: ServiceRegistry):
        """Test concurrent execution of multiple tools."""
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequestModel(function="add_tool", arguments={"a": 10, "b": 20}),
            ActionRequestModel(function="multiply", arguments={"a": 7, "b": 2}),
        ]

        responses = await act(requests, registry, concurrent=True)

        assert len(responses) == 3
        assert all(isinstance(r, ActionResponseModel) for r in responses)

        # Verify results
        assert responses[0].function == "multiply"
        assert responses[0].arguments == {"a": 5, "b": 3}
        assert responses[0].output == 15

        assert responses[1].function == "add_tool"
        assert responses[1].arguments == {"a": 10, "b": 20}
        assert responses[1].output == 30

        assert responses[2].function == "multiply"
        assert responses[2].arguments == {"a": 7, "b": 2}
        assert responses[2].output == 14

    async def test_sequential_execution_success(self, registry: ServiceRegistry):
        """Test sequential execution of multiple tools."""
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequestModel(function="add_tool", arguments={"a": 10, "b": 20}),
        ]

        responses = await act(requests, registry, concurrent=False)

        assert len(responses) == 2
        assert responses[0].output == 15
        assert responses[1].output == 30

    async def test_single_tool_execution(self, registry: ServiceRegistry):
        """Test execution of single tool."""
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 6, "b": 7}),
        ]

        responses = await act(requests, registry)

        assert len(responses) == 1
        assert responses[0].output == 42
        assert responses[0].function == "multiply"
        assert responses[0].arguments == {"a": 6, "b": 7}

    async def test_none_arguments_defaults_to_empty_dict(self, registry: ServiceRegistry):
        """Test that None arguments are converted to empty dict."""

        async def no_arg_tool() -> str:
            return "no args needed"

        # Register tool that takes no args (wrapped in iModel)
        tool = Tool(
            func_callable=no_arg_tool,
            config=ToolConfig(name="no_arg_tool", provider="tool"),
        )
        registry.register(iModel(backend=tool))

        requests = [
            ActionRequestModel(function="no_arg_tool", arguments=None),
        ]

        responses = await act(requests, registry)

        assert len(responses) == 1
        assert responses[0].output == "no args needed"

    async def test_error_handling_returns_error_response(self, registry: ServiceRegistry):
        """Test that tool errors are caught and returned as error responses."""
        requests = [
            ActionRequestModel(function="error_tool", arguments={"message": "test error"}),
        ]

        responses = await act(requests, registry)

        assert len(responses) == 1
        assert responses[0].function == "error_tool"
        assert "ValueError: Test error: test error" in responses[0].output

    async def test_mixed_success_and_error_concurrent(self, registry: ServiceRegistry):
        """Test concurrent execution with mix of success and error."""
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequestModel(function="error_tool", arguments={"message": "fail"}),
            ActionRequestModel(function="add_tool", arguments={"a": 10, "b": 20}),
        ]

        responses = await act(requests, registry, concurrent=True)

        assert len(responses) == 3

        # First request succeeds
        assert responses[0].output == 15

        # Second request has error
        assert "ValueError: Test error: fail" in responses[1].output

        # Third request succeeds
        assert responses[2].output == 30

    async def test_mixed_success_and_error_sequential(self, registry: ServiceRegistry):
        """Test sequential execution with mix of success and error."""
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequestModel(function="error_tool", arguments={"message": "fail"}),
            ActionRequestModel(function="add_tool", arguments={"a": 10, "b": 20}),
        ]

        responses = await act(requests, registry, concurrent=False)

        assert len(responses) == 3

        # First request succeeds
        assert responses[0].output == 15

        # Second request has error
        assert "ValueError: Test error: fail" in responses[1].output

        # Third request succeeds (execution continues despite error)
        assert responses[2].output == 30

    async def test_timeout_handling(self, registry: ServiceRegistry):
        """Test timeout handling for slow tools."""
        requests = [
            ActionRequestModel(function="slow_tool", arguments={"delay": 2.0}),
        ]

        # Set very short timeout
        responses = await act(requests, registry, timeout=0.1)

        assert len(responses) == 1
        # Timeout should result in error response
        assert "TimeoutError" in responses[0].output or "CancelledError" in responses[0].output

    async def test_timeout_not_triggered_for_fast_tool(self, registry: ServiceRegistry):
        """Test that timeout doesn't affect fast tools."""
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 5, "b": 3}),
        ]

        # Set generous timeout
        responses = await act(requests, registry, timeout=5.0)

        assert len(responses) == 1
        assert responses[0].output == 15

    async def test_sync_tool_execution(self, registry: ServiceRegistry):
        """Test execution of synchronous tool."""
        requests = [
            ActionRequestModel(function="sync_tool", arguments={"value": "test"}),
        ]

        responses = await act(requests, registry)

        assert len(responses) == 1
        assert responses[0].output == "sync: test"

    async def test_normalized_response_data_extraction(self, registry: ServiceRegistry):
        """Test extraction of data from NormalizedResponse.

        Tool.call() always wraps results in NormalizedResponse.
        This tests that act() extracts the .data field correctly.
        """

        # Tool returns raw dict (Tool.call() will wrap it in NormalizedResponse)
        async def response_tool(value: str) -> dict:
            return {"result": value}

        tool = Tool(
            func_callable=response_tool,
            config=ToolConfig(name="response_tool", provider="tool"),
        )
        registry.register(iModel(backend=tool))

        requests = [
            ActionRequestModel(function="response_tool", arguments={"value": "test"}),
        ]

        responses = await act(requests, registry)

        assert len(responses) == 1
        # Should extract .data from NormalizedResponse returned by Tool.call()
        assert responses[0].output == {"result": "test"}

    async def test_multiple_errors_all_returned(self, registry: ServiceRegistry):
        """Test that all errors are captured and returned."""
        requests = [
            ActionRequestModel(function="error_tool", arguments={"message": "error1"}),
            ActionRequestModel(function="error_tool", arguments={"message": "error2"}),
            ActionRequestModel(function="error_tool", arguments={"message": "error3"}),
        ]

        responses = await act(requests, registry, concurrent=True)

        assert len(responses) == 3
        assert all("ValueError" in r.output for r in responses)
        assert "error1" in responses[0].output
        assert "error2" in responses[1].output
        assert "error3" in responses[2].output

    async def test_action_response_model_structure(self, registry: ServiceRegistry):
        """Test that ActionResponseModel is properly constructed."""
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 5, "b": 3}),
        ]

        responses = await act(requests, registry)

        response = responses[0]
        assert isinstance(response, ActionResponseModel)
        assert response.function == "multiply"
        assert response.arguments == {"a": 5, "b": 3}
        assert response.output == 15

        # Verify it's a Pydantic model
        assert hasattr(response, "model_dump")
        data = response.model_dump()
        assert data["function"] == "multiply"
        assert data["arguments"] == {"a": 5, "b": 3}
        assert data["output"] == 15

    async def test_concurrent_execution_order_independence(self, registry: ServiceRegistry):
        """Test that concurrent execution doesn't depend on request order."""
        # Create requests with varying execution times (but all fast enough)
        requests = [
            ActionRequestModel(function="multiply", arguments={"a": 1, "b": 1}),
            ActionRequestModel(function="add_tool", arguments={"a": 2, "b": 2}),
            ActionRequestModel(function="multiply", arguments={"a": 3, "b": 3}),
        ]

        responses = await act(requests, registry, concurrent=True)

        # Results should match request order (gather preserves order)
        assert len(responses) == 3
        assert responses[0].output == 1  # 1 * 1
        assert responses[1].output == 4  # 2 + 2
        assert responses[2].output == 9  # 3 * 3
