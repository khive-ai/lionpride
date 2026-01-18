# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for act.py coverage - tool execution and action handling.

Covers:
- act() function execution paths
- Result extraction from various response formats
- execute_tools() helper
- has_action_requests() helper
- Action message persistence
"""

import pytest
from pydantic import BaseModel

from lionpride.errors import AccessError, NotFoundError, ValidationError
from lionpride.operations.operate.act import act, execute_tools, has_action_requests
from lionpride.rules import ActionRequest, ActionResponse
from lionpride.services import iModel
from lionpride.services.types.tool import Tool, ToolConfig
from lionpride.session import Session

# =============================================================================
# Test Tools
# =============================================================================


async def mock_multiply(a: int, b: int) -> int:
    """Mock multiply tool."""
    return a * b


async def mock_adder(a: int, b: int) -> int:
    """Mock adder tool."""
    return a + b


async def mock_error_tool(message: str) -> str:
    """Mock tool that always raises error."""
    raise ValueError(f"Test error: {message}")


def sync_tool(value: str) -> str:
    """Mock synchronous tool."""
    return f"sync: {value}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def session_with_tools():
    """Create session with test tools registered."""
    session = Session()
    tool_names = []

    tools = [
        Tool(
            func_callable=mock_multiply,
            config=ToolConfig(name="multiply", provider="tool"),
        ),
        Tool(func_callable=mock_adder, config=ToolConfig(name="adder", provider="tool")),
        Tool(
            func_callable=mock_error_tool,
            config=ToolConfig(name="error_tool", provider="tool"),
        ),
        Tool(
            func_callable=sync_tool,
            config=ToolConfig(name="sync_tool", provider="tool"),
        ),
    ]

    for tool in tools:
        session.services.register(iModel(backend=tool))
        tool_names.append(tool.name)

    return session, tool_names


# =============================================================================
# Test act() Function
# =============================================================================


class TestAct:
    """Test act() function for tool execution."""

    async def test_empty_requests_returns_empty_list(self, session_with_tools):
        """Test that empty request list returns empty response list."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        responses = await act([], session, branch)

        assert responses == []

    async def test_missing_function_name_raises_not_found_error(self, session_with_tools):
        """Test that empty function string raises NotFoundError (service not found)."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [ActionRequest(function="", arguments={})]

        with pytest.raises(NotFoundError, match="not found in session"):
            await act(requests, session, branch)

    async def test_tool_not_in_registry_raises_not_found_error(self, session_with_tools):
        """Test that tool not in registry raises NotFoundError."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names) | {"nonexistent"})

        requests = [ActionRequest(function="nonexistent", arguments={})]

        with pytest.raises(NotFoundError, match="'nonexistent' not found in session"):
            await act(requests, session, branch)

    async def test_tool_not_in_branch_resources_raises_access_error(self, session_with_tools):
        """Test that tool not in branch resources raises AccessError."""
        session, _tool_names = session_with_tools
        branch = session.create_branch(resources={"multiply"})  # Only multiply

        requests = [ActionRequest(function="adder", arguments={"a": 1, "b": 2})]

        with pytest.raises(AccessError, match="has no access to resource 'adder'"):
            await act(requests, session, branch)

    async def test_single_tool_execution_success(self, session_with_tools):
        """Test successful execution of single tool."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [ActionRequest(function="multiply", arguments={"a": 5, "b": 3})]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert responses[0].output == 15
        assert responses[0].function == "multiply"

    async def test_multiple_tools_concurrent_execution(self, session_with_tools):
        """Test concurrent execution of multiple tools."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequest(function="adder", arguments={"a": 10, "b": 20}),
        ]

        responses = await act(requests, session, branch, max_concurrent=10)

        assert len(responses) == 2
        assert responses[0].output == 15
        assert responses[1].output == 30

    async def test_error_handling_returns_error_response(self, session_with_tools):
        """Test that tool errors are caught and returned as error responses."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [ActionRequest(function="error_tool", arguments={"message": "test"})]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert "ValueError" in responses[0].output

    async def test_sync_tool_execution(self, session_with_tools):
        """Test execution of synchronous tool."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [ActionRequest(function="sync_tool", arguments={"value": "hello"})]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert responses[0].output == "sync: hello"

    async def test_mixed_success_and_error(self, session_with_tools):
        """Test execution with mix of success and error."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 2, "b": 3}),
            ActionRequest(function="error_tool", arguments={"message": "fail"}),
            ActionRequest(function="adder", arguments={"a": 1, "b": 1}),
        ]

        responses = await act(requests, session, branch)

        assert len(responses) == 3
        assert responses[0].output == 6
        assert "ValueError" in responses[1].output
        assert responses[2].output == 2


# =============================================================================
# Test execute_tools()
# =============================================================================


class TestExecuteTools:
    """Test execute_tools helper function."""

    async def test_execute_tools_with_action_requests(self, session_with_tools):
        """Test execute_tools updates response with action_responses."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        class ResponseModel(BaseModel):
            answer: str
            action_requests: list[ActionRequest] | None = None
            action_responses: list[ActionResponse] | None = None

        response = ResponseModel(
            answer="test",
            action_requests=[ActionRequest(function="multiply", arguments={"a": 3, "b": 4})],
        )

        updated, action_responses = await execute_tools(response, session, branch)

        assert len(action_responses) == 1
        assert action_responses[0].output == 12
        assert updated.action_responses is not None
        assert len(updated.action_responses) == 1

    async def test_execute_tools_no_action_requests(self, session_with_tools):
        """Test execute_tools with no action_requests returns unchanged."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        class ResponseModel(BaseModel):
            answer: str

        response = ResponseModel(answer="test")

        updated, action_responses = await execute_tools(response, session, branch)

        assert action_responses == []
        assert updated is response

    async def test_execute_tools_empty_action_requests(self, session_with_tools):
        """Test execute_tools with empty action_requests list."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        class ResponseModel(BaseModel):
            answer: str
            action_requests: list[ActionRequest] | None = None

        response = ResponseModel(answer="test", action_requests=[])

        updated, action_responses = await execute_tools(response, session, branch)

        assert action_responses == []
        assert updated is response

    async def test_execute_tools_model_copy(self, session_with_tools):
        """Test execute_tools uses model_copy for Pydantic models."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        class ResponseWithCopy(BaseModel):
            value: str
            action_requests: list[ActionRequest] | None = None
            action_responses: list[ActionResponse] | None = None

        response = ResponseWithCopy(
            value="test",
            action_requests=[ActionRequest(function="multiply", arguments={"a": 2, "b": 5})],
        )

        updated, _action_responses = await execute_tools(response, session, branch)

        assert updated.action_responses is not None
        assert updated.action_responses[0].output == 10


# =============================================================================
# Test has_action_requests()
# =============================================================================


class TestHasActionRequests:
    """Test has_action_requests helper function."""

    def test_has_action_requests_true(self):
        """Test returns True when action_requests present."""

        class WithActions(BaseModel):
            action_requests: list[ActionRequest] | None = None

        response = WithActions(action_requests=[ActionRequest(function="test", arguments={})])
        assert has_action_requests(response) is True

    def test_has_action_requests_false_none(self):
        """Test returns False when action_requests is None."""

        class WithActions(BaseModel):
            action_requests: list[ActionRequest] | None = None

        assert has_action_requests(WithActions()) is False

    def test_has_action_requests_false_empty(self):
        """Test returns False when action_requests is empty list."""

        class WithActions(BaseModel):
            action_requests: list[ActionRequest] | None = None

        assert has_action_requests(WithActions(action_requests=[])) is False

    def test_has_action_requests_no_attribute(self):
        """Test returns False when no action_requests attribute."""

        class NoActions(BaseModel):
            value: str

        assert has_action_requests(NoActions(value="test")) is False


# =============================================================================
# Test _persist_action_messages()
# =============================================================================


class TestPersistActionMessages:
    """Test action message persistence."""

    async def test_messages_persisted_to_branch(self, session_with_tools):
        """Test that action messages are persisted to branch."""
        session, tool_names = session_with_tools
        branch = session.create_branch(name="test", resources=set(tool_names))

        requests = [ActionRequest(function="multiply", arguments={"a": 5, "b": 3})]

        await act(requests, session, branch)

        # Should have 2 messages: request and response
        messages = session.messages[branch]
        assert len(messages) == 2

        # First is request (branch -> tool)
        req_msg = messages[0]
        assert req_msg.sender == branch.id
        assert req_msg.recipient == "multiply"

        # Second is response (tool -> branch)
        resp_msg = messages[1]
        assert resp_msg.sender == "multiply"
        assert resp_msg.recipient == branch.id

    async def test_multiple_actions_persist_all_messages(self, session_with_tools):
        """Test that multiple actions persist all messages."""
        session, tool_names = session_with_tools
        branch = session.create_branch(name="test", resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 2, "b": 3}),
            ActionRequest(function="adder", arguments={"a": 4, "b": 5}),
        ]

        await act(requests, session, branch)

        # Should have 4 messages: 2 request + 2 response
        messages = session.messages[branch]
        assert len(messages) == 4


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestActEdgeCases:
    """Test edge cases for act()."""

    async def test_default_arguments(self, session_with_tools):
        """Test that default arguments (empty dict) works."""
        session, tool_names = session_with_tools

        async def no_args_tool() -> str:
            return "no args"

        tool = Tool(
            func_callable=no_args_tool,
            config=ToolConfig(name="no_args", provider="tool"),
        )
        session.services.register(iModel(backend=tool))
        branch = session.create_branch(resources=set(tool_names) | {"no_args"})

        requests = [ActionRequest(function="no_args")]  # No arguments

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert responses[0].output == "no args"

    async def test_sequential_execution(self, session_with_tools):
        """Test sequential execution with max_concurrent=1."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 2, "b": 3}),
            ActionRequest(function="adder", arguments={"a": 4, "b": 5}),
        ]

        responses = await act(requests, session, branch, max_concurrent=1)

        assert len(responses) == 2
        assert responses[0].output == 6
        assert responses[1].output == 9

    async def test_retry_timeout_parameter(self, session_with_tools):
        """Test retry_timeout parameter is accepted."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [ActionRequest(function="multiply", arguments={"a": 2, "b": 3})]

        responses = await act(requests, session, branch, retry_timeout=5.0)

        assert len(responses) == 1
        assert responses[0].output == 6

    async def test_throttle_period_parameter(self, session_with_tools):
        """Test throttle_period parameter is accepted."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [ActionRequest(function="multiply", arguments={"a": 2, "b": 3})]

        responses = await act(requests, session, branch, throttle_period=0.01)

        assert len(responses) == 1
        assert responses[0].output == 6


class TestExecuteToolsFallbackPath:
    """Test execute_tools model_dump fallback path (lines 170-172)."""

    async def test_execute_tools_model_dump_fallback(self, session_with_tools):
        """Test execute_tools uses model_dump when model_copy not available."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        # Create a mock that simulates no model_copy
        class MockResponse:
            """Mock response without model_copy."""

            def __init__(self):
                self.value = "test"
                self.action_requests = [
                    ActionRequest(function="multiply", arguments={"a": 2, "b": 5})
                ]
                self.action_responses = None

            def model_dump(self):
                return {
                    "value": self.value,
                    "action_requests": self.action_requests,
                    "action_responses": self.action_responses,
                }

            @classmethod
            def model_validate(cls, data):
                obj = cls()
                obj.value = data.get("value")
                obj.action_requests = data.get("action_requests")
                obj.action_responses = data.get("action_responses")
                return obj

        response = MockResponse()

        updated, _action_responses = await execute_tools(response, session, branch)
        assert updated.action_responses is not None
        assert updated.action_responses[0].output == 10
