# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for action execution (tool calling).

New API: act(requests, session, branch, *, max_concurrent=10, ...)
- session: contains registered tools
- branch: validates resource access
"""

import asyncio

import pytest

from lionpride.errors import AccessError, NotFoundError, ValidationError
from lionpride.operations.operate.act import act, execute_tools, has_action_requests
from lionpride.rules import ActionRequest, ActionResponse
from lionpride.services import iModel
from lionpride.services.types.tool import Tool, ToolConfig
from lionpride.session import Session


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
    def session_with_tools(self) -> tuple[Session, list[str]]:
        """Create session with test tools registered."""
        session = Session()

        # Tool names for branch resources
        tool_names = []

        # Register tools (wrapped in iModel)
        tools = [
            Tool(func_callable=mock_multiply, config=ToolConfig(name="multiply", provider="tool")),
            Tool(func_callable=mock_add_tool, config=ToolConfig(name="add_tool", provider="tool")),
            Tool(
                func_callable=mock_error_tool, config=ToolConfig(name="error_tool", provider="tool")
            ),
            Tool(
                func_callable=mock_slow_tool, config=ToolConfig(name="slow_tool", provider="tool")
            ),
            Tool(func_callable=sync_tool, config=ToolConfig(name="sync_tool", provider="tool")),
        ]

        for tool in tools:
            session.services.register(iModel(backend=tool))
            tool_names.append(tool.name)

        return session, tool_names

    async def test_empty_requests_returns_empty(self, session_with_tools):
        """Test that empty request list returns empty response list."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = []
        responses = await act(requests, session, branch)

        assert len(responses) == 0
        assert isinstance(responses, list)

    async def test_missing_function_name_raises_validation_error(self, session_with_tools):
        """Test that missing function name raises ValidationError at model construction."""
        from pydantic import ValidationError as PydanticValidationError

        # ActionRequest requires function: str - None is rejected at model level
        with pytest.raises(PydanticValidationError, match=r"function"):
            ActionRequest(function=None, arguments={"a": 5, "b": 3})  # type: ignore[arg-type]

    async def test_empty_function_name_raises_validation_error(self, session_with_tools):
        """Test that empty function string raises ValidationError."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [ActionRequest(function="", arguments={})]

        with pytest.raises(ValidationError, match="Action request missing function name"):
            await act(requests, session, branch)

    async def test_tool_not_in_registry_raises_not_found_error(self, session_with_tools):
        """Test that tool not in registry raises NotFoundError."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names) | {"nonexistent_tool"})

        requests = [
            ActionRequest(function="nonexistent_tool", arguments={"a": 5}),
        ]

        with pytest.raises(NotFoundError, match=r"Tool 'nonexistent_tool' not found"):
            await act(requests, session, branch)

    async def test_tool_not_in_branch_resources_raises_access_error(self, session_with_tools):
        """Test that tool not in branch resources raises AccessError."""
        session, _tool_names = session_with_tools
        # Branch only has access to 'multiply', not 'add_tool'
        branch = session.create_branch(resources={"multiply"})

        requests = [
            ActionRequest(function="add_tool", arguments={"a": 10, "b": 20}),
        ]

        with pytest.raises(AccessError, match=r"doesn't have access to tool 'add_tool'"):
            await act(requests, session, branch)

    async def test_concurrent_execution_success(self, session_with_tools):
        """Test concurrent execution of multiple tools."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequest(function="add_tool", arguments={"a": 10, "b": 20}),
            ActionRequest(function="multiply", arguments={"a": 7, "b": 2}),
        ]

        responses = await act(requests, session, branch, max_concurrent=10)

        assert len(responses) == 3
        assert all(isinstance(r, ActionResponse) for r in responses)

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

    async def test_sequential_execution_success(self, session_with_tools):
        """Test sequential execution of multiple tools (max_concurrent=1)."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequest(function="add_tool", arguments={"a": 10, "b": 20}),
        ]

        responses = await act(requests, session, branch, max_concurrent=1)

        assert len(responses) == 2
        assert responses[0].output == 15
        assert responses[1].output == 30

    async def test_single_tool_execution(self, session_with_tools):
        """Test execution of single tool."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 6, "b": 7}),
        ]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert responses[0].output == 42
        assert responses[0].function == "multiply"
        assert responses[0].arguments == {"a": 6, "b": 7}

    async def test_omitted_arguments_defaults_to_empty_dict(self, session_with_tools):
        """Test that omitted arguments default to empty dict."""
        session, tool_names = session_with_tools

        async def no_arg_tool() -> str:
            return "no args needed"

        # Register tool that takes no args
        tool = Tool(
            func_callable=no_arg_tool,
            config=ToolConfig(name="no_arg_tool", provider="tool"),
        )
        session.services.register(iModel(backend=tool))

        # Branch needs access to new tool
        branch = session.create_branch(resources=set(tool_names) | {"no_arg_tool"})

        # ActionRequest.arguments defaults to {} when omitted
        requests = [
            ActionRequest(function="no_arg_tool"),
        ]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert responses[0].output == "no args needed"
        assert responses[0].arguments == {}  # Default empty dict

    async def test_error_handling_returns_error_response(self, session_with_tools):
        """Test that tool errors are caught and returned as error responses."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="error_tool", arguments={"message": "test error"}),
        ]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert responses[0].function == "error_tool"
        assert "ValueError: Test error: test error" in responses[0].output

    async def test_mixed_success_and_error_concurrent(self, session_with_tools):
        """Test concurrent execution with mix of success and error."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequest(function="error_tool", arguments={"message": "fail"}),
            ActionRequest(function="add_tool", arguments={"a": 10, "b": 20}),
        ]

        responses = await act(requests, session, branch, max_concurrent=10)

        assert len(responses) == 3

        # First request succeeds
        assert responses[0].output == 15

        # Second request has error
        assert "ValueError: Test error: fail" in responses[1].output

        # Third request succeeds
        assert responses[2].output == 30

    async def test_mixed_success_and_error_sequential(self, session_with_tools):
        """Test sequential execution with mix of success and error."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
            ActionRequest(function="error_tool", arguments={"message": "fail"}),
            ActionRequest(function="add_tool", arguments={"a": 10, "b": 20}),
        ]

        responses = await act(requests, session, branch, max_concurrent=1)

        assert len(responses) == 3

        # First request succeeds
        assert responses[0].output == 15

        # Second request has error
        assert "ValueError: Test error: fail" in responses[1].output

        # Third request succeeds (execution continues despite error)
        assert responses[2].output == 30

    async def test_timeout_handling(self, session_with_tools):
        """Test timeout handling for slow tools."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="slow_tool", arguments={"delay": 2.0}),
        ]

        # Set very short timeout via retry_timeout
        responses = await act(requests, session, branch, retry_timeout=0.1)

        assert len(responses) == 1
        # Timeout should result in error response
        output = str(responses[0].output)
        assert "TimeoutError" in output or "CancelledError" in output or "asyncio" in output

    async def test_timeout_not_triggered_for_fast_tool(self, session_with_tools):
        """Test that timeout doesn't affect fast tools."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
        ]

        # Set generous timeout
        responses = await act(requests, session, branch, retry_timeout=5.0)

        assert len(responses) == 1
        assert responses[0].output == 15

    async def test_sync_tool_execution(self, session_with_tools):
        """Test execution of synchronous tool."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="sync_tool", arguments={"value": "test"}),
        ]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        assert responses[0].output == "sync: test"

    async def test_normalized_response_data_extraction(self, session_with_tools):
        """Test extraction of data from NormalizedResponse."""
        session, tool_names = session_with_tools

        # Tool returns raw dict (Tool.call() will wrap it in NormalizedResponse)
        async def response_tool(value: str) -> dict:
            return {"result": value}

        tool = Tool(
            func_callable=response_tool,
            config=ToolConfig(name="response_tool", provider="tool"),
        )
        session.services.register(iModel(backend=tool))

        branch = session.create_branch(resources=set(tool_names) | {"response_tool"})

        requests = [
            ActionRequest(function="response_tool", arguments={"value": "test"}),
        ]

        responses = await act(requests, session, branch)

        assert len(responses) == 1
        # Should extract .data from NormalizedResponse
        assert responses[0].output == {"result": "test"}

    async def test_action_response_model_structure(self, session_with_tools):
        """Test that ActionResponse is properly constructed."""
        session, tool_names = session_with_tools
        branch = session.create_branch(resources=set(tool_names))

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
        ]

        responses = await act(requests, session, branch)

        response = responses[0]
        assert isinstance(response, ActionResponse)
        assert response.function == "multiply"
        assert response.arguments == {"a": 5, "b": 3}
        assert response.output == 15

        # Verify it's a Pydantic model
        assert hasattr(response, "model_dump")
        data = response.model_dump()
        assert data["function"] == "multiply"
        assert data["arguments"] == {"a": 5, "b": 3}
        assert data["output"] == 15


class TestExecuteTools:
    """Test execute_tools helper function."""

    @pytest.fixture
    def session_with_multiply(self):
        """Create session with multiply tool."""
        session = Session()
        tool = Tool(
            func_callable=mock_multiply, config=ToolConfig(name="multiply", provider="tool")
        )
        session.services.register(iModel(backend=tool))
        return session

    async def test_execute_tools_updates_response(self, session_with_multiply):
        """Test that execute_tools updates parsed_response with action_responses."""
        from pydantic import BaseModel

        session = session_with_multiply
        branch = session.create_branch(resources={"multiply"})

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
        assert updated.action_responses[0].output == 12

    async def test_execute_tools_empty_action_requests(self, session_with_multiply):
        """Test execute_tools with no action_requests returns unchanged."""
        from pydantic import BaseModel

        session = session_with_multiply
        branch = session.create_branch(resources={"multiply"})

        class ResponseModel(BaseModel):
            answer: str

        response = ResponseModel(answer="test")

        updated, action_responses = await execute_tools(response, session, branch)

        assert action_responses == []
        assert updated is response

    async def test_has_action_requests(self, session_with_multiply):
        """Test has_action_requests helper."""
        from pydantic import BaseModel

        class WithActions(BaseModel):
            action_requests: list[ActionRequest] | None = None

        # No action_requests
        assert has_action_requests(WithActions()) is False
        assert has_action_requests(WithActions(action_requests=[])) is False

        # With action_requests
        assert (
            has_action_requests(
                WithActions(action_requests=[ActionRequest(function="test", arguments={})])
            )
            is True
        )


class TestActionMessagePersistence:
    """Test action request/response message persistence."""

    @pytest.fixture
    def session_with_tool(self):
        """Create session with a simple tool."""
        session = Session()
        tool = Tool(
            func_callable=mock_multiply, config=ToolConfig(name="multiply", provider="tool")
        )
        session.services.register(iModel(backend=tool))
        return session

    async def test_messages_persisted_to_branch(self, session_with_tool):
        """Test that action messages are persisted to branch."""
        session = session_with_tool
        branch = session.create_branch(name="test", resources={"multiply"})

        requests = [
            ActionRequest(function="multiply", arguments={"a": 5, "b": 3}),
        ]

        await act(requests, session, branch)

        # Should have 2 messages: request and response
        messages = session.messages[branch]
        assert len(messages) == 2

        # First is request (branch → tool)
        req_msg = messages[0]
        assert req_msg.sender == branch.id
        assert req_msg.recipient == "multiply"

        # Second is response (tool → branch)
        resp_msg = messages[1]
        assert resp_msg.sender == "multiply"
        assert resp_msg.recipient == branch.id
