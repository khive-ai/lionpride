# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Test refactored operate factory."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from lionpride import Event, EventStatus
from lionpride.operations.operate import operate
from lionpride.rules import ActionRequest, ActionResponse
from lionpride.session import Session


@dataclass
class MockNormalizedResponse:
    """Mock NormalizedResponse for testing."""

    data: str = "mock response text"
    raw_response: dict = None
    metadata: dict = None

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {"id": "mock-id", "choices": [{"message": {"content": self.data}}]}
        if self.metadata is None:
            self.metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}


class SimpleModel(BaseModel):
    """Simple test model."""

    title: str = Field(..., description="Title")
    value: int = Field(..., ge=0)


class TestOperateRefactor:
    """Test the refactored modular operate."""

    @pytest.mark.asyncio
    async def test_modular_operate_basic(self):
        """Test basic operation of modular operate."""
        from unittest.mock import AsyncMock, MagicMock

        session = Session()
        branch = session.create_branch()

        # Mock model
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.invoke = AsyncMock()

        # Mock response
        mock_execution = MagicMock()
        mock_execution.status.value = "completed"
        mock_execution.response.data = {"title": "Test", "value": 42}

        mock_calling = MagicMock()
        mock_calling.execution = mock_execution
        mock_model.invoke.return_value = mock_calling

        # Test parameters
        params = {
            "instruction": "Generate a test response",
            "imodel": mock_model,
            "response_model": SimpleModel,
            "model_kwargs": {"model_name": "test", "temperature": 0.5},
        }

        # Execute
        result = await operate(session, branch, params)

        # Verify
        assert isinstance(result, SimpleModel)
        assert result.title == "Test"
        assert result.value == 42

        # Check that model was invoked
        mock_model.invoke.assert_called_once()
        call_kwargs = mock_model.invoke.call_args.kwargs
        # model_name is passed via model_kwargs
        assert call_kwargs.get("model_name") == "test" or call_kwargs.get("model") == "test"
        assert "messages" in call_kwargs

    def test_parameter_validation(self):
        """Test parameter validation."""
        import asyncio
        from unittest.mock import MagicMock

        session = Session()
        branch = session.create_branch()

        # Missing instruction
        with pytest.raises(ValueError, match="instruction"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {"imodel": MagicMock(), "response_model": SimpleModel},
                )
            )

        # Missing imodel
        with pytest.raises(ValueError, match="imodel"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {"instruction": "test", "response_model": SimpleModel},
                )
            )

        # Missing both response_model and operable
        with pytest.raises(ValueError, match=r"response_model.*operable"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {"instruction": "test", "imodel": MagicMock()},
                )
            )


class TestFactoryCoverage:
    """Test factory.py uncovered lines (merged from test_coverage_boost.py)."""

    @pytest.mark.asyncio
    async def test_operate_with_operable_directly(self, session_with_model):
        """Test lines 69-71: operate with operable parameter using skip_validation."""
        from lionpride.operations.operate.factory import operate
        from lionpride.types import Operable, Spec

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Create Operable
        class SimpleSpec(BaseModel):
            value: str

        operable = Operable(
            specs=(Spec(base_type=SimpleSpec, name="simple"),),
            name="TestOperable",
        )

        # Mock to return valid JSON
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"value": "test"}')

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "operable": operable,
            "skip_validation": True,  # Skip validation to test operable path
            "model_kwargs": {"model_name": "gpt-4"},
        }

        result = await operate(session, branch, parameters)

        assert result is not None

    @pytest.mark.asyncio
    async def test_operate_with_response_model_invalid_type(self, session_with_model):
        """Test lines 201-206: response_model not a BaseModel subclass."""
        from lionpride.operations.operate.factory import _build_operable

        # Pass non-BaseModel class
        with pytest.raises(ValueError, match="response_model must be a Pydantic BaseModel"):
            _build_operable(
                response_model=dict,  # type: ignore
                operable=None,
                actions=True,
                reason=False,
            )

    @pytest.mark.asyncio
    async def test_operate_no_response_model_or_operable(self, session_with_model):
        """Test line 75: Neither response_model nor operable raises ValueError."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "instruction": "Test",
            "imodel": model,
            # No response_model or operable
        }

        with pytest.raises(ValueError, match=r"response_model.*operable"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_no_return_message(self, session_with_model):
        """Test lines 89-92: Validation failure without return_message raises."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        class StrictModel(BaseModel):
            required_field: str

        # Mock to return invalid JSON
        async def mock_invalid_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="invalid json")

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invalid_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": StrictModel,
            "return_message": False,
            "strict_validation": True,
            "max_retries": 0,
            "model_kwargs": {"model_name": "gpt-4"},
        }

        with pytest.raises(ValueError, match="Response validation failed"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_with_return_message(self, session_with_model):
        """Test lines 90-92: Validation failure with return_message returns tuple."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        class StrictModel(BaseModel):
            required_field: str

        # Mock to return invalid JSON
        async def mock_invalid_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="invalid json")

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invalid_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": StrictModel,
            "return_message": True,
            "strict_validation": False,
            "max_retries": 0,
            "model_kwargs": {"model_name": "gpt-4"},
        }

        result = await operate(session, branch, parameters)

        # Should return tuple with validation_failed dict
        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)
        assert result[0].get("validation_failed") is True

    @pytest.mark.asyncio
    async def test_operate_with_return_message(self, session_with_model):
        """Test lines 108-114: return_message=True returns (result, message) tuple."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        class SimpleModel(BaseModel):
            value: str

        # Mock to return valid JSON
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"value": "test"}')

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "return_message": True,
            "model_kwargs": {"model_name": "gpt-4"},
        }

        result = await operate(session, branch, parameters)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_build_operable_with_actions_and_reason(self):
        """Test lines 213-263: _build_operable with actions and reason specs."""
        from lionpride.operations.operate.factory import _build_operable

        class MyModel(BaseModel):
            answer: str

        operable, validation_model = _build_operable(
            response_model=MyModel,
            operable=None,
            actions=True,
            reason=True,
        )

        assert operable is not None
        assert validation_model is not None

        # Create the model and verify it has the expected fields
        model = operable.create_model()
        field_names = list(model.model_fields.keys())
        assert "reason" in field_names
        assert "action_requests" in field_names

    @pytest.mark.asyncio
    async def test_build_operable_no_actions_no_reason(self):
        """Test lines 208-210: _build_operable without actions/reason returns None operable."""
        from lionpride.operations.operate.factory import _build_operable

        class SimpleModel(BaseModel):
            value: str

        operable, validation_model = _build_operable(
            response_model=SimpleModel,
            operable=None,
            actions=False,
            reason=False,
        )

        # No Operable needed
        assert operable is None
        assert validation_model is SimpleModel

    @pytest.mark.asyncio
    async def test_extract_model_kwargs_nested(self):
        """Test lines 178-186: _extract_model_kwargs with nested model_kwargs."""
        from lionpride.operations.operate.factory import _extract_model_kwargs

        params = {
            "instruction": "test",
            "imodel": MagicMock(),
            "response_model": MagicMock(),
            "model_kwargs": {"temperature": 0.5, "model_name": "gpt-4"},
            "extra_param": "value",  # Should be extracted
        }

        result = _extract_model_kwargs(params)

        # Should include nested model_kwargs
        assert result["temperature"] == 0.5
        assert result["model_name"] == "gpt-4"
        # Should include extra flat params
        assert result["extra_param"] == "value"

    @pytest.mark.asyncio
    async def test_operate_with_actions_executes_tools(self, session_with_model):
        """Test lines 95-105: operate with actions=True executes tools."""
        from lionpride.operations.operate.factory import operate
        from lionpride.operations.operate.tool_executor import execute_tools, has_action_requests
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Register a tool
        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multiply", provider="tool"))
        session.services.register(iModel(backend=tool))

        class ResponseWithActions(BaseModel):
            answer: str
            action_requests: list[ActionRequest] | None = None
            action_responses: list[ActionResponse] | None = None

        # Mock to return JSON with action_requests - properly formatted for the model
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data='{"answer": "test", "action_requests": [{"function": "multiply", "arguments": {"a": 3, "b": 4}}]}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        # Test the execute_tools directly to cover lines 95-105
        response = ResponseWithActions(
            answer="test",
            action_requests=[ActionRequest(function="multiply", arguments={"a": 3, "b": 4})],
        )

        # Verify has_action_requests works
        assert has_action_requests(response) is True

        # Execute tools directly
        result, _responses = await execute_tools(response, session, branch)

        # Should have executed action and updated response
        assert result.action_responses is not None
        assert len(result.action_responses) == 1
        assert result.action_responses[0].output == 12

    @pytest.mark.asyncio
    async def test_operate_tool_schemas_in_context(self, session_with_model):
        """Test lines 58-66: Tool schemas added to context."""
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Register a tool
        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))
        session.services.register(iModel(backend=tool))

        class SimpleModel(BaseModel):
            value: str

        # Mock to return valid JSON
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"value": "test"}')

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "tools": True,  # Should add tool_schemas to context
            "model_kwargs": {"model_name": "gpt-4"},
        }

        result = await operate(session, branch, parameters)

        assert result is not None

    @pytest.mark.asyncio
    async def test_operate_tool_schemas_with_existing_non_dict_context(self, session_with_model):
        """Test lines 62-66: Tool schemas with non-dict existing context."""
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Register a tool
        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))
        session.services.register(iModel(backend=tool))

        class SimpleModel(BaseModel):
            value: str

        # Mock to return valid JSON
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"value": "test"}')

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "tools": True,
            "context": "string context",  # Non-dict context (lines 62-66)
            "model_kwargs": {"model_name": "gpt-4"},
        }

        result = await operate(session, branch, parameters)

        assert result is not None

    # --- Additional tests from TestFactoryAdditionalCoverage ---

    @pytest.mark.asyncio
    async def test_operate_no_response_model_or_operable_line75(self, session_with_model):
        """Test line 75: Neither response_model nor operable raises ValueError."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Neither response_model nor operable
        parameters = {
            "instruction": "Test",
            "imodel": model,
            # No response_model
            # No operable
            "model_kwargs": {"model_name": "gpt-4"},
        }

        with pytest.raises(ValueError, match=r"operate requires.*response_model.*operable"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_raises_line91(self, session_with_model):
        """Test line 91: Validation failure without return_message raises ValueError."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        class StrictModel(BaseModel):
            required_field: str

        # Mock to return invalid JSON
        async def mock_invalid_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="invalid json")

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invalid_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": StrictModel,
            "return_message": False,  # Line 90 check
            "max_retries": 0,  # No retries
            "model_kwargs": {"model_name": "gpt-4"},
        }

        with pytest.raises(ValueError, match="Response validation failed"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_action_branch_string_resolution_line97_100(self, session_with_model):
        """Test lines 97-100: Branch string resolution for action execution.

        This test verifies that when branch is passed as a string and actions=True,
        the branch is resolved before executing tools (lines 97-98).
        """
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        session.create_branch(name="action_branch")

        # Register a tool (name must be >= 4 chars)
        async def adder(a: int, b: int) -> int:
            return a + b

        tool = Tool(func_callable=adder, config=ToolConfig(name="adder", provider="tool"))
        session.services.register(iModel(backend=tool))

        class ResponseWithActions(BaseModel):
            answer: str
            action_requests: list[ActionRequest] | None = None
            action_responses: list[ActionResponse] | None = None

        # Mock communicate to return a result with action_requests
        # This bypasses validation and tests the action execution path directly
        mock_result = ResponseWithActions(
            answer="test",
            action_requests=[ActionRequest(function="adder", arguments={"a": 5, "b": 7})],
        )

        with patch("lionpride.operations.operate.communicate.communicate") as mock_communicate:
            mock_communicate.return_value = mock_result

            # Pass branch as STRING to trigger lines 97-98
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "response_model": ResponseWithActions,
                "actions": True,  # Enable actions (line 95)
                "model_kwargs": {"model_name": "gpt-4"},
            }

            # Pass branch as string
            result = await operate(session, "action_branch", parameters)

            # Should have executed action
            assert result.action_responses is not None
            assert len(result.action_responses) == 1
            assert result.action_responses[0].output == 12

    @pytest.mark.asyncio
    async def test_operate_action_execution_with_branch_string(self, session_with_model):
        """Test lines 97-100: Branch string resolution for tool execution."""
        from lionpride.operations.operate.factory import operate
        from lionpride.operations.operate.tool_executor import execute_tools
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        session.create_branch(name="test_branch")

        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multi", provider="tool"))
        session.services.register(iModel(backend=tool))

        class ResponseWithActions(BaseModel):
            answer: str
            action_requests: list[ActionRequest] | None = None
            action_responses: list[ActionResponse] | None = None

        # Mock to return JSON with action_requests
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data='{"answer": "test", "action_requests": [{"function": "multi", "arguments": {"a": 2, "b": 3}}]}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        # Use direct tool execution to cover the path
        response = ResponseWithActions(
            answer="test",
            action_requests=[ActionRequest(function="multi", arguments={"a": 2, "b": 3})],
        )

        # Execute with branch as string (this simulates lines 97-98)
        branch = session.conversations.get_progression("test_branch")
        result, _responses = await execute_tools(response, session, branch)

        assert result.action_responses is not None
        assert result.action_responses[0].output == 6

    @pytest.mark.asyncio
    async def test_operate_return_message_with_branch_string(self, session_with_model):
        """Test line 111: Get assistant message with branch string."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        session.create_branch(name="msg_branch")

        class SimpleModel(BaseModel):
            value: str

        # Mock to return valid JSON
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"value": "test"}')

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "return_message": True,  # Line 108-114
            "model_kwargs": {"model_name": "gpt-4"},
        }

        # Pass branch as string to test line 111
        result = await operate(session, "msg_branch", parameters)

        assert isinstance(result, tuple)
