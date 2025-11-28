# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Test refactored operate factory."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from lionpride import Event, EventStatus
from lionpride.operations.operate import operate
from lionpride.operations.operate.types import (
    CommunicateParams,
    GenerateParams,
    OperateParams,
    ParseParams,
)
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
    async def test_modular_operate_basic(self, session_with_model):
        """Test basic operation of modular operate."""
        session, mock_model = session_with_model
        branch = session.create_branch(capabilities={"simplemodel"}, resources={mock_model.name})

        # Mock response to return structured data wrapped in spec name
        # The Operable creates a model with field "simplemodel" containing SimpleModel
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data='{"simplemodel": {"title": "Test", "value": 42}}'
                    )

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        # Test parameters - nested structure
        # Note: strict_validation=False allows validation without registered rules
        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=mock_model,
                    instruction="Generate a test response",
                    request_model=SimpleModel,
                    imodel_kwargs={"model_name": "test", "temperature": 0.5},
                ),
                parse=ParseParams(),
                strict_validation=False,
            ),
        )

        # Execute
        result = await operate(session, branch, params)

        # Verify - result is the validated model created by Operable
        assert hasattr(result, "simplemodel")
        assert result.simplemodel.title == "Test"
        assert result.simplemodel.value == 42

        # Check that model was invoked
        mock_model.invoke.assert_called_once()
        call_kwargs = mock_model.invoke.call_args.kwargs
        assert "messages" in call_kwargs

    def test_parameter_validation(self):
        """Test parameter validation."""
        import asyncio

        session = Session()
        branch = session.create_branch()

        # Missing communicate params
        with pytest.raises(ValueError, match="communicate"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    OperateParams(),  # No communicate
                )
            )

        # Missing generate params
        with pytest.raises(ValueError, match="generate"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    OperateParams(
                        communicate=CommunicateParams(),  # No generate
                    ),
                )
            )

        # Missing instruction
        with pytest.raises(ValueError, match="instruction"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    OperateParams(
                        communicate=CommunicateParams(
                            generate=GenerateParams(
                                imodel=MagicMock(),
                                request_model=SimpleModel,
                            ),
                        ),
                    ),
                )
            )

        # Missing both request_model and operable
        with pytest.raises(ValueError, match=r"request_model.*operable"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    OperateParams(
                        communicate=CommunicateParams(
                            generate=GenerateParams(
                                instruction="test",
                                imodel=MagicMock(),
                            ),
                        ),
                    ),
                )
            )


class TestFactoryCoverage:
    """Test factory.py uncovered lines."""

    @pytest.mark.asyncio
    async def test_operate_with_operable_directly(self, session_with_model):
        """Test operate with operable parameter using skip_validation."""
        from lionpride.operations.operate.factory import operate
        from lionpride.types import Operable, Spec

        session, model = session_with_model
        branch = session.create_branch(name="test", capabilities={"simple"}, resources={model.name})

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

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Test",
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                operable=operable,
                parse=ParseParams(),
            ),
            skip_validation=True,
        )

        result = await operate(session, branch, params)
        assert result is not None

    @pytest.mark.asyncio
    async def test_operate_with_response_model_invalid_type(self, session_with_model):
        """Test response_model not a BaseModel subclass."""
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
        """Test neither response_model nor operable raises ValueError."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Test",
                ),
                parse=ParseParams(),
            ),
        )

        # This should fail at operate-level validation (before generate)
        with pytest.raises(ValueError, match=r"request_model.*operable"):
            await operate(session, branch, params)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_no_return_message(self, session_with_model):
        """Test validation failure without return_message raises."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities={"strictmodel"}, resources={model.name}
        )

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

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Test",
                    request_model=StrictModel,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                strict_validation=True,
            ),
            return_message=False,
        )

        with pytest.raises(ValueError, match="Response validation failed"):
            await operate(session, branch, params)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_with_return_message(self, session_with_model):
        """Test validation failure with return_message returns tuple."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities={"strictmodel"}, resources={model.name}
        )

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

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Test",
                    request_model=StrictModel,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                strict_validation=False,
            ),
            return_message=True,
        )

        result = await operate(session, branch, params)

        # Should return tuple with validation_failed dict
        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)
        assert result[0].get("validation_failed") is True

    @pytest.mark.asyncio
    async def test_operate_with_return_message(self, session_with_model):
        """Test return_message=True returns (result, message) tuple."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities={"simplemodel"}, resources={model.name}
        )

        class SimpleModel(BaseModel):
            value: str

        # Mock to return valid JSON wrapped in spec name
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data='{"simplemodel": {"value": "test"}}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Test",
                    request_model=SimpleModel,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
            ),
            return_message=True,
        )

        result = await operate(session, branch, params)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_build_operable_with_actions_and_reason(self):
        """Test _build_operable with actions and reason specs."""
        from lionpride.operations.operate.factory import _build_operable

        class MyModel(BaseModel):
            answer: str

        operable, capabilities = _build_operable(
            response_model=MyModel,
            operable=None,
            actions=True,
            reason=True,
        )

        assert operable is not None
        assert capabilities is not None

        # Create the model and verify it has the expected fields
        model = operable.create_model()
        field_names = list(model.model_fields.keys())
        assert "reason" in field_names
        assert "action_requests" in field_names

    @pytest.mark.asyncio
    async def test_build_operable_no_actions_no_reason(self):
        """Test _build_operable without actions/reason."""
        from lionpride.operations.operate.factory import _build_operable

        class SimpleModel(BaseModel):
            value: str

        operable, capabilities = _build_operable(
            response_model=SimpleModel,
            operable=None,
            actions=False,
            reason=False,
        )

        # Operable should still be created (just with the response_model spec)
        assert operable is not None
        assert "simplemodel" in capabilities  # lowercase name

    @pytest.mark.asyncio
    async def test_operate_with_actions_executes_tools(self, session_with_model):
        """Test operate with actions=True executes tools."""
        from lionpride.operations.operate.act import execute_tools, has_action_requests
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(
            name="test",
            capabilities={"responsewithactions", "action_requests"},
            resources={model.name},
        )

        # Register a tool
        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multiply", provider="tool"))
        session.services.register(iModel(backend=tool))

        class ResponseWithActions(BaseModel):
            answer: str
            action_requests: list[ActionRequest] | None = None
            action_responses: list[ActionResponse] | None = None

        # Test the execute_tools directly
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
        """Test tool schemas added to context."""
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities={"simplemodel"}, resources={model.name}
        )

        # Register a tool
        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))
        session.services.register(iModel(backend=tool))

        class SimpleModel(BaseModel):
            value: str

        # Mock to return valid JSON wrapped in spec name
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data='{"simplemodel": {"value": "test"}}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Test",
                    request_model=SimpleModel,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                strict_validation=False,
            ),
        )

        result = await operate(session, branch, params)
        assert result is not None
