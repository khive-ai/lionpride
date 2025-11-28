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


class TestFactoryUncoveredLines:
    """Tests for specific uncovered lines in factory.py.

    These tests cover:
    - Line 65: Missing imodel with no session.default_generate_model
    - Lines 121-122: return_message=True path capturing assistant_msg
    - Lines 126-127: Action execution through operate flow
    - Line 136: Final return with return_message=True
    """

    @pytest.fixture
    def mock_model_local(self):
        """Create a mock iModel for testing without API calls."""
        from lionpride.services.providers.oai_chat import OAIChatEndpoint
        from lionpride.services.types.imodel import iModel

        endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")
        model = iModel(backend=endpoint)

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))
        return model

    @pytest.fixture
    def session_with_model_local(self, mock_model_local):
        """Create session with registered mock model."""
        session = Session()
        session.services.register(mock_model_local, update=True)
        return session, mock_model_local

    def test_missing_imodel_and_no_default_generate_model(self):
        """Test line 65: operate raises when imodel is missing and no default_generate_model."""
        import asyncio

        from lionpride.operations.operate.factory import operate

        # Session without default_generate_model
        session = Session()  # No default_generate_model
        branch = session.create_branch()

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    # No imodel specified
                    instruction="Test",
                    request_model=SimpleModel,
                ),
            ),
        )

        with pytest.raises(
            ValueError,
            match=r"operate requires 'imodel' in communicate\.generate or session\.default_generate_model",
        ):
            asyncio.run(operate(session, branch, params))

    @pytest.mark.asyncio
    async def test_return_message_with_successful_validation(self, session_with_model_local):
        """Test lines 121-122 and 136: return_message=True returns (result, assistant_msg)."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model_local
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
                        data='{"simplemodel": {"value": "test_value"}}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Test instruction",
                    request_model=SimpleModel,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                strict_validation=False,
            ),
            return_message=True,
        )

        result = await operate(session, branch, params)

        # Should return (result, assistant_msg) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        model_result, assistant_msg = result
        # Verify model result
        assert hasattr(model_result, "simplemodel")
        assert model_result.simplemodel.value == "test_value"
        # Verify assistant_msg is a Message (covers lines 121-122 and 136)
        assert assistant_msg is not None

    @pytest.mark.asyncio
    async def test_operate_with_actions_through_full_flow(self, session_with_model_local):
        """Test lines 126-127: Action execution through full operate flow."""
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model_local
        branch = session.create_branch(
            name="test",
            capabilities={"responsemodel", "action_requests"},
            resources={model.name},
        )

        # Register a tool for action execution
        async def add_numbers(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            func_callable=add_numbers, config=ToolConfig(name="add_numbers", provider="tool")
        )
        session.services.register(iModel(backend=tool))

        class ResponseModel(BaseModel):
            answer: str

        # Mock to return response with action_requests
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data='{"responsemodel": {"answer": "will compute"}, "action_requests": [{"function": "add_numbers", "arguments": {"a": 5, "b": 3}}]}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Compute",
                    request_model=ResponseModel,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                strict_validation=False,
            ),
            actions=True,  # Enable action execution
        )

        result = await operate(session, branch, params)

        # Verify action was executed (covers lines 126-127)
        assert hasattr(result, "action_responses")
        assert result.action_responses is not None
        assert len(result.action_responses) == 1
        assert result.action_responses[0].output == 8  # 5 + 3

    @pytest.mark.asyncio
    async def test_operate_with_actions_and_return_message(self, session_with_model_local):
        """Test combined actions + return_message path for full coverage."""
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model_local
        branch = session.create_branch(
            name="test",
            capabilities={"responsemodel", "action_requests"},
            resources={model.name},
        )

        # Register tool
        async def multiply(x: int, y: int) -> int:
            return x * y

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multiply", provider="tool"))
        session.services.register(iModel(backend=tool))

        class ResponseModel(BaseModel):
            answer: str

        # Mock response with action_requests
        async def mock_json_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data='{"responsemodel": {"answer": "computing"}, "action_requests": [{"function": "multiply", "arguments": {"x": 4, "y": 7}}]}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_json_invoke))

        params = OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    imodel=model,
                    instruction="Compute",
                    request_model=ResponseModel,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                strict_validation=False,
            ),
            actions=True,
            return_message=True,
        )

        result = await operate(session, branch, params)

        # Should return tuple with action result
        assert isinstance(result, tuple)
        model_result, assistant_msg = result
        # Verify action was executed
        assert hasattr(model_result, "action_responses")
        assert model_result.action_responses[0].output == 28  # 4 * 7
        # Verify message was captured
        assert assistant_msg is not None
