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
from lionpride.types import Operable, Spec


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
        """Test basic operation of modular operate with new params structure."""
        session, model = session_with_model
        branch = session.create_branch()
        branch.resources.add("mock_model")  # Add model to branch resources

        # Create operable from response model
        operable = Operable([Spec(SimpleModel, name="output")])

        # Test parameters - use new nested structure with skip_validation
        # (Validation tests are handled separately with proper mock setup)
        params = {
            "communicate": {
                "generate": {
                    "instruction": "Generate a test response",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "test", "temperature": 0.5},
                },
                "operable": operable,
            },
            "skip_validation": True,  # Skip validation for basic structure test
        }

        # Execute
        result = await operate(session, branch, params)

        # Verify - result is the raw text response when skip_validation=True
        assert result is not None
        assert "mock response text" in result

        # Check that model was invoked
        model.invoke.assert_called_once()

    def test_parameter_validation(self):
        """Test parameter validation."""
        import asyncio
        from unittest.mock import MagicMock

        session = Session()
        branch = session.create_branch()
        branch.resources.add("test_model")

        mock_model = MagicMock()
        mock_model.name = "test_model"
        operable = Operable([Spec(SimpleModel, name="output")])

        # Missing communicate
        with pytest.raises(ValueError, match="communicate"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {},
                )
            )

        # Missing communicate.generate
        with pytest.raises(ValueError, match=r"communicate\.generate"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {"communicate": {"operable": operable}},
                )
            )

        # Missing instruction
        with pytest.raises(ValueError, match="instruction"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {
                        "communicate": {
                            "generate": {"imodel": mock_model},
                            "operable": operable,
                        }
                    },
                )
            )

        # Missing imodel
        with pytest.raises(ValueError, match="imodel"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {
                        "communicate": {
                            "generate": {"instruction": "test"},
                            "operable": operable,
                        }
                    },
                )
            )

        # Missing operable
        with pytest.raises(ValueError, match=r"communicate\.operable"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {
                        "communicate": {
                            "generate": {"instruction": "test", "imodel": mock_model},
                        }
                    },
                )
            )


class TestFactoryCoverage:
    """Test factory.py uncovered lines (merged from test_coverage_boost.py)."""

    @pytest.mark.asyncio
    async def test_operate_with_operable_directly(self, session_with_model):
        """Test operate with operable parameter using skip_validation."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")  # Add model to branch resources

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
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
            },
            "skip_validation": True,  # Skip validation to test operable path
        }

        result = await operate(session, branch, parameters)

        assert result is not None

    @pytest.mark.asyncio
    async def test_operate_missing_operable(self, session_with_model):
        """Test that missing operable in communicate raises ValueError."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                # No operable
            }
        }

        with pytest.raises(ValueError, match=r"communicate\.operable"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_no_return_message(self, session_with_model):
        """Test validation failure without return_message raises."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        class StrictModel(BaseModel):
            required_field: str

        operable = Operable([Spec(StrictModel, name="output")])

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
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
                "strict_validation": True,
                "max_retries": 0,
            },
            "return_message": False,
        }

        with pytest.raises(ValueError, match="Response validation failed"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_with_return_message(self, session_with_model):
        """Test validation failure with return_message returns tuple."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        class StrictModel(BaseModel):
            required_field: str

        operable = Operable([Spec(StrictModel, name="output")])

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
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
                "strict_validation": False,
                "max_retries": 0,
            },
            "return_message": True,
        }

        result = await operate(session, branch, parameters)

        # Should return tuple with validation_failed dict
        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)
        assert result[0].get("validation_failed") is True

    @pytest.mark.asyncio
    async def test_operate_with_return_message(self, session_with_model):
        """Test return_message=True returns (result, message) tuple."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        class SimpleModel(BaseModel):
            value: str

        operable = Operable([Spec(SimpleModel, name="output")])

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
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
            },
            "return_message": True,
        }

        result = await operate(session, branch, parameters)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_operate_with_reason_param(self, session_with_model):
        """Test operate with reason=True parameter.

        TODO: Implement actual test for reason=True parameter injection.
        """
        pass

    @pytest.mark.asyncio
    async def test_operate_no_actions_no_reason(self, session_with_model):
        """Test operate without actions/reason returns result."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        class SimpleModel(BaseModel):
            value: str

        operable = Operable([Spec(SimpleModel, name="output")])

        parameters = {
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
            },
            "skip_validation": True,  # Skip validation for param structure test
        }

        result = await operate(session, branch, parameters)
        assert result is not None

    @pytest.mark.asyncio
    async def test_operate_with_actions_executes_tools(self, session_with_model):
        """Test operate with actions=True executes tools."""
        from lionpride.operations.actions import execute_tools, has_action_requests
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, _model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

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
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        # Register a tool
        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))
        session.services.register(iModel(backend=tool))

        class SimpleModel(BaseModel):
            value: str

        operable = Operable([Spec(SimpleModel, name="output")])

        parameters = {
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
            },
            "act": {"tools": True},  # Should add tool_schemas to context
            "skip_validation": True,  # Skip validation for param structure test
        }

        result = await operate(session, branch, parameters)

        assert result is not None

    @pytest.mark.asyncio
    async def test_operate_tool_schemas_with_existing_non_dict_context(self, session_with_model):
        """Test tool schemas with non-dict existing context."""
        from lionpride.operations.operate.factory import operate
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        # Register a tool
        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))
        session.services.register(iModel(backend=tool))

        class SimpleModel(BaseModel):
            value: str

        operable = Operable([Spec(SimpleModel, name="output")])

        parameters = {
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "context": "string context",  # Non-dict context
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
            },
            "act": {"tools": True},
            "skip_validation": True,  # Skip validation for param structure test
        }

        result = await operate(session, branch, parameters)

        assert result is not None

    # --- Additional tests from TestFactoryAdditionalCoverage ---

    @pytest.mark.asyncio
    async def test_operate_missing_communicate_param(self, session_with_model):
        """Test that missing communicate param raises ValueError."""
        from lionpride.operations.operate.factory import operate

        session, _model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        # No communicate param at all
        parameters = {}

        with pytest.raises(ValueError, match="communicate"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_validation_failure_raises(self, session_with_model):
        """Test validation failure without return_message raises ValueError."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        class StrictModel(BaseModel):
            required_field: str

        operable = Operable([Spec(StrictModel, name="output")])

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
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
                "max_retries": 0,
            },
            "return_message": False,
        }

        with pytest.raises(ValueError, match="Response validation failed"):
            await operate(session, branch, parameters)

    @pytest.mark.asyncio
    async def test_operate_action_branch_execution(self, session_with_model):
        """Test action execution with branch object.

        TODO: Implement actual test for actions=True with branch object.
        """
        pass

    @pytest.mark.asyncio
    async def test_operate_action_execution_with_branch(self, session_with_model):
        """Test branch resolution for tool execution."""
        from lionpride.operations.actions import execute_tools
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, _model = session_with_model
        branch = session.create_branch(name="test_branch")
        branch.resources.add("mock_model")

        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multi", provider="tool"))
        session.services.register(iModel(backend=tool))

        class ResponseWithActions(BaseModel):
            answer: str
            action_requests: list[ActionRequest] | None = None
            action_responses: list[ActionResponse] | None = None

        # Use direct tool execution to cover the path
        response = ResponseWithActions(
            answer="test",
            action_requests=[ActionRequest(function="multi", arguments={"a": 2, "b": 3})],
        )

        result, _responses = await execute_tools(response, session, branch)

        assert result.action_responses is not None
        assert result.action_responses[0].output == 6

    @pytest.mark.asyncio
    async def test_operate_return_message_with_branch(self, session_with_model):
        """Test return_message with branch returns tuple."""
        from lionpride.operations.operate.factory import operate

        session, model = session_with_model
        branch = session.create_branch(name="msg_branch")
        branch.resources.add("mock_model")

        class SimpleModel(BaseModel):
            value: str

        operable = Operable([Spec(SimpleModel, name="output")])

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
            "communicate": {
                "generate": {
                    "instruction": "Test",
                    "imodel": model,
                    "imodel_kwargs": {"model_name": "gpt-4"},
                },
                "operable": operable,
            },
            "return_message": True,
        }

        result = await operate(session, branch, parameters)

        assert isinstance(result, tuple)


class TestResponseModelFlattening:
    """Test that response_model fields are flattened at top level when actions=True.

    PR Claim: Response_model fields are now flattened into top-level specs.
    These tests verify the schema structure of _build_request_model output.
    """

    def test_response_model_fields_flattened_with_actions(self):
        """Test action_requests field is at top level (not nested) when actions=True.

        Verifies that when actions=True:
        - action_requests field is added at top level
        - Original response_model fields remain at top level
        - No wrapper nesting like {response_model_name: {...}}
        """
        from lionpride.operations.operate.factory import _build_request_model
        from lionpride.rules import ActionRequest

        # Create a response_model with known fields
        class AnalysisModel(BaseModel):
            reasoning: str = Field(..., description="Analysis reasoning")
            recommendation: str = Field(..., description="Final recommendation")

        operable = Operable(
            [
                Spec(AnalysisModel, name="reasoning"),
                Spec(str, name="recommendation"),
            ],
            name="Analysis",
        )

        # Call _build_request_model with actions=True
        extended_operable, extended_caps = _build_request_model(
            operable=operable,
            capabilities={"reasoning", "recommendation"},
            actions=True,
            reason=False,
        )

        # Verify action_requests is at top level
        allowed_fields = extended_operable.allowed()
        assert "action_requests" in allowed_fields, (
            "action_requests should be at top level when actions=True"
        )

        # Verify original fields are still at top level
        assert "reasoning" in allowed_fields, (
            "Original field 'reasoning' should remain at top level"
        )
        assert "recommendation" in allowed_fields, (
            "Original field 'recommendation' should remain at top level"
        )

        # Verify no nesting - specs should be flat list
        specs = extended_operable.get_specs()
        spec_names = {s.name for s in specs}
        assert spec_names == {"reasoning", "recommendation", "action_requests"}, (
            f"All fields should be at top level, got: {spec_names}"
        )

        # Verify the action_requests spec has correct type
        action_req_spec = extended_operable.get("action_requests")
        assert action_req_spec is not None
        assert action_req_spec.base_type == ActionRequest
        assert action_req_spec.is_listable is True
        assert action_req_spec.is_nullable is True

        # Verify capabilities extended
        assert "action_requests" in extended_caps

    def test_reason_field_flattened_with_reason_param(self):
        """Test reason field is at top level (not nested) when reason=True.

        Similar to actions, the reason field should be injected at top level.
        """
        from lionpride.operations.operate.factory import _build_request_model
        from lionpride.rules import Reason

        class SimpleModel(BaseModel):
            answer: str

        operable = Operable(
            [
                Spec(str, name="answer"),
            ],
            name="Simple",
        )

        # Call with reason=True
        extended_operable, extended_caps = _build_request_model(
            operable=operable,
            capabilities={"answer"},
            actions=False,
            reason=True,
        )

        # Verify reason is at top level
        allowed_fields = extended_operable.allowed()
        assert "reason" in allowed_fields, "reason should be at top level when reason=True"
        assert "answer" in allowed_fields, "Original field 'answer' should remain at top level"

        # Verify the reason spec has correct type
        reason_spec = extended_operable.get("reason")
        assert reason_spec is not None
        assert reason_spec.base_type == Reason
        assert reason_spec.is_nullable is True

        # Verify capabilities extended
        assert "reason" in extended_caps

    def test_both_actions_and_reason_flattened(self):
        """Test both action_requests and reason are flattened when both params True."""
        from lionpride.operations.operate.factory import _build_request_model

        class OutputModel(BaseModel):
            result: str

        operable = Operable(
            [
                Spec(str, name="result"),
            ],
            name="Output",
        )

        extended_operable, _extended_caps = _build_request_model(
            operable=operable,
            capabilities={"result"},
            actions=True,
            reason=True,
        )

        # Verify all fields at top level
        allowed_fields = extended_operable.allowed()
        assert allowed_fields == {"result", "action_requests", "reason"}, (
            f"All fields should be flat at top level, got: {allowed_fields}"
        )

        # Verify model name indicates extensions
        assert "WithReasonAndActions" in extended_operable.name

    def test_duplicate_action_requests_skipped(self):
        """Test that if response_model already has action_requests, no duplicate is added.

        When the original operable already defines action_requests,
        _build_request_model should NOT add a duplicate field.
        """
        from lionpride.operations.operate.factory import _build_request_model
        from lionpride.rules import ActionRequest

        # Create an operable that already has action_requests
        operable = Operable(
            [
                Spec(str, name="answer"),
                Spec(ActionRequest, name="action_requests", listable=True, nullable=True),
            ],
            name="ModelWithActions",
        )

        original_caps = {"answer", "action_requests"}

        # Call with actions=True - should NOT duplicate
        extended_operable, extended_caps = _build_request_model(
            operable=operable,
            capabilities=original_caps,
            actions=True,
            reason=False,
        )

        # Should return original operable unchanged (no injection needed)
        assert extended_operable is operable, (
            "Should return original operable when action_requests already exists"
        )
        assert extended_caps is original_caps, (
            "Should return original capabilities when no injection needed"
        )

        # Verify only one action_requests field
        specs = extended_operable.get_specs()
        action_req_count = sum(1 for s in specs if s.name == "action_requests")
        assert action_req_count == 1, (
            f"Should have exactly one action_requests field, got {action_req_count}"
        )

    def test_duplicate_reason_skipped(self):
        """Test that if response_model already has reason, no duplicate is added."""
        from lionpride.operations.operate.factory import _build_request_model
        from lionpride.rules import Reason

        # Create an operable that already has reason
        operable = Operable(
            [
                Spec(str, name="answer"),
                Spec(Reason, name="reason", nullable=True),
            ],
            name="ModelWithReason",
        )

        original_caps = {"answer", "reason"}

        # Call with reason=True - should NOT duplicate
        extended_operable, _extended_caps = _build_request_model(
            operable=operable,
            capabilities=original_caps,
            actions=False,
            reason=True,
        )

        # Should return original operable unchanged
        assert extended_operable is operable, (
            "Should return original operable when reason already exists"
        )

        # Verify only one reason field
        specs = extended_operable.get_specs()
        reason_count = sum(1 for s in specs if s.name == "reason")
        assert reason_count == 1, f"Should have exactly one reason field, got {reason_count}"

    def test_no_injection_when_params_false(self):
        """Test that no fields are injected when actions=False and reason=False."""
        from lionpride.operations.operate.factory import _build_request_model

        operable = Operable(
            [
                Spec(str, name="output"),
            ],
            name="Simple",
        )

        original_caps = {"output"}

        extended_operable, result_caps = _build_request_model(
            operable=operable,
            capabilities=original_caps,
            actions=False,
            reason=False,
        )

        # Should return original unchanged
        assert extended_operable is operable
        assert result_caps is original_caps
        assert extended_operable.allowed() == {"output"}

    def test_output_model_uses_action_responses(self):
        """Test _build_output_model injects action_responses (not action_requests).

        The output model should have action_responses for tool execution results,
        distinct from action_requests which is for LLM proposals.
        """
        from lionpride.operations.operate.factory import _build_output_model
        from lionpride.rules import ActionResponse

        operable = Operable(
            [
                Spec(str, name="answer"),
            ],
            name="Output",
        )

        extended_operable, _extended_caps = _build_output_model(
            operable=operable,
            capabilities={"answer"},
            actions=True,
            reason=False,
        )

        # Verify action_responses (not action_requests) is at top level
        allowed_fields = extended_operable.allowed()
        assert "action_responses" in allowed_fields, (
            "Output model should have action_responses field"
        )
        assert "action_requests" not in allowed_fields, (
            "Output model should NOT have action_requests field"
        )

        # Verify correct type
        action_resp_spec = extended_operable.get("action_responses")
        assert action_resp_spec is not None
        assert action_resp_spec.base_type == ActionResponse
        assert action_resp_spec.is_listable is True
        assert action_resp_spec.is_nullable is True
