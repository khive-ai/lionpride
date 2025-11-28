# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for react.py module.

Comprehensive coverage tests for the ReAct (Reasoning + Acting) operation,
testing parameter validation, execution flow, action handling, and error cases.

Stream-first architecture:
    react_stream() - async generator yielding intermediate results
    react() - wrapper collecting all results
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from lionpride.operations.operate.types import (
    ActParams,
    CommunicateParams,
    GenerateParams,
    OperateParams,
    ParseParams,
    ReactParams,
)
from lionpride.rules import ActionRequest


def _make_react_params(
    *,
    instruction=None,
    imodel=None,
    imodel_kwargs=None,
    max_steps=10,
    return_trace=False,
    context=None,
    request_model=None,
):
    """Helper to build nested ReactParams structure."""
    return ReactParams(
        operate=OperateParams(
            communicate=CommunicateParams(
                generate=GenerateParams(
                    instruction=instruction,
                    imodel=imodel,
                    imodel_kwargs=imodel_kwargs or {},
                    context=context,
                    request_model=request_model,
                ),
                parse=ParseParams(),
                strict_validation=False,
            ),
            actions=True,
            reason=True,
        ),
        max_steps=max_steps,
        return_trace=return_trace,
    )


class TestReactCoverage:
    """Test react.py uncovered lines."""

    async def test_react_missing_operate_params(self, session_with_model):
        """Test missing operate params raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # ReactParams with no operate
        params = ReactParams()

        with pytest.raises(ValueError, match="react requires 'operate' params"):
            await react(session, branch, params)

    async def test_react_missing_instruction(self, session_with_model):
        """Test missing instruction raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = _make_react_params(imodel=model)

        with pytest.raises(ValueError, match="instruction"):
            await react(session, branch, params)

    async def test_react_missing_imodel(self, session_with_model):
        """Test missing imodel raises ValueError."""
        from lionpride.operations.operate.react import react

        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = _make_react_params(instruction="Test")

        with pytest.raises(ValueError, match="imodel"):
            await react(session, branch, params)

    async def test_react_missing_model_name(self, session_with_model):
        """Test missing model_name raises ValueError when imodel has no .name."""
        from unittest.mock import MagicMock

        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Use a mock imodel without .name attribute to test error case
        mock_imodel = MagicMock(spec=[])  # spec=[] means no attributes

        params = _make_react_params(
            instruction="Test",
            imodel=mock_imodel,
            # No model_name in imodel_kwargs and imodel has no .name
        )

        with pytest.raises(ValueError, match="model_name"):
            await react(session, branch, params)

    async def test_react_branch_string_resolution(self, session_with_model):
        """Test branch string resolution."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        session.create_branch(name="test_branch", resources={model.name})

        # Mock operate at the factory module level (operate is imported inside function)
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                imodel_kwargs={"model_name": "gpt-4"},
            )

            result = await react(session, "test_branch", params)

            assert result.completed is True

    async def test_react_step_response_model(self):
        """Test ReactStepResponse model structure."""
        from lionpride.operations.operate.react import ReactStepResponse

        # Verify model has expected fields
        fields = ReactStepResponse.model_fields
        assert "final_answer" in fields
        assert "reasoning" in fields
        assert "action_requests" in fields
        assert "is_done" in fields

    async def test_react_validation_failure(self, session_with_model):
        """Test validation failure handling."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to return validation failure
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.return_value = {"validation_failed": True, "error": "Invalid"}

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                imodel_kwargs={"model_name": "gpt-4"},
            )

            result = await react(session, branch, params)

            assert result.completed is False
            # Validation failure yields step and returns, reason_stopped stays empty
            assert len(result.steps) == 1
            assert "Validation failed" in result.steps[0].reasoning

    async def test_react_exception_handling(self, session_with_model):
        """Test exception handling in react loop."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to raise exception
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.side_effect = RuntimeError("Test error")

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                imodel_kwargs={"model_name": "gpt-4"},
            )

            result = await react(session, branch, params)

            assert result.completed is False
            # Exception yields step with error in reasoning and returns
            assert len(result.steps) == 1
            assert "Error:" in result.steps[0].reasoning
            assert "Test error" in result.steps[0].reasoning

    async def test_react_max_steps_reached(self, session_with_model):
        """Test max steps reached forces completion on last step.

        Note: When max_steps is reached, the last step is automatically
        marked as final (is_done=True on last step), so completed=True
        and reason_stopped='Task completed'.
        """
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to never voluntarily finish
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = False
            mock_result.reasoning = "thinking"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                max_steps=2,
                imodel_kwargs={"model_name": "gpt-4"},
            )

            result = await react(session, branch, params)

            # On last step (step 2), is_done is forced True
            # So completed=True and reason_stopped='Task completed'
            assert result.completed is True
            assert result.total_steps == 2
            assert result.steps[-1].is_final is True

    async def test_react_verbose_logging(self, session_with_model, caplog):
        """Test verbose logging in react."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "I figured it out"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                return_trace=True,  # verbose
                imodel_kwargs={"model_name": "gpt-4"},
            )

            with caplog.at_level(logging.INFO, logger="lionpride.operations.operate.react"):
                result = await react(session, branch, params)

            # Verify logging occurred
            assert result.completed is True
            assert any("ReAct Step" in record.message for record in caplog.records)

    async def test_react_stream_yields_steps(self, session_with_model):
        """Test react_stream yields intermediate steps."""
        from lionpride.operations.operate.react import react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to run 2 steps then complete
        call_count = 0

        async def mock_operate_multi(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_result.reasoning = f"Step {call_count} reasoning"
            mock_result.action_requests = None
            mock_result.is_done = call_count >= 2
            return mock_result

        with patch("lionpride.operations.operate.factory.operate", side_effect=mock_operate_multi):
            params = _make_react_params(
                instruction="Test",
                imodel=model,
                max_steps=5,
                imodel_kwargs={"model_name": "gpt-4"},
            )

            steps = []
            async for step in react_stream(session, branch, params):
                steps.append(step)

            assert len(steps) == 2
            assert steps[0].step == 1
            assert steps[1].step == 2
            assert steps[1].is_final is True

    async def test_react_with_context(self, session_with_model):
        """Test context added to instruction."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                context={"info": "Important context info"},
                imodel_kwargs={"model_name": "gpt-4"},
            )

            result = await react(session, branch, params)

            # Verify operate was called and result completed
            assert result.completed is True
            mock_operate.assert_called()

    async def test_react_action_responses_captured(self, session_with_model):
        """Test action responses are captured in steps."""
        from lionpride.operations.operate.react import react
        from lionpride.rules import ActionResponse

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to return action requests and responses
        call_count = 0

        async def mock_operate_with_actions(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: return with action data
                mock_result = MagicMock()
                mock_result.is_done = False
                mock_result.reasoning = "I need to calculate"
                mock_result.action_requests = [
                    ActionRequest(function="multiply", arguments={"a": 3, "b": 4})
                ]
                mock_result.action_responses = [
                    ActionResponse(function="multiply", arguments={"a": 3, "b": 4}, output=12)
                ]
                return mock_result
            else:
                # Second call: complete
                mock_result = MagicMock()
                mock_result.is_done = True
                mock_result.final_answer = "The result is 12"
                mock_result.reasoning = "Calculation complete"
                mock_result.action_requests = None
                mock_result.action_responses = None
                return mock_result

        with patch(
            "lionpride.operations.operate.factory.operate",
            side_effect=mock_operate_with_actions,
        ):
            params = _make_react_params(
                instruction="Calculate 3 * 4",
                imodel=model,
                imodel_kwargs={"model_name": "gpt-4"},
            )

            result = await react(session, branch, params)

            assert result.completed is True
            assert len(result.steps) >= 2
            # First step should have action execution data
            assert len(result.steps[0].actions_requested) == 1
            assert len(result.steps[0].actions_executed) == 1
            assert result.steps[0].actions_executed[0].output == 12

    async def test_react_verbose_exception_logging(self, session_with_model, caplog):
        """Test verbose exception logging."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Mock operate to raise exception
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.side_effect = RuntimeError("Error for traceback test")

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                return_trace=True,  # verbose
                imodel_kwargs={"model_name": "gpt-4"},
            )

            with caplog.at_level(logging.ERROR, logger="lionpride.operations.operate.react"):
                result = await react(session, branch, params)

            # Exception is captured in step reasoning
            assert result.completed is False
            assert len(result.steps) == 1
            assert "Error:" in result.steps[0].reasoning
            assert "Error for traceback test" in result.steps[0].reasoning
