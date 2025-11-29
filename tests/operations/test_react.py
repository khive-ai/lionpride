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

from lionpride.errors import AccessError, ConfigurationError, ValidationError
from lionpride.operations.operate.types import (
    GenerateParams,
    ParseParams,
    ReactParams,
)
from lionpride.rules import ActionRequest

# React protocol capabilities required on branch
REACT_CAPABILITIES = {"reasoning", "action_requests", "is_done"}


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
    """Helper to build flat ReactParams structure (inheritance-based)."""
    return ReactParams(
        generate=GenerateParams(
            instruction=instruction,
            imodel=imodel,
            imodel_kwargs=imodel_kwargs or {},
            context=context,
            request_model=request_model,
        ),
        parse=ParseParams(),
        strict_validation=False,
        actions=True,
        reason=True,
        max_steps=max_steps,
        return_trace=return_trace,
    )


class TestReactCoverage:
    """Test react.py uncovered lines."""

    async def test_react_missing_generate_params(self, session_with_model):
        """Test missing generate params raises ValidationError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

        # ReactParams with no generate (inherited from CommunicateParams)
        params = ReactParams()

        with pytest.raises(ValidationError, match="react requires 'generate' params"):
            await react(session, branch, params)

    async def test_react_missing_instruction(self, session_with_model):
        """Test missing instruction raises ValidationError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

        params = _make_react_params(imodel=model)

        with pytest.raises(ValidationError, match="instruction"):
            await react(session, branch, params)

    async def test_react_missing_imodel_no_default(self, session_with_model):
        """Test missing imodel raises ConfigurationError when no default_generate_model."""
        from lionpride.operations.operate.react import react

        session, _ = session_with_model
        branch = session.create_branch(name="test", capabilities=REACT_CAPABILITIES)

        params = _make_react_params(instruction="Test")

        with pytest.raises(ConfigurationError, match="imodel"):
            await react(session, branch, params)

    async def test_react_missing_model_name(self, session_with_model):
        """Test missing model_name raises ConfigurationError when imodel has no .name."""
        from unittest.mock import MagicMock

        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

        # Use a mock imodel without .name attribute to test error case
        mock_imodel = MagicMock(spec=[])  # spec=[] means no attributes

        params = _make_react_params(
            instruction="Test",
            imodel=mock_imodel,
            # No model_name in imodel_kwargs and imodel has no .name
        )

        with pytest.raises(ConfigurationError, match="model_name"):
            await react(session, branch, params)

    async def test_react_missing_branch_capabilities(self, session_with_model):
        """Test missing branch capabilities raises AccessError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        # Branch without react capabilities
        branch = session.create_branch(name="test", resources={model.name})

        params = _make_react_params(
            instruction="Test",
            imodel=model,
            imodel_kwargs={"model_name": "gpt-4"},
        )

        with pytest.raises(AccessError, match="missing react capabilities"):
            await react(session, branch, params)

    async def test_react_branch_string_resolution(self, session_with_model):
        """Test branch string resolution."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        session.create_branch(
            name="test_branch", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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

        # Verify model has expected fields (no final_answer - react is pure loop)
        fields = ReactStepResponse.model_fields
        assert "reasoning" in fields
        assert "action_requests" in fields
        assert "is_done" in fields

    async def test_react_exception_handling(self, session_with_model):
        """Test exception handling in react loop."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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
        """Test max steps reached forces completion on last step."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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
            assert result.completed is True
            assert result.total_steps == 2
            assert result.steps[-1].is_final is True

    async def test_react_verbose_logging(self, session_with_model, caplog):
        """Test verbose logging in react."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

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


class TestIntermediateResponseOptions:
    """Tests for intermediate_response_options feature."""

    def test_build_intermediate_operable(self):
        """Test building Operable from intermediate option models."""
        from lionpride.operations.operate.react import build_intermediate_operable

        class ProgressReport(BaseModel):
            progress_pct: int
            status: str

        class PartialResult(BaseModel):
            data: str

        operable = build_intermediate_operable([ProgressReport, PartialResult])

        # Should have specs for both models
        assert operable.name == "IntermediateOptions"
        specs = operable.get_specs()
        assert len(specs) == 2
        assert specs[0].name == "progressreport"
        assert specs[1].name == "partialresult"
        # Both should be nullable
        assert specs[0].is_nullable
        assert specs[1].is_nullable

    def test_build_intermediate_operable_single_model(self):
        """Test building Operable from single model."""
        from lionpride.operations.operate.react import build_intermediate_operable

        class Progress(BaseModel):
            pct: int

        operable = build_intermediate_operable(Progress)

        specs = operable.get_specs()
        assert len(specs) == 1
        assert specs[0].name == "progress"

    def test_build_intermediate_operable_listable(self):
        """Test building Operable with listable option."""
        from lionpride.operations.operate.react import build_intermediate_operable

        class CodeBlock(BaseModel):
            code: str

        operable = build_intermediate_operable(CodeBlock, listable=True)

        specs = operable.get_specs()
        assert specs[0].is_listable
        assert specs[0].is_nullable

    def test_build_step_operable_basic(self):
        """Test building step Operable without intermediate options."""
        from lionpride.operations.operate.react import build_step_operable

        operable = build_step_operable()

        assert operable.name == "ReactStepResponse"
        spec_names = {s.name for s in operable.get_specs()}
        assert "reasoning" in spec_names
        assert "action_requests" in spec_names
        assert "is_done" in spec_names

    def test_build_step_operable_with_intermediate_options(self):
        """Test building step Operable with intermediate options."""
        from lionpride.operations.operate.react import build_step_operable

        class Progress(BaseModel):
            pct: int

        operable = build_step_operable(intermediate_options=[Progress])

        spec_names = {s.name for s in operable.get_specs()}
        assert "intermediate_response_options" in spec_names

    def test_build_step_operable_creates_model(self):
        """Test that step Operable can create a Pydantic model."""
        from lionpride.operations.operate.react import build_step_operable

        class Progress(BaseModel):
            pct: int

        operable = build_step_operable(intermediate_options=[Progress])
        Model = operable.create_model()

        # Should be able to instantiate with defaults
        instance = Model(is_done=False)
        assert instance.is_done is False
        assert instance.reasoning is None
        assert instance.intermediate_response_options is None

    async def test_react_with_intermediate_options(self, session_with_model):
        """Test react with intermediate_response_options parameter."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        # Need intermediate_response_options capability too
        branch = session.create_branch(
            name="test",
            capabilities=REACT_CAPABILITIES | {"intermediate_response_options"},
            resources={model.name},
        )

        class Progress(BaseModel):
            pct: int
            status: str

        # Mock operate to return intermediate options
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.reasoning = "Done"
            mock_result.action_requests = None
            mock_result.action_responses = None
            # Mock intermediate options
            mock_iro = MagicMock()
            mock_iro.model_dump.return_value = {"progress": {"pct": 100, "status": "complete"}}
            mock_result.intermediate_response_options = mock_iro
            mock_operate.return_value = mock_result

            params = ReactParams(
                generate=GenerateParams(
                    instruction="Test",
                    imodel=model,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                actions=True,
                reason=True,
                max_steps=5,
                intermediate_response_options=[Progress],
            )

            result = await react(session, branch, params)

            assert result.completed is True
            assert result.steps[0].intermediate_options is not None
            assert result.steps[0].intermediate_options.get("progress") == {
                "pct": 100,
                "status": "complete",
            }

    async def test_react_model_name_from_imodel_attribute(self, session_with_model):
        """Test model_name extracted from imodel.name attribute."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.reasoning = "done"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            # Do NOT provide model_name in imodel_kwargs - use imodel.name instead
            params = _make_react_params(
                instruction="Test",
                imodel=model,
                imodel_kwargs={},  # No model_name - will use imodel.name
            )

            result = await react(session, branch, params)

            assert result.completed is True
            # Verify operate was called with model from imodel.name
            mock_operate.assert_called()

    async def test_react_verbose_with_intermediate_options(self, session_with_model, caplog):
        """Test verbose logging with intermediate_response_options."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test",
            capabilities=REACT_CAPABILITIES | {"intermediate_response_options"},
            resources={model.name},
        )

        class Progress(BaseModel):
            pct: int

        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.reasoning = "Done"
            mock_result.action_requests = None
            mock_result.action_responses = None
            mock_result.intermediate_response_options = None
            mock_operate.return_value = mock_result

            params = ReactParams(
                generate=GenerateParams(
                    instruction="Test",
                    imodel=model,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                actions=True,
                reason=True,
                max_steps=5,
                intermediate_response_options=[Progress],
                return_trace=True,  # verbose=True
            )

            with caplog.at_level(logging.INFO, logger="lionpride.operations.operate.react"):
                result = await react(session, branch, params)

            assert result.completed is True

    async def test_react_verbose_action_responses_logging(self, session_with_model, caplog):
        """Test verbose logging for action responses."""
        from lionpride.operations.operate.react import react
        from lionpride.rules import ActionResponse

        session, model = session_with_model
        branch = session.create_branch(
            name="test", capabilities=REACT_CAPABILITIES, resources={model.name}
        )

        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.reasoning = "Done with actions"
            mock_result.action_requests = None
            mock_result.action_responses = [
                ActionResponse(function="test_tool", arguments={}, output="tool output result")
            ]
            mock_operate.return_value = mock_result

            params = _make_react_params(
                instruction="Test",
                imodel=model,
                imodel_kwargs={"model_name": "gpt-4"},
                return_trace=True,  # verbose=True
            )

            with caplog.at_level(logging.INFO, logger="lionpride.operations.operate.react"):
                result = await react(session, branch, params)

            assert result.completed is True
            # Check that tool response logging occurred
            assert any("Tool test_tool:" in record.message for record in caplog.records)

    async def test_react_intermediate_options_as_dict(self, session_with_model):
        """Test intermediate_response_options as dict."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(
            name="test",
            capabilities=REACT_CAPABILITIES | {"intermediate_response_options"},
            resources={model.name},
        )

        class Progress(BaseModel):
            pct: int

        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.reasoning = "Done"
            mock_result.action_requests = None
            mock_result.action_responses = None
            # Return intermediate_response_options as a plain dict (not a model)
            mock_result.intermediate_response_options = {
                "progress": {"pct": 50},
                "status": "in_progress",
                "empty_field": None,  # Should be filtered out
            }
            mock_operate.return_value = mock_result

            params = ReactParams(
                generate=GenerateParams(
                    instruction="Test",
                    imodel=model,
                    imodel_kwargs={"model_name": "gpt-4"},
                ),
                parse=ParseParams(),
                actions=True,
                reason=True,
                max_steps=5,
                intermediate_response_options=[Progress],
            )

            result = await react(session, branch, params)

            assert result.completed is True
            # Dict intermediate options should be captured (with None values filtered)
            assert result.steps[0].intermediate_options == {
                "progress": {"pct": 50},
                "status": "in_progress",
            }
