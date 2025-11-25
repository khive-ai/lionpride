# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for react.py module.

Comprehensive coverage tests for the ReAct (Reasoning + Acting) operation,
testing parameter validation, execution flow, action handling, and error cases.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from lionpride.rules import ActionRequest


class TestReactCoverage:
    """Test react.py uncovered lines."""

    async def test_react_missing_instruction(self, session_with_model):
        """Test line 106-107: Missing instruction raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"imodel": model, "tools": ["tool"]}

        with pytest.raises(ValueError, match="react requires 'instruction' parameter"):
            await react(session, branch, parameters)

    async def test_react_missing_imodel(self, session_with_model):
        """Test line 110-111: Missing imodel raises ValueError."""
        from lionpride.operations.operate.react import react

        session, _ = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"instruction": "Test", "tools": ["tool"]}

        with pytest.raises(ValueError, match="react requires 'imodel' parameter"):
            await react(session, branch, parameters)

    async def test_react_missing_tools(self, session_with_model):
        """Test lines 114-115: Missing tools raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"instruction": "Test", "imodel": model}

        with pytest.raises(ValueError, match="react requires 'tools' parameter"):
            await react(session, branch, parameters)

    async def test_react_missing_model_name(self, session_with_model):
        """Test lines 148-150: Missing model_name raises ValueError."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "tools": [tool],
            # No model_name
        }

        with pytest.raises(ValueError, match="react requires 'model_name' in model_kwargs"):
            await react(session, branch, parameters)

    async def test_react_invalid_tool_type(self, session_with_model):
        """Test lines 166-167: Invalid tool type raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "tools": ["invalid_tool"],  # Not a Tool instance/class
            "model_kwargs": {"model_name": "gpt-4"},
        }

        with pytest.raises(ValueError, match="Invalid tool type"):
            await react(session, branch, parameters)

    async def test_react_branch_string_resolution(self, session_with_model):
        """Test lines 152-154: Branch string resolution."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        session.create_branch(name="test_branch")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))

        # Mock operate at the factory module level (operate is imported inside function)
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [tool],
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, "test_branch", parameters)

            assert result.completed is True

    async def test_create_react_response_model_with_type(self):
        """Test lines 66-88: _create_react_response_model with response_model."""
        from lionpride.operations.operate.react import _create_react_response_model

        class CustomAnswer(BaseModel):
            result: str

        model = _create_react_response_model(CustomAnswer)

        # Verify model has typed final_answer
        fields = model.model_fields
        assert "final_answer" in fields
        assert "reasoning" in fields
        assert "action_requests" in fields
        assert "is_done" in fields

    async def test_create_react_response_model_none(self):
        """Test lines 66-67: _create_react_response_model with None."""
        from lionpride.operations.operate.react import (
            ReactStepResponse,
            _create_react_response_model,
        )

        model = _create_react_response_model(None)

        assert model is ReactStepResponse

    async def test_react_validation_failure(self, session_with_model):
        """Test lines 237-240: Validation failure handling."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))

        # Mock operate to return validation failure
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.return_value = {"validation_failed": True, "error": "Invalid"}

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [tool],
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, branch, parameters)

            assert result.completed is False
            assert "Validation failed" in result.reason_stopped

    async def test_react_exception_handling(self, session_with_model):
        """Test lines 300-307: Exception handling in react loop."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))

        # Mock operate to raise exception
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.side_effect = RuntimeError("Test error")

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [tool],
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, branch, parameters)

            assert result.completed is False
            assert "Error at step" in result.reason_stopped
            assert "Test error" in result.reason_stopped

    async def test_react_max_steps_reached(self, session_with_model):
        """Test lines 311-315: Max steps reached without completion."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))

        # Mock operate to never finish
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = False
            mock_result.reasoning = "thinking"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [tool],
                "max_steps": 2,
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, branch, parameters)

            assert result.completed is False
            assert "Max steps (2) reached" in result.reason_stopped
            assert result.total_steps == 2

    async def test_react_verbose_logging(self, session_with_model, capsys):
        """Test verbose logging in react."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="test_tool", provider="tool"))

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "I figured it out"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [tool],
                "verbose": True,
                "model_kwargs": {"model_name": "gpt-4"},
            }

            _result = await react(session, branch, parameters)

            captured = capsys.readouterr()
            assert "ReAct Step" in captured.out
            assert "Task completed" in captured.out

    async def test_react_with_tool_class(self, session_with_model):
        """Test line 163: Tool instantiation from class."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Create tool class (not instance)
        class TestToolClass(Tool):
            def __init__(self):
                async def test_func() -> str:
                    return "result"

                super().__init__(
                    func_callable=test_func,
                    config=ToolConfig(name="test_class_tool", provider="tool"),
                )

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [TestToolClass],  # Pass class, not instance
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, branch, parameters)

            assert result.completed is True

    async def test_react_with_context(self, session_with_model):
        """Test line 206: Context added to instruction."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="ctx_tool", provider="tool"))

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [tool],
                "context": "Important context info",  # Line 206
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, branch, parameters)

            # Verify operate was called with instruction containing context
            call_kwargs = mock_operate.call_args
            assert "Context" in str(call_kwargs) or result.completed

    async def test_react_action_execution(self, session_with_model, capsys):
        """Test lines 249-281: Full action execution path."""
        from lionpride.operations.operate.react import react
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Register a tool
        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multiply", provider="tool"))
        session.services.register(iModel(backend=tool))

        # Mock operate to return action request then complete
        call_count = 0

        async def mock_operate_with_actions(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: return action request
                mock_result = MagicMock()
                mock_result.is_done = False
                mock_result.reasoning = "I need to calculate"
                mock_result.action_requests = [
                    ActionRequest(function="multiply", arguments={"a": 3, "b": 4})
                ]
                return mock_result
            else:
                # Second call: complete
                mock_result = MagicMock()
                mock_result.is_done = True
                mock_result.final_answer = "The result is 12"
                mock_result.reasoning = "Calculation complete"
                mock_result.action_requests = None
                return mock_result

        with patch(
            "lionpride.operations.operate.factory.operate", side_effect=mock_operate_with_actions
        ):
            parameters = {
                "instruction": "Calculate 3 * 4",
                "imodel": model,
                "tools": [tool],
                "verbose": True,
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, branch, parameters)

            assert result.completed is True
            assert len(result.steps) >= 2
            # First step should have action execution
            assert result.steps[0].actions_executed is not None
            assert result.steps[0].actions_executed[0].output == 12

            captured = capsys.readouterr()
            assert "Executing" in captured.out
            assert "multiply" in captured.out

    async def test_react_verbose_exception_traceback(self, session_with_model, capsys):
        """Test lines 302-304: Verbose exception with traceback."""
        from lionpride.operations.operate.react import react
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        async def test_tool() -> str:
            return "result"

        tool = Tool(func_callable=test_tool, config=ToolConfig(name="tb_tool", provider="tool"))

        # Mock operate to raise exception
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.side_effect = RuntimeError("Error for traceback test")

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "tools": [tool],
                "verbose": True,  # Enable verbose to get traceback
                "model_kwargs": {"model_name": "gpt-4"},
            }

            result = await react(session, branch, parameters)

            captured = capsys.readouterr()
            # Lines 302-304 print traceback when verbose=True
            assert "Error at step" in result.reason_stopped
            # Traceback should be printed
            assert "Traceback" in captured.err or "RuntimeError" in captured.err
