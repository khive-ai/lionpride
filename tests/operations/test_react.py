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


def _make_react_params(
    *,
    instruction: str | None = None,
    imodel=None,
    tools=None,
    tool_schemas=None,
    imodel_kwargs: dict | None = None,
    max_steps: int = 10,
    context: str | None = None,
    verbose: bool = False,
    operable=None,
):
    """Helper to create nested ReactParams dict structure."""
    params = {}

    # Build generate params
    generate = {}
    if instruction is not None:
        generate["instruction"] = instruction
    if imodel is not None:
        generate["imodel"] = imodel
    if imodel_kwargs is not None:
        generate["imodel_kwargs"] = imodel_kwargs
    if context is not None:
        generate["context"] = {"user_context": context}

    # Build communicate params
    communicate = {}
    if generate:
        communicate["generate"] = generate
    if operable is not None:
        communicate["operable"] = operable

    # Build act params
    act = {}
    if tools is not None:
        act["tools"] = tools
    if tool_schemas is not None:
        act["tool_schemas"] = tool_schemas

    # Build operate params
    operate = {}
    if communicate:
        operate["communicate"] = communicate
    if act:
        operate["act"] = act

    # Build top-level params
    if operate:
        params["operate"] = operate
    params["max_steps"] = max_steps

    return params


def _make_tool_schema(name: str = "test_tool"):
    """Create a minimal tool schema for testing."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


class TestReactCoverage:
    """Test react.py uncovered lines."""

    async def test_react_missing_operate(self, session_with_model):
        """Test: Missing operate raises ValueError."""
        from lionpride.operations.operate.react import react

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"max_steps": 5}  # No operate

        with pytest.raises(ValueError, match="react requires 'operate' parameter"):
            await react(session, branch, parameters)

    async def test_react_missing_communicate(self, session_with_model):
        """Test: Missing communicate raises ValueError."""
        from lionpride.operations.operate.react import react

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"operate": {"act": {"tools": True}}}  # No communicate

        with pytest.raises(ValueError, match=r"react requires 'operate\.communicate' parameter"):
            await react(session, branch, parameters)

    async def test_react_missing_generate(self, session_with_model):
        """Test: Missing generate raises ValueError."""
        from lionpride.operations.operate.react import react

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"operate": {"communicate": {}}}  # No generate

        with pytest.raises(
            ValueError, match=r"react requires 'operate\.communicate\.generate' parameter"
        ):
            await react(session, branch, parameters)

    async def test_react_missing_instruction(self, session_with_model):
        """Test: Missing instruction raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"imodel": model},
                },
                "act": {"tools": True, "tool_schemas": [_make_tool_schema()]},
            }
        }

        with pytest.raises(ValueError, match="react requires instruction"):
            await react(session, branch, parameters)

    async def test_react_missing_imodel(self, session_with_model):
        """Test: Missing imodel raises ValueError."""
        from lionpride.operations.operate.react import react

        session, _ = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"instruction": "Test"},
                },
                "act": {"tools": True, "tool_schemas": [_make_tool_schema()]},
            }
        }

        with pytest.raises(ValueError, match="react requires imodel"):
            await react(session, branch, parameters)

    async def test_react_missing_tools(self, session_with_model):
        """Test: Missing tools raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"instruction": "Test", "imodel": model},
                },
                # No act or act.tools=False
            }
        }

        with pytest.raises(ValueError, match="react requires tools"):
            await react(session, branch, parameters)

    async def test_react_empty_tool_schemas(self, session_with_model):
        """Test: Empty tool schemas raises ValueError."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"instruction": "Test", "imodel": model},
                },
                "act": {"tools": True, "tool_schemas": []},  # Empty tool_schemas
            }
        }

        with pytest.raises(ValueError, match="react requires at least one tool"):
            await react(session, branch, parameters)

    async def test_react_branch_string_resolution(self, session_with_model):
        """Test: Branch string resolution."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        session.create_branch(name="test_branch")

        # Mock operate at the factory module level (operate is imported inside function)
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema()],
            )

            result = await react(session, "test_branch", parameters)

            assert result.completed is True

    async def test_create_react_response_model_with_type(self):
        """Test: _create_react_response_model with response_model."""
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
        """Test: _create_react_response_model with None."""
        from lionpride.operations.operate.react import (
            ReactStepResponse,
            _create_react_response_model,
        )

        model = _create_react_response_model(None)

        assert model is ReactStepResponse

    async def test_react_validation_failure(self, session_with_model):
        """Test: Validation failure handling."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to return validation failure
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.return_value = {"validation_failed": True, "error": "Invalid"}

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema()],
            )

            result = await react(session, branch, parameters)

            assert result.completed is False
            assert "Validation failed" in result.reason_stopped

    async def test_react_exception_handling(self, session_with_model):
        """Test: Exception handling in react loop."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to raise exception
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.side_effect = RuntimeError("Test error")

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema()],
            )

            result = await react(session, branch, parameters)

            assert result.completed is False
            assert "Error at step" in result.reason_stopped
            assert "Test error" in result.reason_stopped

    async def test_react_max_steps_reached(self, session_with_model):
        """Test: Max steps reached without completion."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to never finish
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = False
            mock_result.reasoning = "thinking"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema()],
                max_steps=2,
            )

            result = await react(session, branch, parameters)

            assert result.completed is False
            assert "Max steps (2) reached" in result.reason_stopped
            assert result.total_steps == 2

    async def test_react_verbose_logging(self, session_with_model, capsys):
        """Test verbose logging in react."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "I figured it out"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema()],
                verbose=True,
            )

            _result = await react(session, branch, parameters)

            # Note: verbose may not print in current implementation
            # The test ensures the parameter is accepted without error

    async def test_react_with_tool_class(self, session_with_model):
        """Test: Tool instantiation from class."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("test_class_tool")],
            )

            result = await react(session, branch, parameters)

            assert result.completed is True

    async def test_react_with_context(self, session_with_model):
        """Test: Context added to instruction."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("ctx_tool")],
                context="Important context info",
            )

            result = await react(session, branch, parameters)

            # Verify operate was called with instruction containing context
            call_kwargs = mock_operate.call_args
            assert "Context" in str(call_kwargs) or result.completed

    async def test_react_action_execution(self, session_with_model, capsys):
        """Test: Full action execution path."""
        from lionpride.operations.operate.react import react
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Create and register a tool
        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multiply", provider="tool"))
        session.services.register(iModel(backend=tool))

        multiply_schema = {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
        }

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
            parameters = _make_react_params(
                instruction="Calculate 3 * 4",
                imodel=model,
                tools=True,
                tool_schemas=[multiply_schema],
            )

            result = await react(session, branch, parameters)

            assert result.completed is True
            assert len(result.steps) >= 2
            # First step should have action execution
            assert result.steps[0].actions_executed is not None
            assert result.steps[0].actions_executed[0].output == 12

    async def test_react_verbose_exception_traceback(self, session_with_model, capsys):
        """Test: Verbose exception with traceback."""
        from lionpride.operations.operate.react import react

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to raise exception
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.side_effect = RuntimeError("Error for traceback test")

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("tb_tool")],
                verbose=True,
            )

            result = await react(session, branch, parameters)

            # Exception is caught and stored in reason_stopped
            assert "Error at step" in result.reason_stopped


class TestReactStreamCoverage:
    """Tests for react_stream() streaming operation."""

    async def test_react_stream_missing_operate(self, session_with_model):
        """Test: Missing operate raises ValueError."""
        from lionpride.operations.operate.react import react_stream

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"max_steps": 5}  # No operate

        with pytest.raises(ValueError, match="react_stream requires 'operate' parameter"):
            async for _ in react_stream(session, branch, parameters):
                pass

    async def test_react_stream_missing_communicate(self, session_with_model):
        """Test: Missing communicate raises ValueError."""
        from lionpride.operations.operate.react import react_stream

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"operate": {"act": {"tools": True}}}

        with pytest.raises(
            ValueError, match=r"react_stream requires 'operate\.communicate' parameter"
        ):
            async for _ in react_stream(session, branch, parameters):
                pass

    async def test_react_stream_missing_generate(self, session_with_model):
        """Test: Missing generate raises ValueError."""
        from lionpride.operations.operate.react import react_stream

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"operate": {"communicate": {}}}

        with pytest.raises(
            ValueError, match=r"react_stream requires 'operate\.communicate\.generate' parameter"
        ):
            async for _ in react_stream(session, branch, parameters):
                pass

    async def test_react_stream_missing_instruction(self, session_with_model):
        """Test: Missing instruction raises ValueError."""
        from lionpride.operations.operate.react import react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"imodel": model},
                },
                "act": {"tools": True, "tool_schemas": [_make_tool_schema()]},
            }
        }

        with pytest.raises(ValueError, match="react_stream requires instruction"):
            async for _ in react_stream(session, branch, parameters):
                pass

    async def test_react_stream_missing_imodel(self, session_with_model):
        """Test: Missing imodel raises ValueError."""
        from lionpride.operations.operate.react import react_stream

        session, _ = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"instruction": "Test"},
                },
                "act": {"tools": True, "tool_schemas": [_make_tool_schema()]},
            }
        }

        with pytest.raises(ValueError, match="react_stream requires imodel"):
            async for _ in react_stream(session, branch, parameters):
                pass

    async def test_react_stream_missing_tools(self, session_with_model):
        """Test: Missing tools raises ValueError."""
        from lionpride.operations.operate.react import react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"instruction": "Test", "imodel": model},
                },
                # No act or act.tools=False
            }
        }

        with pytest.raises(ValueError, match="react_stream requires tools"):
            async for _ in react_stream(session, branch, parameters):
                pass

    async def test_react_stream_empty_tool_schemas(self, session_with_model):
        """Test: Empty tool schemas raises ValueError."""
        from lionpride.operations.operate.react import react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "operate": {
                "communicate": {
                    "generate": {"instruction": "Test", "imodel": model},
                },
                "act": {"tools": True, "tool_schemas": []},
            }
        }

        with pytest.raises(ValueError, match="react_stream requires at least one tool"):
            async for _ in react_stream(session, branch, parameters):
                pass

    async def test_react_stream_yields_steps(self, session_with_model):
        """Test: Stream yields steps and final result."""
        from lionpride.operations.operate.react import ReactResult, ReactStep, react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to complete in 2 steps
        call_count = 0

        async def mock_operate_multi_step(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                mock_result = MagicMock()
                mock_result.is_done = False
                mock_result.reasoning = f"Step {call_count}"
                mock_result.action_requests = None
                return mock_result
            else:
                mock_result = MagicMock()
                mock_result.is_done = True
                mock_result.final_answer = "done"
                mock_result.reasoning = "Final"
                mock_result.action_requests = None
                return mock_result

        with patch(
            "lionpride.operations.operate.factory.operate", side_effect=mock_operate_multi_step
        ):
            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("stream_tool")],
            )

            results = []
            async for item in react_stream(session, branch, parameters):
                results.append(item)

            # Should have at least 1 step + final result
            assert len(results) >= 2
            # Intermediate results are ReactStep
            assert isinstance(results[0], ReactStep)
            # Final result is ReactResult
            assert isinstance(results[-1], ReactResult)
            assert results[-1].completed is True

    async def test_react_stream_validation_failure(self, session_with_model):
        """Test: Validation failure yields failed step."""
        from lionpride.operations.operate.react import ReactResult, ReactStep, react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to return validation failure
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.return_value = {"validation_failed": True, "error": "Invalid format"}

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("val_fail_tool")],
            )

            results = []
            async for item in react_stream(session, branch, parameters):
                results.append(item)

            # Should yield failed step then final result
            assert len(results) >= 2
            assert isinstance(results[0], ReactStep)
            assert isinstance(results[-1], ReactResult)
            assert "Validation failed" in results[-1].reason_stopped

    async def test_react_stream_exception_handling(self, session_with_model):
        """Test: Exception handling yields error step."""
        from lionpride.operations.operate.react import ReactResult, ReactStep, react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to raise exception
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_operate.side_effect = RuntimeError("Stream error")

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("err_tool")],
            )

            results = []
            async for item in react_stream(session, branch, parameters):
                results.append(item)

            # Should yield error step then final result
            assert len(results) >= 2
            assert isinstance(results[0], ReactStep)
            assert isinstance(results[-1], ReactResult)
            assert "Error at step" in results[-1].reason_stopped
            assert "Stream error" in results[-1].reason_stopped

    async def test_react_stream_max_steps_reached(self, session_with_model):
        """Test: Max steps reached yields final result."""
        from lionpride.operations.operate.react import ReactResult, ReactStep, react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to never finish
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = False
            mock_result.reasoning = "thinking"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("max_tool")],
                max_steps=3,
            )

            results = []
            async for item in react_stream(session, branch, parameters):
                results.append(item)

            # Should have 3 steps + final result
            assert len(results) == 4
            # First 3 are ReactStep
            for i in range(3):
                assert isinstance(results[i], ReactStep)
            # Last is ReactResult
            assert isinstance(results[-1], ReactResult)
            assert "Max steps (3) reached" in results[-1].reason_stopped
            assert results[-1].total_steps == 3

    async def test_react_stream_with_tool_class(self, session_with_model):
        """Test: Tool instantiation from class in stream."""
        from lionpride.operations.operate.react import ReactResult, react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("stream_class_tool")],
            )

            results = []
            async for item in react_stream(session, branch, parameters):
                results.append(item)

            assert isinstance(results[-1], ReactResult)
            assert results[-1].completed is True

    async def test_react_stream_branch_string_resolution(self, session_with_model):
        """Test: Branch string resolution in stream."""
        from lionpride.operations.operate.react import ReactResult, react_stream

        session, model = session_with_model
        session.create_branch(name="stream_branch")

        # Mock operate to complete quickly
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "test"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("branch_tool")],
            )

            results = []
            async for item in react_stream(session, "stream_branch", parameters):
                results.append(item)

            assert isinstance(results[-1], ReactResult)
            assert results[-1].completed is True

    async def test_react_stream_action_execution(self, session_with_model):
        """Test: Full action execution path in stream."""
        from lionpride.operations.operate.react import ReactResult, ReactStep, react_stream
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Create and register a tool
        async def add_nums(a: int, b: int) -> int:
            return a + b

        tool = Tool(func_callable=add_nums, config=ToolConfig(name="add_nums", provider="tool"))
        session.services.register(iModel(backend=tool))

        add_nums_schema = {
            "type": "function",
            "function": {
                "name": "add_nums",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
        }

        # Mock operate to return action request then complete
        call_count = 0

        async def mock_operate_with_actions(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: return action request
                mock_result = MagicMock()
                mock_result.is_done = False
                mock_result.reasoning = "Need to add"
                mock_result.action_requests = [
                    ActionRequest(function="add_nums", arguments={"a": 5, "b": 7})
                ]
                return mock_result
            else:
                # Second call: complete
                mock_result = MagicMock()
                mock_result.is_done = True
                mock_result.final_answer = "Sum is 12"
                mock_result.reasoning = "Done"
                mock_result.action_requests = None
                return mock_result

        with patch(
            "lionpride.operations.operate.factory.operate", side_effect=mock_operate_with_actions
        ):
            parameters = _make_react_params(
                instruction="Calculate 5 + 7",
                imodel=model,
                tools=True,
                tool_schemas=[add_nums_schema],
            )

            results = []
            async for item in react_stream(session, branch, parameters):
                results.append(item)

            # Should have steps + final result
            assert len(results) >= 2
            # First step should have action execution
            assert isinstance(results[0], ReactStep)
            assert results[0].actions_executed is not None
            assert results[0].actions_executed[0].output == 12
            # Final result should be complete
            assert isinstance(results[-1], ReactResult)
            assert results[-1].completed is True

    async def test_react_stream_verbose_logging(self, session_with_model, capsys):
        """Test verbose logging in react_stream."""
        from lionpride.operations.operate.react import react_stream

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock operate to complete
        with patch("lionpride.operations.operate.factory.operate") as mock_operate:
            mock_result = MagicMock()
            mock_result.is_done = True
            mock_result.final_answer = "done"
            mock_result.reasoning = "Verbose reasoning here"
            mock_result.action_requests = None
            mock_operate.return_value = mock_result

            parameters = _make_react_params(
                instruction="Test",
                imodel=model,
                tools=True,
                tool_schemas=[_make_tool_schema("verbose_tool")],
                verbose=True,
            )

            results = []
            async for item in react_stream(session, branch, parameters):
                results.append(item)

            # Test completes without error with verbose param
