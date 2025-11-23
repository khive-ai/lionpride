# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Functional tests for Branch convenience methods: generate(), operate(), react().

Coverage targets:
- Session binding (_bind_session, _require_session)
- Branch.generate() - basic generation, context, return_message
- Branch.operate() - structured output, LNDL, reason, actions
- Branch.react() - multi-step loop, tool execution, termination conditions
- Error handling for unbound branches
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from lionpride import Event, EventStatus
from lionpride.services.providers.oai_chat import OAIChatEndpoint
from lionpride.services.types.imodel import iModel
from lionpride.services.types.tool import Tool, ToolConfig
from lionpride.session import Branch, Session
from lionpride.session.messages import Message

# =============================================================================
# Test Models
# =============================================================================


class Analysis(BaseModel):
    """Simple response model for testing operate()."""

    summary: str = Field(description="Summary of the analysis")
    confidence: float = Field(description="Confidence score", default=0.5)


class FinalAnswer(BaseModel):
    """Response model for react() final output."""

    answer: str = Field(description="The final answer")
    confidence: float = Field(description="Confidence level", default=0.9)


# =============================================================================
# Mock Tools for ReAct Testing
# =============================================================================


async def calculator_add(a: int, b: int) -> dict:
    """Add two numbers."""
    return {"result": a + b}


async def calculator_multiply(a: int, b: int) -> dict:
    """Multiply two numbers."""
    return {"result": a * b}


def create_calculator_tool() -> Tool:
    """Create a mock calculator tool for testing."""
    return Tool(
        func_callable=calculator_add,
        config=ToolConfig(name="calculator", provider="tool"),
    )


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockResponse:
    """Mock response for model invocation."""

    status: str = "success"
    data: str | dict = ""
    raw_response: dict = None
    metadata: dict = None

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {"id": "mock-id", "choices": []}
        if self.metadata is None:
            self.metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}


@pytest.fixture
def mock_model():
    """Create a mock iModel for testing without API calls."""
    endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")
    model = iModel(backend=endpoint)

    async def mock_invoke(model_name: str | None = None, messages: list | None = None, **kwargs):
        class MockCalling(Event):
            def __init__(self, response_data: str | dict):
                super().__init__()
                self.status = EventStatus.COMPLETED
                self.execution.response = MockResponse(status="success", data=response_data)

        response = kwargs.get("_test_response", "mock response")
        return MockCalling(response)

    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

    return model


@pytest.fixture
def mock_model_json():
    """Create mock model that returns JSON matching Analysis model."""
    endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")
    model = iModel(backend=endpoint)

    async def mock_invoke(model_name: str | None = None, messages: list | None = None, **kwargs):
        class MockCalling(Event):
            def __init__(self, response_data: str):
                super().__init__()
                self.status = EventStatus.COMPLETED
                self.execution.response = MockResponse(status="success", data=response_data)

        # Return JSON matching Analysis model directly (for non-LNDL)
        response = '{"summary": "Test summary", "confidence": 0.85}'
        return MockCalling(response)

    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

    return model


@pytest.fixture
def mock_model_react_done():
    """Create mock model that returns is_done=True (single step completion)."""
    endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")
    model = iModel(backend=endpoint)

    async def mock_invoke(model_name: str | None = None, messages: list | None = None, **kwargs):
        class MockCalling(Event):
            def __init__(self, response_data: str):
                super().__init__()
                self.status = EventStatus.COMPLETED
                self.execution.response = MockResponse(status="success", data=response_data)

        # Return response indicating task is done
        response = """{
            "reasoning": "I know the answer directly",
            "action_requests": [],
            "is_done": true,
            "final_answer": "Paris is the capital of France"
        }"""
        return MockCalling(response)

    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

    return model


@pytest.fixture
def mock_model_react_tool():
    """Create mock model that requests tool call then completes."""
    endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")
    model = iModel(backend=endpoint)
    call_count = [0]

    async def mock_invoke(model_name: str | None = None, messages: list | None = None, **kwargs):
        class MockCalling(Event):
            def __init__(self, response_data: str):
                super().__init__()
                self.status = EventStatus.COMPLETED
                self.execution.response = MockResponse(status="success", data=response_data)

        call_count[0] += 1

        if call_count[0] == 1:
            # First call: request tool execution
            response = """{
                "reasoning": "I need to calculate 5 + 3",
                "action_requests": [
                    {"function": "calculator", "arguments": {"a": 5, "b": 3}}
                ],
                "is_done": false,
                "final_answer": null
            }"""
        else:
            # Second call: return final answer
            response = """{
                "reasoning": "Now I have the result: 8",
                "action_requests": [],
                "is_done": true,
                "final_answer": "The result of 5 + 3 is 8"
            }"""

        return MockCalling(response)

    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

    return model


@pytest.fixture
def mock_model_react_max_steps():
    """Create mock model that never returns is_done=True (for max_steps test)."""
    endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")
    model = iModel(backend=endpoint)
    call_count = [0]

    async def mock_invoke(model_name: str | None = None, messages: list | None = None, **kwargs):
        class MockCalling(Event):
            def __init__(self, response_data: str):
                super().__init__()
                self.status = EventStatus.COMPLETED
                self.execution.response = MockResponse(status="success", data=response_data)

        call_count[0] += 1

        # Always request more actions, never complete
        response = f"""{{
            "reasoning": "Step {call_count[0]}: I need more information",
            "action_requests": [
                {{"function": "calculator", "arguments": {{"a": {call_count[0]}, "b": 1}}}}
            ],
            "is_done": false,
            "final_answer": null
        }}"""

        return MockCalling(response)

    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

    return model


@pytest.fixture
def session_with_branch(mock_model):
    """Create session with registered mock model and bound branch."""
    session = Session()
    session.services.register(mock_model, update=True)
    branch = session.create_branch(name="test")
    return session, branch, mock_model


# =============================================================================
# Test Session Binding
# =============================================================================


class TestSessionBinding:
    """Test Branch session binding functionality."""

    def test_create_branch_auto_binds_session(self):
        """Test that Session.create_branch() automatically binds session reference."""
        session = Session()
        branch = session.create_branch(name="test")

        assert branch._session is session

    def test_fork_auto_binds_session(self):
        """Test that Session.fork() automatically binds session reference."""
        session = Session()
        original = session.create_branch(name="original")
        forked = session.fork(original, name="forked")

        assert forked._session is session

    def test_unbound_branch_raises_on_require_session(self):
        """Test that _require_session() raises RuntimeError for unbound branch."""
        session = Session()
        unbound = Branch(session_id=session.id, user="test")

        with pytest.raises(RuntimeError, match="Branch not bound to session"):
            unbound._require_session()

    def test_bind_session_returns_branch(self):
        """Test that _bind_session() returns the branch for chaining."""
        session = Session()
        branch = Branch(session_id=session.id, user="test")

        result = branch._bind_session(session)

        assert result is branch
        assert branch._session is session

    def test_branch_repr_shows_bound_status(self):
        """Test that Branch repr shows bound status."""
        session = Session()
        unbound = Branch(session_id=session.id, user="test")
        bound = session.create_branch(name="bound")

        assert "bound" not in repr(unbound)
        assert "bound" in repr(bound)


# =============================================================================
# Test Branch.generate() - Stateless
# =============================================================================


class TestBranchGenerate:
    """Test Branch.generate() convenience method - stateless."""

    async def test_unbound_branch_raises_runtime_error(self, mock_model):
        """Test that generate() raises RuntimeError for unbound branch."""
        session = Session()
        unbound = Branch(session_id=session.id, user="test")

        with pytest.raises(RuntimeError, match="Branch not bound to session"):
            await unbound.generate(
                imodel=mock_model,
                messages=[{"role": "user", "content": "Hello"}],
            )

    async def test_basic_generation_stateless(self, session_with_branch):
        """Test basic text generation via Branch.generate() - no messages added."""
        _session, branch, model = session_with_branch

        result = await branch.generate(
            imodel=model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert isinstance(result, str)
        assert result == "mock response"
        # Verify NO messages added (stateless)
        assert len(branch) == 0

    async def test_return_as_message(self, session_with_branch):
        """Test return_as='message' returns Message with metadata."""
        _session, branch, model = session_with_branch

        result = await branch.generate(
            imodel=model,
            messages=[{"role": "user", "content": "Hello"}],
            return_as="message",
        )

        assert isinstance(result, Message)
        assert hasattr(result, "content")
        assert "raw_response" in result.metadata

    async def test_model_kwargs_passed_through(self, session_with_branch):
        """Test that model_kwargs are passed to model.invoke()."""
        _session, branch, model = session_with_branch

        await branch.generate(
            imodel=model,
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=100,
        )

        # Verify invoke was called with kwargs
        model.invoke.assert_called_once()
        call_kwargs = model.invoke.call_args[1]
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("max_tokens") == 100


# =============================================================================
# Test Branch.communicate() - Stateful
# =============================================================================


class TestBranchCommunicate:
    """Test Branch.communicate() convenience method - stateful."""

    async def test_unbound_branch_raises_runtime_error(self, mock_model):
        """Test that communicate() raises RuntimeError for unbound branch."""
        session = Session()
        unbound = Branch(session_id=session.id, user="test")

        with pytest.raises(RuntimeError, match="Branch not bound to session"):
            await unbound.communicate("Hello", imodel=mock_model)

    async def test_basic_communication(self, session_with_branch):
        """Test basic text communication via Branch.communicate()."""
        _session, branch, model = session_with_branch

        result = await branch.communicate("What is 2+2?", imodel=model)

        assert isinstance(result, str)
        assert result == "mock response"
        # Verify messages were added to branch (stateful)
        assert len(branch) == 2  # instruction + assistant

    async def test_communication_with_context(self, session_with_branch):
        """Test communication with context parameter."""
        _session, branch, model = session_with_branch

        result = await branch.communicate(
            "Summarize this data",
            imodel=model,
            context={"data": "some important data", "source": "test"},
        )

        assert isinstance(result, str)
        assert len(branch) == 2

    async def test_return_as_message(self, session_with_branch):
        """Test that return_as='message' returns Message object."""
        _session, branch, model = session_with_branch

        result = await branch.communicate("Hello", imodel=model, return_as="message")

        assert isinstance(result, Message)
        assert hasattr(result, "content")
        assert hasattr(result, "sender")

    async def test_return_as_text_default(self, session_with_branch):
        """Test that return_as='text' (default) returns string."""
        _session, branch, model = session_with_branch

        result = await branch.communicate("Hello", imodel=model, return_as="text")

        assert isinstance(result, str)

    async def test_model_kwargs_passed_through(self, session_with_branch):
        """Test that model_kwargs are passed to model.invoke()."""
        _session, branch, model = session_with_branch

        await branch.communicate(
            "Test",
            imodel=model,
            temperature=0.7,
            max_tokens=100,
        )

        # Verify invoke was called with kwargs
        model.invoke.assert_called_once()
        call_kwargs = model.invoke.call_args[1]
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("max_tokens") == 100


# =============================================================================
# Test Branch.operate()
# =============================================================================


class TestBranchOperate:
    """Test Branch.operate() convenience method."""

    async def test_unbound_branch_raises_runtime_error(self, mock_model_json):
        """Test that operate() raises RuntimeError for unbound branch."""
        session = Session()
        unbound = Branch(session_id=session.id, user="test")

        with pytest.raises(RuntimeError, match="Branch not bound to session"):
            await unbound.operate(
                "Analyze this",
                imodel=mock_model_json,
                response_model=Analysis,
                model_name="gpt-4.1-mini",
            )

    async def test_basic_structured_output(self, mock_model_json):
        """Test basic structured output via Branch.operate()."""
        session = Session()
        session.services.register(mock_model_json, update=True)
        branch = session.create_branch(name="test")

        result = await branch.operate(
            "Analyze this text",
            imodel=mock_model_json,
            response_model=Analysis,
            model_name="gpt-4.1-mini",
        )

        # Result should be parsed model or validation dict
        assert result is not None
        # Messages were added to branch
        assert len(branch) >= 2

    @pytest.mark.skip(
        reason="Mock response needs to match composite model structure with reason field"
    )
    async def test_operate_with_reason(self, mock_model_json):
        """Test operate() with reason=True includes reasoning field."""
        session = Session()
        session.services.register(mock_model_json, update=True)
        branch = session.create_branch(name="test")

        result = await branch.operate(
            "Analyze this",
            imodel=mock_model_json,
            response_model=Analysis,
            reason=True,
            model_name="gpt-4.1-mini",
        )

        # Should complete without error
        assert result is not None

    async def test_operate_with_context(self, mock_model_json):
        """Test operate() with context parameter."""
        session = Session()
        session.services.register(mock_model_json, update=True)
        branch = session.create_branch(name="test")

        result = await branch.operate(
            "Analyze this",
            imodel=mock_model_json,
            response_model=Analysis,
            context={"key": "value", "data": [1, 2, 3]},
            model_name="gpt-4.1-mini",
        )

        assert result is not None


# =============================================================================
# Test Branch.react()
# =============================================================================


class TestBranchReact:
    """Test Branch.react() convenience method for multi-step reasoning."""

    async def test_unbound_branch_raises_runtime_error(self, mock_model_react_done):
        """Test that react() raises RuntimeError for unbound branch."""
        session = Session()
        unbound = Branch(session_id=session.id, user="test")

        with pytest.raises(RuntimeError, match="Branch not bound to session"):
            await unbound.react(
                "What is the capital of France?",
                imodel=mock_model_react_done,
                tools=[create_calculator_tool()],
                model_name="gpt-4.1-mini",
            )

    async def test_single_step_completion(self, mock_model_react_done):
        """Test react() completes in single step when is_done=True."""
        from lionpride.operations.operate.react import ReactResult

        session = Session()
        session.services.register(mock_model_react_done, update=True)
        branch = session.create_branch(name="test")

        result = await branch.react(
            "What is the capital of France?",
            imodel=mock_model_react_done,
            tools=[create_calculator_tool()],
            max_steps=3,
            model_name="gpt-4.1-mini",
        )

        assert isinstance(result, ReactResult)
        assert result.completed is True
        assert result.total_steps == 1
        assert result.reason_stopped == "Task completed"
        assert len(result.steps) == 1
        assert result.steps[0].is_final is True

    async def test_multi_step_with_tool_execution(self, mock_model_react_tool):
        """Test react() executes tools and continues reasoning."""
        from lionpride.operations.operate.react import ReactResult

        session = Session()
        session.services.register(mock_model_react_tool, update=True)
        branch = session.create_branch(name="test")

        result = await branch.react(
            "Calculate 5 + 3",
            imodel=mock_model_react_tool,
            tools=[create_calculator_tool()],
            max_steps=5,
            model_name="gpt-4.1-mini",
        )

        assert isinstance(result, ReactResult)
        assert result.completed is True
        assert result.total_steps == 2
        assert result.reason_stopped == "Task completed"

        # First step should have action execution
        assert len(result.steps[0].actions_requested) > 0
        assert len(result.steps[0].actions_executed) > 0

        # Second step should be final
        assert result.steps[1].is_final is True

    async def test_max_steps_exhausted(self, mock_model_react_max_steps):
        """Test react() stops when max_steps is reached."""
        from lionpride.operations.operate.react import ReactResult

        session = Session()
        session.services.register(mock_model_react_max_steps, update=True)
        branch = session.create_branch(name="test")

        result = await branch.react(
            "Keep calculating forever",
            imodel=mock_model_react_max_steps,
            tools=[create_calculator_tool()],
            max_steps=3,
            model_name="gpt-4.1-mini",
        )

        assert isinstance(result, ReactResult)
        assert result.completed is False
        assert result.total_steps == 3
        assert "Max steps" in result.reason_stopped

    async def test_react_registers_tools_with_session(self, mock_model_react_done):
        """Test that tools are auto-registered with session.services."""
        session = Session()
        session.services.register(mock_model_react_done, update=True)
        branch = session.create_branch(name="test")

        # Tool not registered yet
        assert not session.services.has("calculator")

        await branch.react(
            "Calculate something",
            imodel=mock_model_react_done,
            tools=[create_calculator_tool()],
            model_name="gpt-4.1-mini",
        )

        # Tool should now be registered
        assert session.services.has("calculator")

    async def test_react_with_verbose(self, mock_model_react_done, capsys):
        """Test react() with verbose=True prints step info."""
        session = Session()
        session.services.register(mock_model_react_done, update=True)
        branch = session.create_branch(name="test")

        await branch.react(
            "What is the capital of France?",
            imodel=mock_model_react_done,
            tools=[create_calculator_tool()],
            verbose=True,
            model_name="gpt-4.1-mini",
        )

        captured = capsys.readouterr()
        assert "ReAct Step" in captured.out

    async def test_react_with_tool_instance(self, mock_model_react_done):
        """Test react() accepts tool instance."""
        session = Session()
        session.services.register(mock_model_react_done, update=True)
        branch = session.create_branch(name="test")

        # Pass tool instance
        result = await branch.react(
            "What is the capital of France?",
            imodel=mock_model_react_done,
            tools=[create_calculator_tool()],
            model_name="gpt-4.1-mini",
        )

        assert result.completed is True


# =============================================================================
# Test ReactResult and ReactStep Models
# =============================================================================


class TestReactModels:
    """Test ReactResult and ReactStep model structure."""

    def test_react_step_defaults(self):
        """Test ReactStep has correct default values."""
        from lionpride.operations.operate.react import ReactStep

        step = ReactStep(step=1)

        assert step.step == 1
        assert step.reasoning is None
        assert step.actions_requested == []
        assert step.actions_executed == []
        assert step.is_final is False

    def test_react_result_defaults(self):
        """Test ReactResult has correct default values."""
        from lionpride.operations.operate.react import ReactResult

        result = ReactResult()

        assert result.steps == []
        assert result.final_response is None
        assert result.total_steps == 0
        assert result.completed is False
        assert result.reason_stopped == ""


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in convenience methods."""

    async def test_generate_missing_model_raises(self):
        """Test generate() without model raises appropriate error."""
        session = Session()
        branch = session.create_branch(name="test")

        with pytest.raises(ValueError, match="model"):
            await branch.generate(imodel=None, messages=[{"role": "user", "content": "Hello"}])  # type: ignore

    async def test_communicate_missing_model_raises(self):
        """Test communicate() without model raises appropriate error."""
        session = Session()
        branch = session.create_branch(name="test")

        with pytest.raises(ValueError, match="model"):
            await branch.communicate("Hello", imodel=None)  # type: ignore

    async def test_react_missing_tools_raises(self, mock_model_react_done):
        """Test react() without tools raises ValueError."""
        session = Session()
        session.services.register(mock_model_react_done, update=True)
        branch = session.create_branch(name="test")

        with pytest.raises(ValueError, match="tools"):
            await branch.react(
                "Hello",
                imodel=mock_model_react_done,
                tools=[],  # Empty tools list
                model_name="gpt-4.1-mini",
            )

    async def test_react_missing_instruction_raises(self, mock_model_react_done):
        """Test react() without instruction raises ValueError."""
        session = Session()
        session.services.register(mock_model_react_done, update=True)
        branch = session.create_branch(name="test")

        with pytest.raises(ValueError, match="instruction"):
            await branch.react(
                "",  # Empty instruction
                imodel=mock_model_react_done,
                tools=[create_calculator_tool()],
                model_name="gpt-4.1-mini",
            )
