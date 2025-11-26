# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for stateless generate() operation.

Focus areas:
- Parameter validation (imodel required)
- return_as variants (text, raw, message)
- Error handling (failed invocation)
- Stateless behavior (no message persistence)
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from lionpride import Event, EventStatus
from lionpride.operations.operate.generate import generate
from lionpride.session import Session
from lionpride.session.messages import Message


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


@pytest.fixture
def mock_model():
    """Create a mock iModel for testing without API calls."""
    from lionpride.services.providers.oai_chat import OAIChatEndpoint
    from lionpride.services.types.imodel import iModel

    # Create mock endpoint
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
def session_with_model(mock_model):
    """Create session with registered mock model."""
    session = Session()
    session.services.register(mock_model, update=True)
    return session, mock_model


class TestGenerateValidation:
    """Test parameter validation."""

    async def test_missing_imodel_raises_error(self, session_with_model):
        """Test that missing imodel parameter raises ValueError."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {"instruction": "test"}

        with pytest.raises(ValueError, match="No imodel specified"):
            await generate(session, branch, parameters)


class TestReturnAsVariants:
    """Test return_as parameter behavior."""

    async def test_return_as_calling_default(self, session_with_model):
        """Test default return_as='calling' returns Event object."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "imodel": "mock_model",
            "instruction": "test",
        }

        result = await generate(session, branch, parameters)

        # Default return_as is "calling" which returns the Event
        assert isinstance(result, Event)
        assert result.status == EventStatus.COMPLETED

    async def test_return_as_text_explicit(self, session_with_model):
        """Test explicit return_as='text'."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "imodel": "mock_model",
            "return_as": "text",
            "instruction": "test",
        }

        result = await generate(session, branch, parameters)

        assert isinstance(result, str)
        assert result == "mock response text"

    async def test_return_as_raw(self, session_with_model):
        """Test return_as='raw' returns full raw API response."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "imodel": "mock_model",
            "return_as": "raw",
            "instruction": "test",
        }

        result = await generate(session, branch, parameters)

        assert isinstance(result, dict)
        assert "id" in result
        assert "choices" in result

    async def test_return_as_message(self, session_with_model):
        """Test return_as='message' returns Message with metadata."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "imodel": "mock_model",
            "return_as": "message",
            "instruction": "test",
        }

        result = await generate(session, branch, parameters)

        assert isinstance(result, Message)
        # Verify content
        assert hasattr(result.content, "assistant_response")
        assert result.content.assistant_response == "mock response text"
        # Verify metadata preserved
        assert "raw_response" in result.metadata
        assert "usage" in result.metadata

    async def test_invalid_return_as_raises_error(self, session_with_model):
        """Test invalid return_as raises ValueError."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "imodel": "mock_model",
            "return_as": "invalid",
            "instruction": "test",
        }

        with pytest.raises(ValueError, match="Unsupported return_as"):
            await generate(session, branch, parameters)


class TestMessageBehavior:
    """Test message handling behavior."""

    async def test_instruction_added_to_branch(self, session_with_model):
        """Test that generate adds instruction message to branch."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        # Verify branch starts empty
        initial_count = len(session.messages[branch])
        assert initial_count == 0

        parameters = {
            "imodel": "mock_model",
            "instruction": "test",
        }

        await generate(session, branch, parameters)

        # Verify instruction message was added
        final_count = len(session.messages[branch])
        assert final_count == 1


class TestErrorHandling:
    """Test error handling for various failure modes."""

    async def test_failed_invocation_returns_failed_calling(self, session_with_model):
        """Test failed invocation returns Calling with FAILED status."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        async def mock_invoke_failed(**kwargs):
            class FailedCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.FAILED
                    self.execution.error = "API error occurred"

            return FailedCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_failed))

        parameters = {
            "imodel": "mock_model",
            "instruction": "test",
            "return_as": "calling",
        }

        # With return_as="calling", failed invocation returns the Calling object
        result = await generate(session, branch, parameters)
        assert isinstance(result, Event)
        assert result.status == EventStatus.FAILED
        assert result.execution.error == "API error occurred"

    async def test_failed_invocation_text_access_raises_attribute_error(self, session_with_model):
        """Test accessing text from failed invocation raises AttributeError."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        async def mock_invoke_failed(**kwargs):
            class FailedCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.FAILED
                    self.execution.error = "API error occurred"

                @property
                def response(self):
                    from lionpride.types import Unset

                    return Unset

            return FailedCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_failed))

        parameters = {
            "imodel": "mock_model",
            "instruction": "test",
            "return_as": "text",
        }

        # When invocation fails, response is Unset, accessing .data raises AttributeError
        with pytest.raises(AttributeError):
            await generate(session, branch, parameters)


class TestParameterPassthrough:
    """Test that parameters are passed through to imodel.invoke()."""

    async def test_parameters_forwarded_to_invoke(self, session_with_model):
        """Test that extra parameters are forwarded to imodel.invoke()."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "imodel": "mock_model",
            "instruction": "test",
            "imodel_kwargs": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100,
            },
        }

        await generate(session, branch, parameters)

        # Verify invoke was called with forwarded parameters
        model.invoke.assert_called_once()
        call_kwargs = model.invoke.call_args.kwargs
        # messages are constructed internally from instruction
        assert "messages" in call_kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100


class TestGenerateCoverage:
    """Test generate.py uncovered lines."""

    async def test_return_as_calling(self, session_with_model):
        """Test line 59: return_as='calling' returns the Calling object."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        parameters = {
            "imodel": "mock_model",
            "return_as": "calling",
            "instruction": "test",
        }

        result = await generate(session, branch, parameters)

        # Should return Calling object (Event subclass)
        assert isinstance(result, Event)
        assert result.status == EventStatus.COMPLETED
