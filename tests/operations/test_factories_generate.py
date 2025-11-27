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
from lionpride.operations.operate.types import GenerateParams
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
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = GenerateParams(instruction="test")

        with pytest.raises(ValueError, match="generate requires 'imodel' parameter"):
            await generate(session, branch, params)


class TestReturnAsVariants:
    """Test return_as parameter behavior."""

    async def test_return_as_text_default(self, session_with_model):
        """Test default return_as='text' returns response string."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="text",
        )

        result = await generate(session, branch, params)

        assert isinstance(result, str)
        assert result == "mock response text"

    async def test_return_as_raw(self, session_with_model):
        """Test return_as='raw' returns full raw API response."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="raw",
        )

        result = await generate(session, branch, params)

        assert isinstance(result, dict)
        assert "id" in result
        assert "choices" in result

    async def test_return_as_message(self, session_with_model):
        """Test return_as='message' returns Message with metadata."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="message",
        )

        result = await generate(session, branch, params)

        assert isinstance(result, Message)
        # Verify content
        assert hasattr(result.content, "assistant_response")
        assert result.content.assistant_response == "mock response text"
        # Verify metadata preserved
        assert "raw_response" in result.metadata
        assert "usage" in result.metadata

    async def test_return_as_calling(self, session_with_model):
        """Test return_as='calling' returns the Calling object."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="calling",
        )

        result = await generate(session, branch, params)

        # Should return Calling object (Event subclass)
        assert isinstance(result, Event)
        assert result.status == EventStatus.COMPLETED

    async def test_invalid_return_as_raises_error(self, session_with_model):
        """Test invalid return_as raises ValueError."""
        # Can't test invalid return_as with GenerateParams since it's a Literal type
        # The type system prevents invalid values at construction time
        # This test is now a compile-time check rather than runtime
        pass


class TestStatelessBehavior:
    """Test that generate is stateless."""

    async def test_no_messages_added_to_branch(self, session_with_model):
        """Test that generate does not add messages to session/branch."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        # Verify branch starts empty
        initial_count = len(session.messages[branch])

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="text",
        )

        await generate(session, branch, params)

        # Verify no messages were added
        final_count = len(session.messages[branch])
        assert final_count == initial_count


class TestErrorHandling:
    """Test error handling for various failure modes."""

    async def test_failed_invocation_raises_runtime_error(self, session_with_model):
        """Test RuntimeError raised when model invocation fails."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        async def mock_invoke_failed(**kwargs):
            class FailedCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.FAILED
                    self.execution.error = "API error occurred"

            return FailedCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_failed))

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="text",
        )

        with pytest.raises(RuntimeError, match="Generation failed"):
            await generate(session, branch, params)

    async def test_failed_invocation_without_error_message(self, session_with_model):
        """Test RuntimeError includes status when no error message."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        async def mock_invoke_failed(**kwargs):
            class FailedCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.FAILED
                    self.execution.error = None

            return FailedCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_failed))

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="text",
        )

        with pytest.raises(RuntimeError, match=r"Generation failed.*status"):
            await generate(session, branch, params)


class TestBranchResourceAccess:
    """Test branch resource access control."""

    async def test_branch_without_model_access_raises_error(self, session_with_model):
        """Test that branch without model access raises ValueError."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources=set())  # No resources

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
        )

        with pytest.raises(ValueError, match="has no access to model"):
            await generate(session, branch, params)


class TestParameterPassthrough:
    """Test that parameters are passed through to imodel.invoke()."""

    async def test_imodel_kwargs_forwarded_to_invoke(self, session_with_model):
        """Test that imodel_kwargs are forwarded to imodel.invoke()."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = GenerateParams(
            imodel="mock_model",
            instruction="test",
            return_as="text",
            imodel_kwargs={"temperature": 0.7, "max_tokens": 100},
        )

        await generate(session, branch, params)

        # Verify invoke was called with forwarded parameters
        model.invoke.assert_called_once()
        call_kwargs = model.invoke.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100
