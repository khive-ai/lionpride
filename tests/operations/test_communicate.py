# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for communicate operation.

Tests:
- Text path (no operable): Generate → persist → return text
- IPU path (with operable): Generate → parse → validate → persist → return model
- Capability enforcement
- Message persistence

Note: Uses mock_normalized_response, mock_model, and session_with_model fixtures from conftest.py.
"""

from unittest.mock import AsyncMock

import pytest

from lionpride import Event, EventStatus
from lionpride.errors import AccessError, ValidationError
from lionpride.operations.operate.communicate import communicate
from lionpride.operations.operate.types import (
    CommunicateParams,
    GenerateParams,
    ParseParams,
)
from lionpride.session import Session
from lionpride.types import Operable, Spec
from tests.operations.conftest import mock_normalized_response


class TestCommunicateValidation:
    """Test parameter validation."""

    async def test_missing_generate_params_raises_error(self, session_with_model):
        """Test that missing generate params raises ValidationError."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = CommunicateParams()  # No generate params

        with pytest.raises(ValidationError, match="communicate requires 'generate' params"):
            await communicate(session, branch, params)


class TestTextPath:
    """Test text path (no operable) - simple stateful chat."""

    async def test_text_path_returns_string(self, session_with_model):
        """Test text path returns response string."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        params = CommunicateParams(
            generate=GenerateParams(
                imodel="mock_model",
                instruction="Test instruction",
            ),
        )

        result = await communicate(session, branch, params)

        assert isinstance(result, str)
        assert result == "mock response text"

    async def test_text_path_persists_messages(self, session_with_model):
        """Test that text path adds messages to branch."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={"mock_model"})

        initial_count = len(session.messages[branch])

        params = CommunicateParams(
            generate=GenerateParams(
                imodel="mock_model",
                instruction="Test instruction",
            ),
        )

        await communicate(session, branch, params)

        # Should have added 2 messages (instruction + response)
        final_count = len(session.messages[branch])
        assert final_count == initial_count + 2

    async def test_branch_as_string(self, session_with_model):
        """Test that branch can be passed as string name."""
        session, _ = session_with_model
        branch_name = "test_branch"
        session.create_branch(name=branch_name, resources={"mock_model"})

        params = CommunicateParams(
            generate=GenerateParams(
                imodel="mock_model",
                instruction="Test",
            ),
        )

        result = await communicate(session, branch_name, params)

        assert isinstance(result, str)
        assert result == "mock response text"


class TestIPUPath:
    """Test IPU path (with operable) - structured output with validation."""

    @pytest.fixture
    def simple_operable(self):
        """Create a simple operable for testing."""
        return Operable(
            specs=[
                Spec(str, name="name"),
                Spec(int, name="value"),
            ],
            name="TestOperable",
        )

    async def test_operable_requires_capabilities(self, session_with_model, simple_operable):
        """Test that operable path requires explicit capabilities at params construction."""
        # CommunicateParams validates capabilities at construction time
        # when operable is provided
        with pytest.raises(ValueError, match="capabilities must be explicitly declared"):
            CommunicateParams(
                generate=GenerateParams(
                    imodel="mock_model",
                    instruction="Return JSON",
                ),
                parse=ParseParams(),
                operable=simple_operable,
                # No capabilities set - raises ValueError at construction
            )

    async def test_capabilities_must_be_subset_of_branch(self, session_with_model, simple_operable):
        """Test that requested capabilities must be subset of branch capabilities."""
        session, _ = session_with_model
        # Branch with limited capabilities
        branch = session.create_branch(
            name="test",
            resources={"mock_model"},
            capabilities={"name"},  # Only 'name' allowed
        )

        params = CommunicateParams(
            generate=GenerateParams(
                imodel="mock_model",
                instruction="Return JSON",
            ),
            parse=ParseParams(),
            operable=simple_operable,
            capabilities={"name", "value"},  # Requesting more than allowed
        )

        with pytest.raises(AccessError, match="missing capabilities"):
            await communicate(session, branch, params)

    async def test_operable_path_with_valid_json(
        self, session_with_model, simple_operable, mock_model
    ):
        """Test IPU path with valid JSON response."""

        # Mock response with valid JSON
        async def mock_invoke_json(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = mock_normalized_response(
                        data='{"name": "test", "value": 42}'
                    )

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_invoke_json))

        session, _ = session_with_model
        branch = session.create_branch(
            name="test",
            resources={"mock_model"},
            capabilities={"name", "value"},
        )

        params = CommunicateParams(
            generate=GenerateParams(
                imodel="mock_model",
                instruction="Return JSON",
            ),
            parse=ParseParams(),
            operable=simple_operable,
            capabilities={"name", "value"},
        )

        result = await communicate(session, branch, params)

        # Result should be validated model instance
        assert hasattr(result, "name")
        assert hasattr(result, "value")
        assert result.name == "test"
        assert result.value == 42

    async def test_operable_path_persists_messages(
        self, session_with_model, simple_operable, mock_model
    ):
        """Test that IPU path adds messages to branch."""

        async def mock_invoke_json(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = mock_normalized_response(
                        data='{"name": "test", "value": 42}'
                    )

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_invoke_json))

        session, _ = session_with_model
        branch = session.create_branch(
            name="test",
            resources={"mock_model"},
            capabilities={"name", "value"},
        )

        initial_count = len(session.messages[branch])

        params = CommunicateParams(
            generate=GenerateParams(
                imodel="mock_model",
                instruction="Return JSON",
            ),
            parse=ParseParams(),
            operable=simple_operable,
            capabilities={"name", "value"},
        )

        await communicate(session, branch, params)

        # Should have added 2 messages
        final_count = len(session.messages[branch])
        assert final_count == initial_count + 2
