# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for operations tests.

This module provides reusable mock models and session fixtures for testing
operations without making actual API calls.

Fixtures:
    mock_model: A mock iModel that returns mock_normalized_response on invoke.
    session_with_model: A Session with a registered mock model.

Functions:
    mock_normalized_response: Factory for test NormalizedResponse objects.
"""

from unittest.mock import AsyncMock

import pytest

from lionpride import Event, EventStatus
from lionpride.services.types import NormalizedResponse
from lionpride.session import Session


def mock_normalized_response(data: str = "mock response text") -> NormalizedResponse:
    """Create a real NormalizedResponse for testing."""
    return NormalizedResponse(
        data=data,
        status="completed",
        raw_response={"id": "mock-id", "choices": [{"message": {"content": data}}]},
        metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
    )


@pytest.fixture
def mock_model():
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
                self.execution.response = mock_normalized_response()

        return MockCalling()

    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))
    return model


@pytest.fixture
def session_with_model(mock_model):
    """Create session with registered mock model."""
    session = Session()
    session.services.register(mock_model, update=True)
    return session, mock_model
