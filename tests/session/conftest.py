# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for session tests."""

from uuid import uuid4

import pytest

from lionpride.session.messages.content import (
    ActionRequestContent,
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    SystemContent,
)


@pytest.fixture
def system_content():
    """System message content."""
    return SystemContent.create(system_message="You are a helpful assistant")


@pytest.fixture
def instruction_content():
    """User instruction content."""
    return InstructionContent.create(instruction="What is the capital of France?")


@pytest.fixture
def assistant_content():
    """Assistant response content."""
    return AssistantResponseContent.create(assistant_response="The capital is Paris.")


@pytest.fixture
def action_request_content():
    """Action request content."""
    return ActionRequestContent.create(function="get_weather", arguments={"city": "NYC"})


@pytest.fixture
def action_response_content():
    """Action response content (success)."""
    return ActionResponseContent.create(request_id="req_123", result={"temp": 72})


@pytest.fixture
def action_error_content():
    """Action response content (error)."""
    return ActionResponseContent.create(request_id="req_456", error="API timeout")


# =============================================================================
# UUID Fixtures
# =============================================================================


@pytest.fixture
def sample_uuid():
    """Sample UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_uuid_str(sample_uuid):
    """String representation of sample UUID."""
    return str(sample_uuid)


@pytest.fixture
def non_uuid_string():
    """Regular string that's not a UUID."""
    return "custom_sender_name"
