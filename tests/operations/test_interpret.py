# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for interpret operation (lines 43-83 coverage)."""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from lionpride import Event, EventStatus
from lionpride.operations.operate.interpret import interpret
from lionpride.operations.operate.types import InterpretParams
from lionpride.session import Session


@dataclass
class MockNormalizedResponse:
    """Mock NormalizedResponse for testing."""

    data: str = "refined instruction text"
    raw_response: dict = None
    metadata: dict = None

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {"id": "mock-id"}
        if self.metadata is None:
            self.metadata = {}


class TestInterpretValidation:
    """Test parameter validation (lines 43-55)."""

    @pytest.mark.asyncio
    async def test_missing_text_raises_valueerror(self, session_with_model):
        """Line 43-44: Missing text parameter raises ValueError."""
        session, model = session_with_model
        branch = session.create_branch(resources={model.name})

        params = InterpretParams(
            imodel=model.name,
            # text is sentinel (missing)
        )

        with pytest.raises(ValueError, match="interpret requires 'text' parameter"):
            await interpret(session, branch, params)

    @pytest.mark.asyncio
    async def test_missing_imodel_raises_valueerror(self, session_with_model):
        """Line 46-47: Missing imodel parameter raises ValueError."""
        session, model = session_with_model
        branch = session.create_branch(resources={model.name})

        params = InterpretParams(
            text="some user input",
            # imodel is sentinel (missing)
        )

        with pytest.raises(ValueError, match="interpret requires 'imodel' parameter"):
            await interpret(session, branch, params)

    @pytest.mark.asyncio
    async def test_model_not_in_branch_resources_raises_permissionerror(self, session_with_model):
        """Line 50-55: Model not in branch.resources raises PermissionError."""
        session, model = session_with_model
        # Branch without any resources
        branch = session.create_branch(resources=set())

        params = InterpretParams(
            text="some user input",
            imodel=model.name,
        )

        with pytest.raises(PermissionError, match="cannot access model"):
            await interpret(session, branch, params)

    @pytest.mark.asyncio
    async def test_imodel_object_not_in_resources_raises_permissionerror(self, session_with_model):
        """Line 50: Test with iModel object (not string) - extracts name."""
        session, model = session_with_model
        # Branch without any resources
        branch = session.create_branch(resources=set())

        params = InterpretParams(
            text="some user input",
            imodel=model,  # Pass iModel object, not string
        )

        with pytest.raises(PermissionError, match="cannot access model"):
            await interpret(session, branch, params)


class TestInterpretExecution:
    """Test interpret execution (lines 57-83)."""

    @pytest.mark.asyncio
    async def test_interpret_basic_success(self, session_with_model):
        """Lines 57-83: Basic successful interpretation."""
        session, model = session_with_model
        branch = session.create_branch(resources={model.name})

        # Mock to return refined text
        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(
                        data="  Refined: Clear instruction  "
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="fix bug",
            imodel=model.name,
            domain="programming",
            style="detailed",
        )

        result = await interpret(session, branch, params)

        # Verify result is stripped
        assert result == "Refined: Clear instruction"

        # Verify model was called
        model.invoke.assert_called_once()

        # Verify instruction content includes domain and style
        call_kwargs = model.invoke.call_args.kwargs
        messages = call_kwargs["messages"]
        # Find user message content
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "programming" in user_msg["content"]
        assert "detailed" in user_msg["content"]
        assert "fix bug" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_interpret_with_sample_writing(self, session_with_model):
        """Line 66-67: Test sample_writing parameter inclusion."""
        session, model = session_with_model
        branch = session.create_branch(resources={model.name})

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="refined")

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="do something",
            imodel=model.name,
            domain="general",
            style="concise",
            sample_writing="Example: Be brief and direct.",
        )

        result = await interpret(session, branch, params)

        assert result == "refined"

        # Verify sample_writing is included in instruction
        call_kwargs = model.invoke.call_args.kwargs
        messages = call_kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Sample writing style:" in user_msg["content"]
        assert "Example: Be brief and direct." in user_msg["content"]

    @pytest.mark.asyncio
    async def test_interpret_uses_temperature_param(self, session_with_model):
        """Line 79: Verify temperature parameter is passed to imodel_kwargs."""
        session, model = session_with_model
        branch = session.create_branch(resources={model.name})

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="result")

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="input",
            imodel=model.name,
            temperature=0.7,
        )

        await interpret(session, branch, params)

        # Verify temperature passed to model
        call_kwargs = model.invoke.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_interpret_with_imodel_object(self, session_with_model):
        """Line 50: Test passing iModel object extracts name correctly."""
        session, model = session_with_model
        branch = session.create_branch(resources={model.name})

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="result")

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="input",
            imodel=model,  # Pass iModel object directly
        )

        result = await interpret(session, branch, params)
        assert result == "result"
        model.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_interpret_system_instruction_format(self, session_with_model):
        """Lines 58-63: Verify system instruction format."""
        session, model = session_with_model
        branch = session.create_branch(resources={model.name})

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="output")

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="raw input",
            imodel=model.name,
        )

        await interpret(session, branch, params)

        # Verify system instruction content
        call_kwargs = model.invoke.call_args.kwargs
        messages = call_kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")

        # Check key phrases from system instruction
        assert "rewrite" in user_msg["content"].lower()
        assert "clearer" in user_msg["content"].lower()
        assert "structured" in user_msg["content"].lower()
        assert "raw input" in user_msg["content"]
