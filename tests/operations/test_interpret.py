# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for interpret.py - instruction refinement operation."""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from lionpride import Event, EventStatus
from lionpride.operations.operate.interpret import interpret
from lionpride.operations.operate.types import InterpretParams


class TestInterpret:
    """Tests for interpret() function."""

    async def test_interpret_missing_text_raises(self, session_with_model):
        """Test missing text parameter raises ValueError."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        params = InterpretParams(
            imodel=model,
            # text is missing
        )

        with pytest.raises(ValueError, match="interpret requires 'text' parameter"):
            await interpret(session, branch, params)

    async def test_interpret_missing_imodel_raises(self, session_with_model):
        """Test missing imodel parameter raises ValueError."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        params = InterpretParams(
            text="Some instruction to refine",
            # imodel is missing
        )

        with pytest.raises(ValueError, match="interpret requires 'imodel' parameter"):
            await interpret(session, branch, params)

    async def test_interpret_resource_access_denied(self, session_with_model):
        """Test branch without model access raises PermissionError."""
        session, model = session_with_model
        # Branch without access to model
        branch = session.create_branch(name="restricted", resources=set())

        params = InterpretParams(
            text="Refine this instruction",
            imodel=model,
        )

        with pytest.raises(PermissionError, match="cannot access model"):
            await interpret(session, branch, params)

    async def test_interpret_basic_flow(self, session_with_model):
        """Test basic interpret flow with mocked LLM."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        @dataclass
        class MockRefinedResponse:
            data: str = "Refined: Please analyze the sales data and provide insights."
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockRefinedResponse()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="look at the sales data",
            imodel=model,
        )

        result = await interpret(session, branch, params)

        assert isinstance(result, str)
        assert "Refined" in result
        model.invoke.assert_called_once()

    async def test_interpret_dict_params_conversion(self, session_with_model):
        """Test dict params are converted to InterpretParams."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        @dataclass
        class MockResponse:
            data: str = "Refined instruction"
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        # Pass dict instead of InterpretParams
        params = {
            "text": "raw user input",
            "imodel": model,
        }

        result = await interpret(session, branch, params)
        assert isinstance(result, str)

    async def test_interpret_with_string_imodel_name(self, session_with_model):
        """Test imodel as string name."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        @dataclass
        class MockResponse:
            data: str = "Refined instruction"
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="analyze this",
            imodel="mock_model",  # String name instead of object
        )

        result = await interpret(session, branch, params)
        assert isinstance(result, str)

    async def test_interpret_custom_domain(self, session_with_model):
        """Test interpret with custom domain hint."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        captured_kwargs = {}

        @dataclass
        class MockResponse:
            data: str = "Domain-specific refined instruction"
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke(**kwargs):
            captured_kwargs.update(kwargs)

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="check the code",
            imodel=model,
            domain="software engineering",
            style="detailed",
        )

        result = await interpret(session, branch, params)

        assert isinstance(result, str)
        # Verify domain was included in prompt
        messages = captured_kwargs.get("messages", [])
        assert any("software engineering" in str(m) for m in messages)

    async def test_interpret_with_sample_writing(self, session_with_model):
        """Test interpret with sample writing style."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        captured_kwargs = {}

        @dataclass
        class MockResponse:
            data: str = "Styled refined instruction"
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke(**kwargs):
            captured_kwargs.update(kwargs)

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="write something",
            imodel=model,
            sample_writing="Please ensure all instructions are clear and actionable.",
        )

        result = await interpret(session, branch, params)

        assert isinstance(result, str)
        # Verify sample writing was included
        messages = captured_kwargs.get("messages", [])
        assert any("clear and actionable" in str(m) for m in messages)

    async def test_interpret_custom_temperature(self, session_with_model):
        """Test interpret with custom temperature."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        captured_kwargs = {}

        @dataclass
        class MockResponse:
            data: str = "Temperature-adjusted response"
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke(**kwargs):
            captured_kwargs.update(kwargs)

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="creative writing prompt",
            imodel=model,
            temperature=0.8,  # Higher temperature for creativity
        )

        result = await interpret(session, branch, params)

        assert isinstance(result, str)
        # Temperature should be passed through
        assert captured_kwargs.get("temperature") == 0.8

    async def test_interpret_returns_stripped_string(self, session_with_model):
        """Test interpret strips whitespace from result."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        @dataclass
        class MockResponseWithWhitespace:
            data: str = "  \n  Refined instruction with whitespace  \n  "
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponseWithWhitespace()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

        params = InterpretParams(
            text="some input",
            imodel=model,
        )

        result = await interpret(session, branch, params)

        # Result should be stripped
        assert not result.startswith(" ")
        assert not result.endswith(" ")
        assert not result.startswith("\n")
        assert not result.endswith("\n")

    async def test_interpret_empty_text_raises(self, session_with_model):
        """Test empty text raises ValueError."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        params = InterpretParams(
            text="",  # Empty string
            imodel=model,
        )

        with pytest.raises(ValueError, match="interpret requires 'text' parameter"):
            await interpret(session, branch, params)
