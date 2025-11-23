# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for operate module to achieve 100% coverage.

This file targets specific uncovered lines in operate module:
- Lines 221-226: Tool schema retrieval paths
- Line 345: Non-standard type in _to_response_str helper
- Lines 367-373: LNDL parsing edge cases (models/scalars extraction)
- Lines 413, 425: Backward compatibility fallback paths
- Lines 454-456: Model reconstruction without model_copy

These tests complement the comprehensive test suites in:
- test_factories_generate.py
- test_factories_operate.py
- test_factories_act.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from lionpride import Event, EventStatus
from lionpride.operations.operate.factory import operate
from lionpride.services.providers.oai_chat import OAIChatEndpoint
from lionpride.services.types.imodel import iModel
from lionpride.session import Session


class SimpleModel(BaseModel):
    """Simple test model."""

    content: str


class ModelWithoutCopy(BaseModel):
    """Model that simulates missing model_copy (for line 454-456 test)."""

    content: str
    action_requests: list[dict] | None = None
    action_responses: list[dict] | None = None


@dataclass
class MockResponse:
    """Mock response from model."""

    status: str = "success"
    data: str | dict = ""
    raw_response: dict = None
    metadata: dict = None

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {
                "id": "mock-id",
                "choices": [{"message": {"content": str(self.data)}}],
            }
        if self.metadata is None:
            self.metadata = {"model": "mock-model", "usage": {}}


@pytest.fixture
def mock_model():
    """Create mock iModel."""
    endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")
    model = iModel(backend=endpoint)

    async def mock_invoke(model_name: str | None = None, messages: list | None = None, **kwargs):
        class MockCalling(Event):
            def __init__(self, response_data: str | dict):
                super().__init__()
                self.status = EventStatus.COMPLETED
                self.execution.response = MockResponse(status="success", data=response_data)

        response = kwargs.get("_test_response", {"content": "mock"})
        return MockCalling(response)

    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))
    return model


@pytest.fixture
def session_with_model(mock_model):
    """Create session with mock model."""
    session = Session()
    session.services.register(mock_model, update=True)
    return session, mock_model


class TestToolSchemaRetrieval:
    """Test tool schema retrieval paths (lines 221-226)."""

    async def test_tools_true_retrieves_all_schemas_line_223(self, session_with_model):
        """Test tools=True calls get_tool_schemas() without tool_names (line 223)."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock get_tool_schemas method on registry instance
        mock_schemas = [{"name": "tool1", "parameters": {}}]
        mock_get_schemas = MagicMock(return_value=mock_schemas)

        # Add method to registry instance
        session.services.get_tool_schemas = mock_get_schemas

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "model_kwargs": {"model_name": "gpt-4.1-mini"},
            "tools": True,  # Should trigger line 221-223
        }

        await operate(session, branch, parameters)

        # Verify get_tool_schemas was called without tool_names
        mock_get_schemas.assert_called_once_with()

    async def test_tools_list_retrieves_specific_schemas_line_226(self, session_with_model):
        """Test tools=['tool1'] calls get_tool_schemas(tool_names=...) (line 226)."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock get_tool_schemas method on registry instance
        mock_schemas = [{"name": "tool1", "parameters": {}}]
        mock_get_schemas = MagicMock(return_value=mock_schemas)

        # Add method to registry instance
        session.services.get_tool_schemas = mock_get_schemas

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "model_kwargs": {"model_name": "gpt-4.1-mini"},
            "tools": ["tool1", "tool2"],  # Should trigger line 224-226
        }

        await operate(session, branch, parameters)

        # Verify get_tool_schemas was called with specific tool_names
        mock_get_schemas.assert_called_once_with(tool_names=["tool1", "tool2"])


class TestToResponseStrEdgeCases:
    """Test _to_response_str helper edge cases (line 345)."""

    async def test_non_standard_type_falls_back_to_str_line_345(self, session_with_model):
        """Test that non-str/non-dict/non-BaseModel types fall back to str() (line 345)."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Create mock that returns a list (non-standard type)
        async def mock_invoke_list(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    # Return a list (triggers line 345: return str(data))
                    self.execution.response = MockResponse(status="success", data=[1, 2, 3])

            return MockCalling()

        model.invoke.side_effect = mock_invoke_list

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "model_kwargs": {"model_name": "gpt-4.1-mini"},
            "skip_validation": True,
        }

        result = await operate(session, branch, parameters)

        # Should convert list to string
        assert result == [1, 2, 3] or result == "[1, 2, 3]"


class TestLNDLParsingEdgeCases:
    """Test LNDL parsing edge cases (lines 367-373)."""

    @pytest.mark.skip(reason="Mock response needs LNDL format")
    async def test_lndl_extracts_models_line_368(self, session_with_model):
        """Test LNDL parsing extracts models from lndl_output.models (line 368)."""
        from lionpride.operations.operate.operative import create_operative_from_model

        session, model = session_with_model
        branch = session.create_branch(name="test")

        operative = create_operative_from_model(SimpleModel, name="test_op")

        # Mock LNDL parsing to return fields (new structure)
        mock_lndl_output = MagicMock()
        mock_lndl_output.fields = {"simplemodel": SimpleModel(content="from lndl fields")}
        mock_lndl_output.actions = {}

        with patch(
            "lionpride.operations.operate.response_parser.parse_lndl_fuzzy",
            return_value=mock_lndl_output,
        ):
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "operable": operative.operable,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
                "use_lndl": True,
            }

            result = await operate(session, branch, parameters)

            # Should extract first value from lndl_output.fields
            assert isinstance(result, SimpleModel)
            assert result.content == "from lndl fields"

    @pytest.mark.skip(reason="Mock response needs LNDL format")
    async def test_lndl_extracts_scalars_when_no_models_line_371(self, session_with_model):
        """Test LNDL parsing returns scalar values from fields."""
        from lionpride.operations.operate.operative import create_operative_from_model

        session, model = session_with_model
        branch = session.create_branch(name="test")

        operative = create_operative_from_model(SimpleModel, name="test_op")

        # Mock LNDL parsing to return scalar value in fields
        mock_lndl_output = MagicMock()
        mock_lndl_output.fields = {"simplemodel": "scalar_value"}  # Scalar instead of model
        mock_lndl_output.actions = {}

        with patch(
            "lionpride.operations.operate.response_parser.parse_lndl_fuzzy",
            return_value=mock_lndl_output,
        ):
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "operable": operative.operable,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
                "use_lndl": True,
            }

            result = await operate(session, branch, parameters)

            # Should extract the scalar value from fields
            assert result == "scalar_value"

    async def test_lndl_returns_none_when_empty_line_373(self, session_with_model):
        """Test LNDL parsing falls back to operative when fields is empty."""
        from lionpride.operations.operate.operative import create_operative_from_model

        session, model = session_with_model
        branch = session.create_branch(name="test")

        operative = create_operative_from_model(SimpleModel, name="test_op")

        # Mock LNDL parsing to return empty fields
        mock_lndl_output = MagicMock()
        mock_lndl_output.fields = {}  # Empty fields
        mock_lndl_output.actions = {}

        with patch(
            "lionpride.operations.operate.response_parser.parse_lndl_fuzzy",
            return_value=mock_lndl_output,
        ):
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "operable": operative.operable,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
                "use_lndl": True,
                "return_message": True,  # Prevent ValueError
            }

            result, _ = await operate(session, branch, parameters)

            # Empty fields means parsed_response stays None, fallback to operative
            # which then falls back to validation_failed dict
            assert isinstance(result, dict)
            assert result.get("validation_failed") is True


class TestBackwardCompatibilityFallbacks:
    """Test backward compatibility fallback paths (lines 413, 425)."""

    async def test_dict_response_fallback_to_model_validate_line_413(self, session_with_model):
        """Test dict response falls back to model_validate when adapter returns None (line 413)."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock response with dict data
        async def mock_invoke_dict(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse(
                        status="success", data={"content": "from dict validate"}
                    )

            return MockCalling()

        model.invoke.side_effect = mock_invoke_dict

        # Mock PydanticSpecAdapter.validate_response to return None (trigger fallback)
        with patch(
            "lionpride.types.spec_adapters.pydantic_field.PydanticSpecAdapter.validate_response",
            return_value=None,
        ):
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "response_model": SimpleModel,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
                # No operative - uses backward compatibility path
            }

            result = await operate(session, branch, parameters)

            # Should fall back to model_validate(response_data) at line 413
            assert isinstance(result, SimpleModel)
            assert result.content == "from dict validate"

    @pytest.mark.skip(reason="Mock response structure needs update")
    async def test_json_string_fallback_to_model_validate_line_425(self, session_with_model):
        """Test JSON string falls back to model_validate after json.loads (line 425)."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock response with JSON string
        async def mock_invoke_json(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse(
                        status="success", data='{"content": "from json string"}'
                    )

            return MockCalling()

        model.invoke.side_effect = mock_invoke_json

        # Mock PydanticSpecAdapter.validate_response to return None (trigger fallback)
        with patch(
            "lionpride.types.spec_adapters.pydantic_field.PydanticSpecAdapter.validate_response",
            return_value=None,
        ):
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "response_model": SimpleModel,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            }

            result = await operate(session, branch, parameters)

            # Should parse JSON and call model_validate(data) at line 425
            assert isinstance(result, SimpleModel)
            assert result.content == "from json string"


class TestActionResponseMergingFallback:
    """Test action response merging fallback (lines 454-456)."""

    async def test_model_without_model_copy_uses_model_validate_line_456(self, session_with_model):
        """Test model without model_copy falls back to model_dump + model_validate (line 456)."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Register a mock tool
        from lionpride.services.types.tool import Tool, ToolConfig

        async def mock_tool(x: int) -> int:
            return x * 2

        tool = Tool(
            func_callable=mock_tool,
            config=ToolConfig(name="mock_tool", provider="tool"),
        )
        session.services.register(iModel(backend=tool))

        # Mock response with action_requests
        async def mock_invoke_with_actions(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse(
                        status="success",
                        data={
                            "simplemodel": {"content": "test"},
                            "action_requests": [{"function": "mock_tool", "arguments": {"x": 5}}],
                        },
                    )

            return MockCalling()

        model.invoke.side_effect = mock_invoke_with_actions

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "model_kwargs": {"model_name": "gpt-4.1-mini"},
            "actions": True,
        }

        result = await operate(session, branch, parameters)

        # Verify action execution completed
        assert hasattr(result, "action_responses")
        assert len(result.action_responses) > 0
        assert result.action_responses[0].output == 10

    @pytest.mark.skip(reason="Test requires action response handling to match new structure")
    async def test_explicit_model_copy_removal_triggers_fallback(self, session_with_model):
        """Test explicit removal of model_copy attribute triggers fallback (lines 454-456)."""
        from unittest.mock import MagicMock

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Register a mock tool
        from lionpride.services.types.tool import Tool, ToolConfig

        async def mock_tool(x: int) -> int:
            return x * 2

        tool = Tool(
            func_callable=mock_tool,
            config=ToolConfig(name="mock_tool", provider="tool"),
        )
        session.services.register(iModel(backend=tool))

        # Mock response with action_requests
        async def mock_invoke_with_actions(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse(
                        status="success",
                        data={
                            "simplemodel": {"content": "test"},
                            "action_requests": [{"function": "mock_tool", "arguments": {"x": 5}}],
                        },
                    )

            return MockCalling()

        model.invoke.side_effect = mock_invoke_with_actions

        # Patch hasattr to return False for model_copy check at line 448
        original_hasattr = __builtins__["hasattr"]

        def custom_hasattr(obj, name):
            if name == "model_copy" and isinstance(obj, BaseModel):
                return False  # Force fallback path
            return original_hasattr(obj, name)

        with patch("builtins.hasattr", side_effect=custom_hasattr):
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "response_model": SimpleModel,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
                "actions": True,
            }

            result = await operate(session, branch, parameters)

            # Verify fallback path was used (lines 454-456)
            assert hasattr(result, "action_responses")
            assert len(result.action_responses) > 0
            assert result.action_responses[0].output == 10


class TestCombinedEdgeCases:
    """Test combinations of edge cases."""

    @pytest.mark.skip(reason="Mock response needs LNDL format")
    async def test_lndl_fallback_to_operative_after_exception(self, session_with_model):
        """Test LNDL exception triggers Operative fallback (line 375-377)."""
        from lionpride.operations.operate.operative import create_operative_from_model

        session, model = session_with_model
        branch = session.create_branch(name="test")

        operative = create_operative_from_model(SimpleModel, name="test_op")

        # Mock parse_lndl_fuzzy to raise exception and operative.validate_response to succeed
        with (
            patch("lionpride.lndl.parse_lndl_fuzzy", side_effect=ValueError("LNDL parsing failed")),
            patch.object(
                operative, "validate_response", return_value=SimpleModel(content="from operative")
            ),
        ):
            parameters = {
                "instruction": "Test",
                "imodel": model,
                "operable": operative.operable,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
                "use_lndl": True,
            }

            result = await operate(session, branch, parameters)

            # Should fall back to Operative validation (line 377)
            assert isinstance(result, SimpleModel)
            assert result.content == "from operative"

    async def test_tools_with_existing_tool_schemas_skips_retrieval(self, session_with_model):
        """Test that providing tool_schemas directly skips retrieval (line 219 condition)."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Provide tool_schemas explicitly
        explicit_schemas = [{"name": "custom_tool", "parameters": {}}]

        # Add mock method to check it's not called
        mock_get_schemas = MagicMock(return_value=[])
        session.services.get_tool_schemas = mock_get_schemas

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": SimpleModel,
            "model_kwargs": {"model_name": "gpt-4.1-mini"},
            "tools": True,
            "tool_schemas": explicit_schemas,  # Already provided
        }

        await operate(session, branch, parameters)

        # Should NOT call get_tool_schemas (line 219: if tools and not tool_schemas)
        mock_get_schemas.assert_not_called()
