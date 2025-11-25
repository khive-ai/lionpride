# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for communicate.py coverage.

Targets coverage for lines:
- 41, 45, 54, 73, 77: Parameter validation
- 152-175: Retry instruction construction
- 186-188, 198-200: JSON validation
- 215, 218, 220, 222, 224: Result formatting
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from lionpride import Event, EventStatus
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


class TestCommunicateCoverage:
    """Test communicate.py uncovered lines."""

    async def test_communicate_missing_instruction(self, session_with_model):
        """Test line 41: Missing instruction raises ValueError."""
        from lionpride.operations.operate.communicate import communicate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"imodel": model}

        with pytest.raises(ValueError, match="communicate requires 'instruction' parameter"):
            await communicate(session, branch, parameters)

    async def test_communicate_missing_imodel(self, session_with_model):
        """Test line 45: Missing imodel raises ValueError."""
        from lionpride.operations.operate.communicate import communicate

        session, _ = session_with_model
        branch = session.create_branch(name="test")

        parameters = {"instruction": "Test"}

        with pytest.raises(ValueError, match="communicate requires 'imodel' parameter"):
            await communicate(session, branch, parameters)

    async def test_communicate_imodel_without_name_attr(self, session_with_model):
        """Test line 54: imodel object without name attribute raises ValueError."""
        from lionpride.operations.operate.communicate import communicate

        session, _ = session_with_model
        branch = session.create_branch(name="test")

        # Mock imodel without name attribute
        mock_imodel = MagicMock(spec=[])  # No attributes

        parameters = {"instruction": "Test", "imodel": mock_imodel}

        with pytest.raises(
            ValueError, match="imodel must be a string name or have a 'name' attribute"
        ):
            await communicate(session, branch, parameters)

    async def test_communicate_return_as_model_without_response_model(self, session_with_model):
        """Test line 73: return_as='model' without response_model raises ValueError."""
        from lionpride.operations.operate.communicate import communicate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "return_as": "model",
            # No response_model
        }

        with pytest.raises(
            ValueError, match="return_as='model' requires 'response_model' parameter"
        ):
            await communicate(session, branch, parameters)

    async def test_communicate_branch_string_resolution(self, session_with_model):
        """Test line 77: Branch string resolution."""
        from lionpride.operations.operate.communicate import communicate

        session, model = session_with_model
        session.create_branch(name="test_branch")

        parameters = {"instruction": "Test", "imodel": model}

        # Pass branch as string
        result = await communicate(session, "test_branch", parameters)

        assert isinstance(result, str)

    async def test_communicate_return_as_raw(self, session_with_model):
        """Test line 218: return_as='raw' returns raw response."""
        from lionpride.operations.operate.communicate import communicate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "return_as": "raw",
        }

        result = await communicate(session, branch, parameters)

        assert isinstance(result, dict)
        assert "id" in result or "choices" in result

    async def test_communicate_return_as_message(self, session_with_model):
        """Test line 220: return_as='message' returns Message."""
        from lionpride.operations.operate.communicate import communicate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "return_as": "message",
        }

        result = await communicate(session, branch, parameters)

        assert isinstance(result, Message)

    async def test_communicate_invalid_return_as(self, session_with_model):
        """Test line 224: Invalid return_as raises ValueError."""
        from lionpride.operations.operate.communicate import communicate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Mock to bypass normal flow
        with patch("lionpride.operations.operate.communicate._format_result") as mock_format:
            mock_format.side_effect = ValueError("Unsupported return_as: invalid")

            parameters = {
                "instruction": "Test",
                "imodel": model,
                "return_as": "invalid",
            }

            with pytest.raises(ValueError, match="Unsupported return_as"):
                await communicate(session, branch, parameters)

    async def test_format_result_text_with_basemodel(self):
        """Test line 215: Text format with BaseModel returns model_dump_json."""
        from lionpride.operations.operate.communicate import _format_result

        class TestModel(BaseModel):
            value: str

        validated = TestModel(value="test")
        result = _format_result(
            return_as="text",
            validated=validated,
            response_text="raw text",
            raw_response={},
            assistant_msg=MagicMock(),
            response_model=TestModel,
        )

        assert isinstance(result, str)
        assert "test" in result

    async def test_format_result_model(self):
        """Test line 222: Model format returns validated model."""
        from lionpride.operations.operate.communicate import _format_result

        class TestModel(BaseModel):
            value: str

        validated = TestModel(value="test")
        result = _format_result(
            return_as="model",
            validated=validated,
            response_text="raw text",
            raw_response={},
            assistant_msg=MagicMock(),
            response_model=TestModel,
        )

        assert isinstance(result, TestModel)
        assert result.value == "test"

    async def test_validate_json_with_dict(self):
        """Test lines 186-188: _validate_json with dict input."""
        from lionpride.operations.operate.communicate import _validate_json

        class TestModel(BaseModel):
            title: str

        validated, error = _validate_json(
            response_data={"title": "Test"},
            response_model=TestModel,
            strict=False,
            fuzzy_parse=True,
        )

        assert error is None
        assert isinstance(validated, TestModel)
        assert validated.title == "Test"

    async def test_validate_json_returns_none(self):
        """Test lines 198: _validate_json returns None validation error."""
        from lionpride.operations.operate.communicate import _validate_json

        class TestModel(BaseModel):
            title: str

        # Pass invalid string that won't validate
        validated, error = _validate_json(
            response_data="not json",
            response_model=TestModel,
            strict=False,
            fuzzy_parse=True,
        )

        # Should return None and error message
        assert validated is None
        assert error is not None

    async def test_validate_json_exception(self):
        """Test lines 199-200: _validate_json exception handling."""
        from lionpride.operations.operate.communicate import _validate_json

        class TestModel(BaseModel):
            title: str

        # This should cause validation error
        validated, error = _validate_json(
            response_data='{"invalid": "json"}',
            response_model=TestModel,
            strict=True,
            fuzzy_parse=False,
        )

        assert validated is None
        assert error is not None

    async def test_communicate_with_string_imodel_name(self, session_with_model):
        """Test lines 49-50: imodel as string name resolution."""
        from lionpride.operations.operate.communicate import communicate

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        # Pass imodel as string name (lines 49-50)
        parameters = {
            "instruction": "Test",
            "imodel": "mock_model",  # String name, not object
        }

        result = await communicate(session, branch, parameters)

        assert isinstance(result, str)

    async def test_communicate_retry_on_validation_failure(self, session_with_model):
        """Test lines 157-167: Retry instruction construction."""
        from lionpride.operations.operate.communicate import communicate

        session, model = session_with_model
        branch = session.create_branch(name="test")

        class StrictModel(BaseModel):
            required_field: str

        call_count = 0

        async def mock_invoke_with_retry(**kwargs):
            nonlocal call_count
            call_count += 1

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    if call_count == 1:
                        # First call: invalid response
                        self.execution.response = MockNormalizedResponse(data="invalid json")
                    else:
                        # Second call: valid response
                        self.execution.response = MockNormalizedResponse(
                            data='{"required_field": "valid"}'
                        )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_with_retry))

        parameters = {
            "instruction": "Test",
            "imodel": model,
            "response_model": StrictModel,
            "return_as": "model",  # Return as model to get the StrictModel instance
            "max_retries": 1,  # Allow 1 retry to hit lines 157-167
            "model_kwargs": {"model_name": "gpt-4"},
        }

        result = await communicate(session, branch, parameters)

        # Should succeed on retry
        assert result.required_field == "valid"
        assert call_count == 2

    async def test_format_result_invalid_return_as(self):
        """Test line 224: Invalid return_as in _format_result."""
        from lionpride.operations.operate.communicate import _format_result

        with pytest.raises(ValueError, match="Unsupported return_as"):
            _format_result(
                return_as="invalid_type",
                validated=None,
                response_text="text",
                raw_response={},
                assistant_msg=MagicMock(),
                response_model=None,
            )
