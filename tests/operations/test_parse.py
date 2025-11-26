# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for parse.py - JSON extraction with LLM fallback."""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from lionpride import Event, EventStatus
from lionpride.operations.operate.parse import _try_direct_extract, parse
from lionpride.operations.operate.types import ParseParams


class TestTryDirectExtract:
    """Tests for _try_direct_extract helper function."""

    def test_valid_json_string(self):
        """Test extraction from valid JSON string."""
        text = '{"name": "test", "value": 42}'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result == {"name": "test", "value": 42}

    def test_markdown_wrapped_json(self):
        """Test extraction from markdown code block."""
        text = """```json
{"name": "test", "value": 42}
```"""
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result == {"name": "test", "value": 42}

    def test_json_with_surrounding_text_returns_none(self):
        """Test JSON embedded in plain text returns None (not in code block)."""
        text = 'Here is the response: {"key": "value"} and more text'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        # extract_json only extracts from clean JSON or markdown code blocks
        assert result is None

    def test_malformed_json_returns_none(self):
        """Test malformed JSON returns None."""
        text = '{"name": "test", value: 42}'  # Missing quotes around value
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_empty_text_returns_none(self):
        """Test empty text returns None."""
        result = _try_direct_extract(
            text="",
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_non_json_text_returns_none(self):
        """Test plain text without JSON returns None."""
        result = _try_direct_extract(
            text="This is just plain text without any JSON",
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_json_array_extracts_first_dict(self):
        """Test JSON array returns first dict element."""
        text = '[{"first": 1}, {"second": 2}]'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result == {"first": 1}

    def test_empty_array_returns_none(self):
        """Test empty JSON array returns None."""
        text = "[]"
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_with_target_keys_fuzzy_match(self):
        """Test fuzzy key matching with target_keys."""
        text = '{"user_name": "Alice", "usr_age": 30}'
        result = _try_direct_extract(
            text=text,
            target_keys=["username", "userage"],
            similarity_threshold=0.7,
            handle_unmatched="force",
        )
        # fuzzy_validate_mapping should match similar keys
        assert result is not None


class TestParse:
    """Tests for parse() function."""

    async def test_parse_with_valid_json_no_llm(self, session_with_model):
        """Test parse with valid JSON doesn't call LLM."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        params = ParseParams(
            text='{"result": "success"}',
            imodel=model,
        )

        result = await parse(session, branch, params)

        assert result == {"result": "success"}
        # LLM should NOT have been called (direct extraction succeeded)
        model.invoke.assert_not_called()

    async def test_parse_empty_text_returns_none(self, session_with_model):
        """Test empty text returns None."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(text="")

        result = await parse(session, branch, params)
        assert result is None

    async def test_parse_none_text_returns_none(self, session_with_model):
        """Test None text returns None."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams()  # text is None by default

        result = await parse(session, branch, params)
        assert result is None

    async def test_parse_invalid_json_no_imodel_returns_none(self, session_with_model):
        """Test invalid JSON without imodel returns None."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(
            text="This is not JSON at all",
            imodel=None,  # No LLM fallback
        )

        result = await parse(session, branch, params)
        assert result is None

    async def test_parse_dict_params_conversion(self, session_with_model):
        """Test dict params are converted to ParseParams."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        # Pass dict instead of ParseParams
        params = {
            "text": '{"key": "value"}',
            "imodel": model,
        }

        result = await parse(session, branch, params)
        assert result == {"key": "value"}

    async def test_parse_resource_access_denied(self, session_with_model):
        """Test branch without model access raises PermissionError."""
        session, model = session_with_model
        # Branch without access to model
        branch = session.create_branch(name="restricted", resources=set())

        params = ParseParams(
            text="invalid json text",  # Will trigger LLM fallback
            imodel=model,
        )

        with pytest.raises(PermissionError, match="cannot access model"):
            await parse(session, branch, params)

    async def test_parse_with_string_imodel_name(self, session_with_model):
        """Test imodel as string name."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        params = ParseParams(
            text='{"data": 123}',
            imodel="mock_model",  # String name
        )

        result = await parse(session, branch, params)
        assert result == {"data": 123}

    async def test_parse_llm_fallback_on_invalid_json(self, session_with_model):
        """Test LLM fallback is triggered for invalid JSON."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        @dataclass
        class MockResponseWithJSON:
            data: str = '{"extracted": "data"}'
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        # Mock LLM to return valid JSON
        async def mock_invoke_json(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponseWithJSON()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_json))

        params = ParseParams(
            text="This is not valid JSON at all",
            imodel=model,
            max_retries=1,
        )

        result = await parse(session, branch, params)

        # LLM should have been called
        model.invoke.assert_called()
        assert result == {"extracted": "data"}

    async def test_parse_llm_fallback_failure_returns_none(self, session_with_model):
        """Test LLM fallback returning non-JSON returns None."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        @dataclass
        class MockResponseNoJSON:
            data: str = "Still not valid JSON"
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"choices": [{"message": {"content": self.data}}]}
                if self.metadata is None:
                    self.metadata = {}

        async def mock_invoke_no_json(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponseNoJSON()

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_no_json))

        params = ParseParams(
            text="Invalid JSON here",
            imodel=model,
            max_retries=2,
        )

        result = await parse(session, branch, params)
        assert result is None

    async def test_parse_with_target_keys(self, session_with_model):
        """Test parse with target_keys for fuzzy matching."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(
            text='{"username": "alice", "user_score": 100}',
            target_keys=["username", "userscore"],
            similarity_threshold=0.7,
        )

        result = await parse(session, branch, params)
        assert result is not None
        assert "username" in result or "user_score" in result

    async def test_parse_max_retries_respected(self, session_with_model):
        """Test max_retries limits LLM attempts."""
        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        call_count = 0

        async def mock_invoke_failing(**kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("API Error")

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_failing))

        params = ParseParams(
            text="Invalid JSON",
            imodel=model,
            max_retries=3,
        )

        result = await parse(session, branch, params)

        assert result is None
        assert call_count == 3  # Should have tried exactly max_retries times
