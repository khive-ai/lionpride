# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for parse operation.

Tests cover:
- Direct JSON extraction
- LLM fallback path
- Empty/sentinel input handling
- List result handling
- Permission checks
- Retry logic
- Error types with retryable semantics
"""

import pytest

from lionpride.errors import ConfigurationError, ExecutionError, ValidationError
from lionpride.operations.operate.parse import _direct_parse, parse
from lionpride.operations.operate.types import ParseParams


class TestDirectParse:
    """Tests for _direct_parse helper function."""

    def test_valid_json_returns_dict(self):
        """Test direct JSON extraction from valid JSON string."""
        text = '{"key": "value", "number": 42}'
        result = _direct_parse(
            text=text,
            target_keys=["key", "number"],
            similarity_threshold=0.85,
            handle_unmatched="force",
            structure_format="json",
        )
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_json_in_markdown_block(self):
        """Test extraction from markdown code block."""
        text = '```json\n{"name": "test"}\n```'
        result = _direct_parse(
            text=text,
            target_keys=["name"],
            similarity_threshold=0.85,
            handle_unmatched="force",
            structure_format="json",
        )
        assert result["name"] == "test"

    def test_invalid_json_raises_execution_error(self):
        """Test that invalid JSON raises ExecutionError (retryable)."""
        text = "this is not json at all"
        with pytest.raises(ExecutionError) as exc_info:
            _direct_parse(
                text=text,
                target_keys=["key"],
                similarity_threshold=0.85,
                handle_unmatched="force",
                structure_format="json",
            )
        assert exc_info.value.retryable is True

    def test_empty_extraction_raises_execution_error(self):
        """Test that empty extraction raises ExecutionError (retryable)."""
        text = "no json here at all"
        with pytest.raises(ExecutionError) as exc_info:
            _direct_parse(
                text=text,
                target_keys=["key"],
                similarity_threshold=0.85,
                handle_unmatched="force",
                structure_format="json",
            )
        assert exc_info.value.retryable is True
        assert "No JSON" in str(exc_info.value)

    def test_custom_format_without_parser_raises_configuration_error(self):
        """Test that custom format without parser raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            _direct_parse(
                text='{"key": "value"}',
                target_keys=["key"],
                similarity_threshold=0.85,
                handle_unmatched="force",
                structure_format="custom",
            )
        assert "requires a custom_parser" in str(exc_info.value)
        assert exc_info.value.retryable is False

    def test_custom_format_with_parser(self):
        """Test that custom format with parser works correctly."""

        def my_parser(text: str, target_keys: list[str], **kwargs) -> dict:
            # Simple mock parser that extracts key-value pairs
            return {"key": "parsed_value"}

        result = _direct_parse(
            text="some text",
            target_keys=["key"],
            similarity_threshold=0.85,
            handle_unmatched="force",
            structure_format="custom",
            custom_parser=my_parser,
        )
        assert result == {"key": "parsed_value"}

    def test_unsupported_format_raises_validation_error(self):
        """Test that unsupported format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _direct_parse(
                text='{"key": "value"}',
                target_keys=["key"],
                similarity_threshold=0.85,
                handle_unmatched="force",
                structure_format="xml",
            )
        assert "Unsupported structure_format" in str(exc_info.value)
        assert exc_info.value.retryable is False

    def test_no_target_keys_raises_validation_error(self):
        """Test that missing target_keys raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _direct_parse(
                text='{"key": "value"}',
                target_keys=[],
                similarity_threshold=0.85,
                handle_unmatched="force",
                structure_format="json",
            )
        assert "No target_keys" in str(exc_info.value)
        assert exc_info.value.retryable is False

    def test_list_result_takes_first_dict(self):
        """Test that list result extracts first dict."""
        # Multiple JSON blocks - returns list
        text = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        result = _direct_parse(
            text=text,
            target_keys=["first"],
            similarity_threshold=0.85,
            handle_unmatched="force",
            structure_format="json",
        )
        assert result["first"] == 1

    def test_target_keys_fuzzy_matching(self):
        """Test fuzzy key matching with target_keys."""
        text = '{"usr_name": "test", "val": 42}'
        result = _direct_parse(
            text=text,
            target_keys=["user_name", "value"],
            similarity_threshold=0.75,
            handle_unmatched="force",
            structure_format="json",
        )
        # Fuzzy matching should map keys
        assert result is not None
        assert "user_name" in result or "usr_name" in result


class TestParse:
    """Tests for main parse function."""

    @pytest.mark.asyncio
    async def test_sentinel_text_raises_validation_error(self, session_with_model):
        """Test that sentinel text raises ValidationError."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams()  # text is sentinel (not provided)

        with pytest.raises(ValidationError) as exc_info:
            await parse(session, branch, params)
        assert "No text provided" in str(exc_info.value)
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_direct_extract_success(self, session_with_model):
        """Test successful direct extraction without LLM fallback."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(text='{"key": "value"}', target_keys=["key"])
        result = await parse(session, branch, params)

        assert result["key"] == "value"

    @pytest.mark.asyncio
    async def test_max_retries_zero_raises_configuration_error(self, session_with_model):
        """Test that max_retries=0 raises ConfigurationError when direct parse fails."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(
            text="invalid json",
            target_keys=["key"],
            max_retries=0,  # Disable retries
        )

        with pytest.raises(ConfigurationError) as exc_info:
            await parse(session, branch, params)
        assert "max_retries" in str(exc_info.value)
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_max_retries_exceeds_limit_raises_validation_error(self, session_with_model):
        """Test that max_retries > 5 raises ValidationError."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(
            text="invalid json",
            target_keys=["key"],
            max_retries=10,  # Exceeds limit
        )

        with pytest.raises(ValidationError) as exc_info:
            await parse(session, branch, params)
        assert "cannot exceed 5" in str(exc_info.value)
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_parse_with_nested_json(self, session_with_model):
        """Test parsing nested JSON structures."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(
            text='{"outer": {"inner": "value"}, "array": [1, 2, 3]}',
            target_keys=["outer", "array"],
        )
        result = await parse(session, branch, params)

        assert result["outer"] == {"inner": "value"}
        assert result["array"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_parse_with_target_keys_and_fuzzy_match(self, session_with_model):
        """Test parse with target_keys triggers fuzzy matching."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(
            text='{"user_nam": "test", "val": 42}',
            target_keys=["user_name", "value"],
            similarity_threshold=0.75,
            handle_unmatched="force",
        )
        result = await parse(session, branch, params)

        # Should have attempted fuzzy matching
        assert result is not None


class TestParseErrorRetryability:
    """Tests verifying error retryability semantics."""

    def test_validation_errors_are_not_retryable(self):
        """Verify ValidationError has retryable=False by default."""
        error = ValidationError("test")
        assert error.retryable is False

    def test_configuration_errors_are_not_retryable(self):
        """Verify ConfigurationError has retryable=False by default."""
        error = ConfigurationError("test")
        assert error.retryable is False

    def test_execution_errors_are_retryable(self):
        """Verify ExecutionError has retryable=True by default."""
        error = ExecutionError("test")
        assert error.retryable is True

    def test_execution_error_retryable_can_be_overridden(self):
        """Verify ExecutionError retryable can be set to False."""
        error = ExecutionError("test", retryable=False)
        assert error.retryable is False

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates_immediately(self, session_with_model):
        """Test that non-retryable errors propagate without retry."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        # custom format without parser is ConfigurationError (not retryable)
        params = ParseParams(
            text='{"key": "value"}',
            target_keys=["key"],
            structure_format="custom",
            max_retries=5,  # Would retry if error was retryable
        )

        with pytest.raises(ConfigurationError):
            await parse(session, branch, params)
