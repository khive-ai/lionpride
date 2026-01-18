# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
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
- Custom parser exception handling
- extract_json exception handling
- fuzzy_validate_mapping exception handling
- _llm_reparse function
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lionpride.errors import (
    ConfigurationError,
    ExecutionError,
    LionprideError,
    ValidationError,
)
from lionpride.operations.operate.parse import _direct_parse, _llm_reparse, parse
from lionpride.operations.operate.types import ParseParams
from lionpride.types._sentinel import Unset


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

    def test_custom_parser_exception_raises_execution_error(self):
        """Test that custom parser exception wraps in ExecutionError (lines 104-110)."""

        def failing_parser(text: str, target_keys: list[str], **kwargs) -> dict:
            raise ValueError("Parser failed!")

        with pytest.raises(ExecutionError) as exc_info:
            _direct_parse(
                text="some text",
                target_keys=["key"],
                similarity_threshold=0.85,
                handle_unmatched="force",
                structure_format="custom",
                custom_parser=failing_parser,
            )
        assert "Custom parser failed" in str(exc_info.value)
        assert exc_info.value.retryable is True
        # Verify the cause is preserved
        assert exc_info.value.__cause__ is not None
        assert "Parser failed!" in str(exc_info.value.__cause__)

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

    def test_extract_json_exception_raises_execution_error(self):
        """Test that extract_json exception wraps in ExecutionError (lines 118-125)."""
        with patch("lionpride.operations.operate.parse.extract_json") as mock_extract:
            mock_extract.side_effect = ValueError("JSON parsing error")

            with pytest.raises(ExecutionError) as exc_info:
                _direct_parse(
                    text='{"key": "value"}',
                    target_keys=["key"],
                    similarity_threshold=0.85,
                    handle_unmatched="force",
                    structure_format="json",
                )
            assert "Failed to extract JSON" in str(exc_info.value)
            assert exc_info.value.retryable is True

    def test_fuzzy_validate_mapping_exception_raises_execution_error(self):
        """Test that fuzzy_validate_mapping exception wraps in ExecutionError (lines 139-145)."""
        with patch("lionpride.operations.operate.parse.fuzzy_validate_mapping") as mock_validate:
            mock_validate.side_effect = ValueError("Validation error")

            with pytest.raises(ExecutionError) as exc_info:
                _direct_parse(
                    text='{"key": "value"}',
                    target_keys=["key"],
                    similarity_threshold=0.85,
                    handle_unmatched="force",
                    structure_format="json",
                )
            assert "Failed to validate extracted JSON" in str(exc_info.value)
            assert exc_info.value.retryable is True


class TestParse:
    """Tests for main parse function."""

    @pytest.fixture
    def mock_session_branch(self):
        """Create mock session and branch for testing."""
        session = MagicMock()
        session.id = "mock-session-id"
        branch = MagicMock()
        branch.id = "mock-branch-id"
        return session, branch

    @pytest.mark.asyncio
    async def test_sentinel_text_raises_validation_error(self, mock_session_branch):
        """Test that sentinel text raises ValidationError."""
        session, branch = mock_session_branch

        params = ParseParams(text="placeholder", target_keys=["key"])
        # Force set text to a sentinel value
        object.__setattr__(params, "text", Unset)

        with pytest.raises(ValidationError) as exc_info:
            await parse(session, branch, params)
        assert "parse requires 'text' parameter" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_direct_extract_success(self, mock_session_branch):
        """Test successful direct extraction without LLM fallback."""
        session, branch = mock_session_branch

        params = ParseParams(text='{"key": "value"}', target_keys=["key"])
        result = await parse(session, branch, params)

        assert result["key"] == "value"

    @pytest.mark.asyncio
    async def test_max_retries_zero_raises_configuration_error(self, mock_session_branch):
        """Test that max_retries=0 raises ConfigurationError when direct parse fails."""
        session, branch = mock_session_branch

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
    async def test_max_retries_exceeds_limit_raises_validation_error(self, mock_session_branch):
        """Test that max_retries > 5 raises ValidationError."""
        session, branch = mock_session_branch

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
    async def test_parse_with_nested_json(self, mock_session_branch):
        """Test parsing nested JSON structures."""
        session, branch = mock_session_branch

        params = ParseParams(
            text='{"outer": {"inner": "value"}, "array": [1, 2, 3]}',
            target_keys=["outer", "array"],
        )
        result = await parse(session, branch, params)

        assert result["outer"] == {"inner": "value"}
        assert result["array"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_parse_with_target_keys_and_fuzzy_match(self, mock_session_branch):
        """Test parse with target_keys triggers fuzzy matching."""
        session, branch = mock_session_branch

        params = ParseParams(
            text='{"user_nam": "test", "val": 42}',
            target_keys=["user_name", "value"],
            similarity_threshold=0.75,
            handle_unmatched="force",
        )
        result = await parse(session, branch, params)

        # Should have attempted fuzzy matching
        assert result is not None

    @pytest.mark.asyncio
    async def test_non_retryable_lionpride_error_propagates(self, mock_session_branch):
        """Test that non-retryable LionprideError is re-raised immediately (lines 49-50)."""
        session, branch = mock_session_branch

        # ConfigurationError has retryable=False by default
        # Test with custom format without parser (ConfigurationError, not retryable)
        params = ParseParams(
            text='{"key": "value"}',
            target_keys=["key"],
            structure_format="custom",
            max_retries=5,  # Would retry if error was retryable
        )

        with pytest.raises(ConfigurationError):
            await parse(session, branch, params)

    @pytest.mark.asyncio
    async def test_non_retryable_lionpride_error_with_explicit_flag(self, mock_session_branch):
        """Test non-retryable LionprideError with explicit retryable=False (lines 49-50)."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.parse._direct_parse") as mock_direct:
            # Raise a LionprideError with explicit retryable=False
            mock_direct.side_effect = LionprideError("Non-retryable error", retryable=False)

            params = ParseParams(
                text='{"key": "value"}',
                target_keys=["key"],
                max_retries=5,  # Would retry if error was retryable
            )

            with pytest.raises(LionprideError) as exc_info:
                await parse(session, branch, params)
            assert exc_info.value.retryable is False
            assert "Non-retryable error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_reparse_retry_loop_success(self, mock_session_branch):
        """Test LLM reparse succeeds on first retry (lines 67-75)."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.parse._llm_reparse") as mock_reparse:
            mock_reparse.return_value = {"key": "reparsed_value"}

            params = ParseParams(
                text="invalid json",
                target_keys=["key"],
                max_retries=3,
            )

            result = await parse(session, branch, params)

            assert result == {"key": "reparsed_value"}
            mock_reparse.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_reparse_retry_loop_retries_on_retryable_error(self, mock_session_branch):
        """Test LLM reparse retries on retryable LionprideError (lines 67-78)."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.parse._llm_reparse") as mock_reparse:
            # First two attempts raise retryable error, third succeeds
            mock_reparse.side_effect = [
                LionprideError("Retry 1", retryable=True),
                LionprideError("Retry 2", retryable=True),
                {"key": "success"},
            ]

            params = ParseParams(
                text="invalid json",
                target_keys=["key"],
                max_retries=3,
            )

            result = await parse(session, branch, params)

            assert result == {"key": "success"}
            assert mock_reparse.call_count == 3

    @pytest.mark.asyncio
    async def test_llm_reparse_stops_on_non_retryable_error(self, mock_session_branch):
        """Test LLM reparse stops on non-retryable LionprideError (lines 76-78)."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.parse._llm_reparse") as mock_reparse:
            # First attempt raises non-retryable error
            mock_reparse.side_effect = LionprideError("Non-retryable", retryable=False)

            params = ParseParams(
                text="invalid json",
                target_keys=["key"],
                max_retries=5,
            )

            with pytest.raises(LionprideError) as exc_info:
                await parse(session, branch, params)

            assert exc_info.value.retryable is False
            # Should only be called once due to non-retryable
            assert mock_reparse.call_count == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_execution_error(self, mock_session_branch):
        """Test that exhausting all retries raises ExecutionError (lines 80-83)."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.parse._llm_reparse") as mock_reparse:
            # All attempts raise retryable error
            mock_reparse.side_effect = LionprideError("Always fail", retryable=True)

            params = ParseParams(
                text="invalid json",
                target_keys=["key"],
                max_retries=3,
            )

            with pytest.raises(ExecutionError) as exc_info:
                await parse(session, branch, params)

            assert "All parse attempts" in str(exc_info.value)
            assert exc_info.value.retryable is True
            assert mock_reparse.call_count == 3

    @pytest.mark.asyncio
    async def test_generic_exception_falls_through_to_llm_reparse(self, mock_session_branch):
        """Test that generic Exception in _direct_parse triggers LLM reparse (lines 51-55)."""
        session, branch = mock_session_branch

        with (
            patch("lionpride.operations.operate.parse._direct_parse") as mock_direct,
            patch("lionpride.operations.operate.parse._llm_reparse") as mock_reparse,
        ):
            # First call raises generic Exception (not LionprideError)
            # This should trigger the except Exception path (lines 51-55)
            mock_direct.side_effect = RuntimeError("Some unexpected error")
            mock_reparse.return_value = {"key": "reparsed"}

            params = ParseParams(
                text='{"key": "value"}',
                target_keys=["key"],
                max_retries=3,
            )

            result = await parse(session, branch, params)

            # Direct parse was called but raised generic exception
            mock_direct.assert_called_once()
            # LLM reparse should be triggered
            mock_reparse.assert_called_once()
            assert result == {"key": "reparsed"}


class TestLlmReparse:
    """Tests for _llm_reparse helper function (lines 149-188)."""

    @pytest.fixture
    def mock_session_branch(self):
        """Create mock session and branch for testing."""
        session = MagicMock()
        session.id = "mock-session-id"
        branch = MagicMock()
        branch.id = "mock-branch-id"
        return session, branch

    @pytest.mark.asyncio
    async def test_llm_reparse_success(self, mock_session_branch):
        """Test _llm_reparse calls generate and parses result (lines 155-188)."""
        session, branch = mock_session_branch

        # Patch the generate function in the generate module (where it's imported from)
        with patch("lionpride.operations.operate.generate.generate") as mock_generate:
            # Mock generate to return valid JSON
            mock_generate.return_value = '{"key": "value"}'

            params = ParseParams(
                text="unparseable text",
                target_keys=["key"],
                max_retries=3,
            )

            result = await _llm_reparse(session, branch, params)

            assert result == {"key": "value"}
            mock_generate.assert_called_once()
            # Verify the call arguments
            call_args = mock_generate.call_args
            assert call_args[0][0] == session
            assert call_args[0][1] == branch

    @pytest.mark.asyncio
    async def test_llm_reparse_constructs_proper_instruction(self, mock_session_branch):
        """Test _llm_reparse constructs instruction with target_keys."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.generate.generate") as mock_generate:
            mock_generate.return_value = '{"name": "John", "age": 30}'

            params = ParseParams(
                text="Some text about John who is 30",
                target_keys=["name", "age"],
                max_retries=3,
            )

            result = await _llm_reparse(session, branch, params)

            assert result is not None
            # Verify generate was called with GenerateParams
            call_args = mock_generate.call_args
            gen_params = call_args[0][2]
            assert gen_params.return_as == "text"

    @pytest.mark.asyncio
    async def test_llm_reparse_uses_imodel_from_params(self, mock_session_branch):
        """Test _llm_reparse uses imodel from params."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.generate.generate") as mock_generate:
            mock_generate.return_value = '{"key": "value"}'

            params = ParseParams(
                text="unparseable text",
                target_keys=["key"],
                max_retries=3,
                imodel="gpt-4",
                imodel_kwargs={"temperature": 0.5},
            )

            await _llm_reparse(session, branch, params)

            # Verify the GenerateParams has correct imodel
            call_args = mock_generate.call_args
            gen_params = call_args[0][2]
            assert gen_params.imodel == "gpt-4"
            assert gen_params.imodel_kwargs == {"temperature": 0.5}

    @pytest.mark.asyncio
    async def test_llm_reparse_passes_poll_params(self, mock_session_branch):
        """Test _llm_reparse passes poll_timeout and poll_interval."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.generate.generate") as mock_generate:
            mock_generate.return_value = '{"key": "value"}'

            params = ParseParams(
                text="unparseable text",
                target_keys=["key"],
                max_retries=3,
            )

            await _llm_reparse(session, branch, params, poll_timeout=30.0, poll_interval=0.5)

            # Verify poll params were passed
            call_args = mock_generate.call_args
            assert call_args[0][3] == 30.0  # poll_timeout
            assert call_args[0][4] == 0.5  # poll_interval

    @pytest.mark.asyncio
    async def test_llm_reparse_direct_parse_failure(self, mock_session_branch):
        """Test _llm_reparse handles direct_parse failure from LLM result."""
        session, branch = mock_session_branch

        with patch("lionpride.operations.operate.generate.generate") as mock_generate:
            # Mock generate to return invalid JSON
            mock_generate.return_value = "still not valid json"

            params = ParseParams(
                text="unparseable text",
                target_keys=["key"],
                max_retries=3,
            )

            with pytest.raises(ExecutionError):
                await _llm_reparse(session, branch, params)


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

    def test_lionpride_error_defaults_to_retryable(self):
        """Verify base LionprideError defaults to retryable=True."""
        error = LionprideError("test")
        assert error.retryable is True

    def test_lionpride_error_retryable_can_be_overridden(self):
        """Verify LionprideError retryable can be set to False."""
        error = LionprideError("test", retryable=False)
        assert error.retryable is False
