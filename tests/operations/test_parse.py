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
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from lionpride import Event, EventStatus
from lionpride.operations.operate.parse import _llm_reparse, _try_direct_extract, parse
from lionpride.operations.operate.types import ParseParams
from lionpride.session import Session

# Import from conftest - use pytest_plugins or rely on conftest.py in same directory
# The conftest.py fixtures are automatically available


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


class TestTryDirectExtract:
    """Tests for _try_direct_extract helper function."""

    def test_valid_json_returns_dict(self):
        """Test direct JSON extraction from valid JSON string."""
        text = '{"key": "value", "number": 42}'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result == {"key": "value", "number": 42}

    def test_json_in_markdown_block(self):
        """Test extraction from markdown code block."""
        text = '```json\n{"name": "test"}\n```'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result == {"name": "test"}

    def test_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        text = "this is not json at all"
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_list_result_takes_first_dict(self):
        """Test that list result extracts first dict (line 115)."""
        # Multiple JSON blocks - returns list
        text = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result == {"first": 1}

    def test_empty_list_returns_none(self):
        """Test that empty list result returns None (line 115 else branch)."""
        # This happens when extract_json returns []
        text = "no json here"
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_non_dict_result_returns_none(self):
        """Test that non-dict extraction returns None (line 117)."""
        # JSON array at top level
        text = "[1, 2, 3]"
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_primitive_json_returns_none(self):
        """Test that primitive JSON (string, number) returns None."""
        text = '"just a string"'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result is None

    def test_target_keys_fuzzy_matching(self):
        """Test fuzzy key matching with target_keys."""
        text = '{"usr_name": "test", "val": 42}'
        result = _try_direct_extract(
            text=text,
            target_keys=["user_name", "value"],
            similarity_threshold=0.75,
            handle_unmatched="force",
        )
        # Fuzzy matching should map keys
        assert result is not None
        assert "user_name" in result or "usr_name" in result

    def test_empty_target_keys_no_fuzzy_matching(self):
        """Test that empty target_keys skips fuzzy matching."""
        text = '{"key": "value"}'
        result = _try_direct_extract(
            text=text,
            target_keys=[],
            similarity_threshold=0.85,
            handle_unmatched="force",
        )
        assert result == {"key": "value"}


class TestParse:
    """Tests for main parse function."""

    @pytest.mark.asyncio
    async def test_none_text_returns_none(self, session_with_model):
        """Test that None text returns None immediately (line 56)."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(text=None)
        result = await parse(session, branch, params)

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self, session_with_model):
        """Test that empty text returns None (line 56 - empty_as_sentinel)."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(text="")
        result = await parse(session, branch, params)

        assert result is None

    @pytest.mark.asyncio
    async def test_direct_extract_success(self, session_with_model):
        """Test successful direct extraction without LLM fallback."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(text='{"key": "value"}')
        result = await parse(session, branch, params)

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_no_imodel_returns_none_on_failed_extract(self, session_with_model):
        """Test that missing imodel returns None when direct extract fails (line 70)."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(
            text="invalid json",
            imodel=None,  # No fallback model
        )
        result = await parse(session, branch, params)

        assert result is None

    @pytest.mark.asyncio
    async def test_imodel_as_string_permission_check(self, session_with_model):
        """Test permission check with imodel as string (line 72-77)."""
        session, _mock_model = session_with_model
        # Branch without access to mock_model
        branch = session.create_branch(name="test", resources=set())

        params = ParseParams(
            text="invalid json",
            imodel="mock_model",  # String imodel name
        )

        with pytest.raises(PermissionError, match="cannot access model"):
            await parse(session, branch, params)

    @pytest.mark.asyncio
    async def test_imodel_as_model_permission_check(self, session_with_model):
        """Test permission check with iModel object (line 72-77)."""
        session, mock_model = session_with_model
        # Branch without access to mock_model
        branch = session.create_branch(name="test", resources=set())

        params = ParseParams(
            text="invalid json",
            imodel=mock_model,  # iModel object
        )

        with pytest.raises(PermissionError, match="cannot access model 'mock_model'"):
            await parse(session, branch, params)

    @pytest.mark.asyncio
    async def test_llm_fallback_success(self, session_with_model, mock_model):
        """Test successful LLM reparse fallback (lines 80-95)."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        # Mock to return valid JSON
        async def mock_reparse_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"reparsed": true}')

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_reparse_invoke))

        params = ParseParams(
            text="malformed json { reparsed: true }",
            imodel=mock_model.name,
            max_retries=3,
        )

        result = await parse(session, branch, params)

        assert result == {"reparsed": True}

    @pytest.mark.asyncio
    async def test_llm_fallback_retries_on_failure(self, session_with_model, mock_model):
        """Test that LLM fallback retries on exception (lines 96-97)."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        call_count = 0

        async def mock_failing_then_success(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Simulated API error")

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"success": true}')

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_failing_then_success))

        params = ParseParams(
            text="bad json",
            imodel=mock_model.name,
            max_retries=5,
        )

        result = await parse(session, branch, params)

        assert result == {"success": True}
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_llm_fallback_exhausted_returns_none(self, session_with_model, mock_model):
        """Test that exhausted retries returns None (line 99)."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        async def mock_always_fail(**kwargs):
            raise RuntimeError("Always fails")

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_always_fail))

        params = ParseParams(
            text="bad json",
            imodel=mock_model.name,
            max_retries=2,
        )

        result = await parse(session, branch, params)

        assert result is None

    @pytest.mark.asyncio
    async def test_llm_fallback_returns_none_if_reparse_fails(self, session_with_model, mock_model):
        """Test that LLM fallback returns None if reparse result is not parseable."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        # Mock to return non-JSON response
        async def mock_invalid_reparse(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data="still not valid json")

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_invalid_reparse))

        params = ParseParams(
            text="bad json",
            imodel=mock_model.name,
            max_retries=1,
        )

        result = await parse(session, branch, params)

        assert result is None


class TestLLMReparse:
    """Tests for _llm_reparse helper function (lines 144-187)."""

    @pytest.mark.asyncio
    async def test_llm_reparse_builds_instruction(self, session_with_model, mock_model):
        """Test that _llm_reparse builds correct instruction."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        captured_kwargs = {}

        async def capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"result": "ok"}')

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=capture_invoke))

        result = await _llm_reparse(
            session=session,
            branch=branch,
            text="malformed { json }",
            target_keys=[],  # No target keys to avoid fuzzy matching
            model_name=mock_model.name,
            similarity_threshold=0.85,
            handle_unmatched="ignore",  # Don't force keys
            imodel_kwargs={},
        )

        assert result == {"result": "ok"}
        # Check the instruction was built correctly
        assert "messages" in captured_kwargs
        messages = captured_kwargs["messages"]
        assert len(messages) > 0
        # The instruction should mention extracting JSON
        instruction_text = str(messages[-1])
        assert "JSON" in instruction_text or "json" in instruction_text.lower()

    @pytest.mark.asyncio
    async def test_llm_reparse_with_target_keys(self, session_with_model, mock_model):
        """Test that _llm_reparse includes target_keys in instruction (line 152)."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        captured_kwargs = {}

        async def capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    # Return JSON with exactly the target keys
                    self.execution.response = MockNormalizedResponse(
                        data='{"name": "test", "value": 42}'
                    )

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=capture_invoke))

        result = await _llm_reparse(
            session=session,
            branch=branch,
            text="bad json",
            target_keys=["name", "value"],
            model_name=mock_model.name,
            similarity_threshold=0.85,
            handle_unmatched="ignore",  # Don't force missing keys
            imodel_kwargs={},
        )

        # Verify result is correct
        assert result == {"name": "test", "value": 42}
        # Verify target keys are mentioned in instruction
        messages = captured_kwargs["messages"]
        instruction_text = str(messages[-1])
        assert "name" in instruction_text and "value" in instruction_text

    @pytest.mark.asyncio
    async def test_llm_reparse_without_target_keys(self, session_with_model, mock_model):
        """Test that _llm_reparse works without target_keys."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        async def mock_invoke(**kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"generic": "response"}')

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=mock_invoke))

        result = await _llm_reparse(
            session=session,
            branch=branch,
            text="bad json",
            target_keys=[],  # No target keys
            model_name=mock_model.name,
            similarity_threshold=0.85,
            handle_unmatched="force",
            imodel_kwargs={},
        )

        assert result == {"generic": "response"}

    @pytest.mark.asyncio
    async def test_llm_reparse_non_string_result_returns_none(self, session_with_model, mock_model):
        """Test that non-string generate result returns None (line 187)."""
        from unittest.mock import patch

        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        # Mock the generate function to return a non-string value (e.g., a Message)
        async def mock_generate(*args, **kwargs):
            # Return a non-string value - triggers line 187
            return {"not": "a string"}

        # Patch at the generate module level since it's imported inside _llm_reparse
        with patch("lionpride.operations.operate.generate.generate", side_effect=mock_generate):
            result = await _llm_reparse(
                session=session,
                branch=branch,
                text="bad json",
                target_keys=[],
                model_name=mock_model.name,
                similarity_threshold=0.85,
                handle_unmatched="ignore",
                imodel_kwargs={},
            )

        # Should return None because result was not a string
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_reparse_with_imodel_kwargs(self, session_with_model, mock_model):
        """Test that _llm_reparse passes imodel_kwargs."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        captured_kwargs = {}

        async def capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"key": "value"}')

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=capture_invoke))

        await _llm_reparse(
            session=session,
            branch=branch,
            text="bad json",
            target_keys=[],
            model_name=mock_model.name,
            similarity_threshold=0.85,
            handle_unmatched="force",
            imodel_kwargs={"temperature": 0.5, "max_tokens": 100},
        )

        # imodel_kwargs should be passed through
        assert captured_kwargs.get("temperature") == 0.5
        assert captured_kwargs.get("max_tokens") == 100

    @pytest.mark.asyncio
    async def test_llm_reparse_with_poll_params(self, session_with_model, mock_model):
        """Test that _llm_reparse passes poll_timeout and poll_interval."""
        session, _ = session_with_model
        branch = session.create_branch(name="test", resources={mock_model.name})

        captured_kwargs = {}

        async def capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockNormalizedResponse(data='{"key": "value"}')

            return MockCalling()

        object.__setattr__(mock_model, "invoke", AsyncMock(side_effect=capture_invoke))

        await _llm_reparse(
            session=session,
            branch=branch,
            text="bad json",
            target_keys=[],
            model_name=mock_model.name,
            similarity_threshold=0.85,
            handle_unmatched="force",
            imodel_kwargs={},
            poll_timeout=30.0,
            poll_interval=0.5,
        )

        assert captured_kwargs.get("poll_timeout") == 30.0
        assert captured_kwargs.get("poll_interval") == 0.5


class TestParseEdgeCases:
    """Edge case tests for parse operation."""

    @pytest.mark.asyncio
    async def test_parse_with_whitespace_only_text(self, session_with_model):
        """Test that whitespace-only text is treated as empty."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(text="   \n\t  ")
        result = await parse(session, branch, params)

        # Whitespace-only should be treated as valid text (not sentinel)
        # but won't extract any JSON
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_with_nested_json(self, session_with_model):
        """Test parsing nested JSON structures."""
        session, _ = session_with_model
        branch = session.create_branch(name="test")

        params = ParseParams(text='{"outer": {"inner": "value"}, "array": [1, 2, 3]}')
        result = await parse(session, branch, params)

        assert result == {"outer": {"inner": "value"}, "array": [1, 2, 3]}

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
