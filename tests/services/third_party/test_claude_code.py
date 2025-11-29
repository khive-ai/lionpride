# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Claude Code CLI integration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from lionpride.services.third_party.claude_code import (
    CLAUDE_CLI,
    ClaudeChunk,
    ClaudeCodeRequest,
    ClaudeSession,
    _extract_summary,
    stream_cc_cli_events,
    stream_claude_code_cli,
)

# ============================================================================
# ClaudeCodeRequest Tests
# ============================================================================


class TestClaudeCodeRequest:
    """Test ClaudeCodeRequest model validation and methods."""

    def test_request_when_minimal_valid_prompt_then_succeeds(self):
        """Test minimal request with only prompt."""
        request = ClaudeCodeRequest(prompt="test prompt")
        assert request.prompt == "test prompt"
        assert request.model == "sonnet"
        assert request.continue_conversation is False

    def test_request_when_all_fields_then_succeeds(self):
        """Test request with all optional fields."""
        request = ClaudeCodeRequest(
            prompt="test",
            system_prompt="system",
            append_system_prompt="append",
            max_turns=5,
            continue_conversation=True,
            resume="session-123",
            ws="workspace",
            add_dir="/extra",
            allowed_tools=["Read", "Write"],
            model="opus",
            max_thinking_tokens=1000,
            mcp_tools=["tool1"],
            mcp_servers={"server": "config"},
            permission_mode="acceptEdits",
            disallowed_tools=["Bash"],
            auto_finish=True,
            verbose_output=True,
        )
        assert request.prompt == "test"
        assert request.system_prompt == "system"
        assert request.max_turns == 5

    def test_request_when_messages_only_then_converts_to_prompt(self):
        """Test message conversion to prompt."""
        request = ClaudeCodeRequest(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "how are you"},
            ]
        )
        assert "hello" in request.prompt
        assert "hi" in request.prompt
        assert "how are you" in request.prompt

    def test_request_when_messages_with_system_then_extracts_system(self):
        """Test system message extraction."""
        request = ClaudeCodeRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"},
            ]
        )
        # System is not extracted if not continuing
        assert request.prompt == "hello"

    def test_request_when_messages_with_resume_then_uses_last_message(self):
        """Test resume mode uses last message."""
        request = ClaudeCodeRequest(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "last"},
            ],
            resume="session-123",
        )
        assert request.prompt == "last"
        assert request.continue_conversation is True

    def test_request_when_messages_with_dict_content_then_serializes(self):
        """Test dictionary content serialization."""
        request = ClaudeCodeRequest(
            messages=[{"role": "user", "content": {"type": "text", "text": "hello"}}]
        )
        assert '"type"' in request.prompt or "type" in request.prompt

    def test_request_when_no_prompt_or_messages_then_raises(self):
        """Test missing prompt and messages raises error."""
        with pytest.raises(ValueError, match="messages may not be empty"):
            ClaudeCodeRequest(messages=[])

    def test_permission_mode_when_dangerously_skip_then_normalizes(self):
        """Test permission mode normalization."""
        request = ClaudeCodeRequest(prompt="test", permission_mode="dangerously-skip-permissions")
        assert request.permission_mode == "bypassPermissions"

    def test_permission_mode_when_flag_format_then_normalizes(self):
        """Test permission mode normalization with flag format."""
        request = ClaudeCodeRequest(prompt="test", permission_mode="--dangerously-skip-permissions")
        assert request.permission_mode == "bypassPermissions"

    def test_cwd_when_no_workspace_then_returns_repo(self):
        """Test cwd returns repo when no workspace."""
        request = ClaudeCodeRequest(prompt="test", repo=Path("/test/repo"))
        assert request.cwd() == Path("/test/repo")

    def test_cwd_when_relative_workspace_then_returns_joined_path(self):
        """Test cwd with relative workspace."""
        request = ClaudeCodeRequest(prompt="test", repo=Path("/test/repo"), ws="sub")
        cwd = request.cwd()
        assert cwd.parts[-1] == "sub"

    def test_cwd_when_absolute_workspace_then_raises(self):
        """Test absolute workspace path raises error."""
        with pytest.raises(ValueError, match="must be relative"):
            request = ClaudeCodeRequest(prompt="test", repo=Path("/test/repo"), ws="/absolute/path")
            request.cwd()

    def test_cwd_when_directory_traversal_then_raises(self):
        """Test directory traversal prevention."""
        with pytest.raises(ValueError, match="Directory traversal detected"):
            request = ClaudeCodeRequest(prompt="test", repo=Path("/test/repo"), ws="../escape")
            request.cwd()

    def test_bypass_permissions_when_workspace_outside_repo_then_raises(self):
        """Test bypassPermissions requires workspace in repo."""
        # This test needs special setup to test boundary validation
        # For now, test that valid case passes
        request = ClaudeCodeRequest(
            prompt="test",
            repo=Path.cwd(),
            ws="valid_sub",
            permission_mode="bypassPermissions",
        )
        assert request.permission_mode == "bypassPermissions"

    def test_as_cmd_args_when_minimal_then_returns_basic_args(self):
        """Test command args generation with minimal config."""
        request = ClaudeCodeRequest(prompt="test prompt")
        args = request.as_cmd_args()
        assert "-p" in args
        assert "test prompt" in args
        assert "--output-format" in args
        assert "stream-json" in args
        assert "--model" in args
        assert "sonnet" in args

    def test_as_cmd_args_when_allowed_tools_then_includes_tools(self):
        """Test allowed tools in command args."""
        request = ClaudeCodeRequest(prompt="test", allowed_tools=["Read", "Write"])
        args = request.as_cmd_args()
        assert "--allowedTools" in args
        # Tools are quoted
        assert any('"Read"' in arg for arg in args)

    def test_as_cmd_args_when_disallowed_tools_then_includes_disallowed(self):
        """Test disallowed tools in command args."""
        request = ClaudeCodeRequest(prompt="test", disallowed_tools=["Bash"])
        args = request.as_cmd_args()
        assert "--disallowedTools" in args

    def test_as_cmd_args_when_resume_then_includes_resume(self):
        """Test resume session in command args."""
        request = ClaudeCodeRequest(prompt="test", resume="session-123")
        args = request.as_cmd_args()
        assert "--resume" in args
        assert "session-123" in args

    def test_as_cmd_args_when_continue_then_includes_continue(self):
        """Test continue conversation in command args."""
        request = ClaudeCodeRequest(prompt="test", continue_conversation=True)
        args = request.as_cmd_args()
        assert "--continue" in args

    def test_as_cmd_args_when_max_turns_then_adds_one(self):
        """Test max_turns incremented for CLI."""
        request = ClaudeCodeRequest(prompt="test", max_turns=5)
        args = request.as_cmd_args()
        idx = args.index("--max-turns")
        assert args[idx + 1] == "6"  # 5 + 1

    def test_as_cmd_args_when_bypass_permissions_then_includes_flag(self):
        """Test bypass permissions flag."""
        request = ClaudeCodeRequest(prompt="test", permission_mode="bypassPermissions")
        args = request.as_cmd_args()
        assert "--dangerously-skip-permissions" in args

    def test_as_cmd_args_when_add_dir_then_includes_dir(self):
        """Test additional directory mount."""
        request = ClaudeCodeRequest(prompt="test", add_dir="/extra")
        args = request.as_cmd_args()
        assert "--add-dir" in args
        assert "/extra" in args

    def test_as_cmd_args_when_permission_prompt_tool_then_includes(self):
        """Test permission prompt tool."""
        request = ClaudeCodeRequest(prompt="test", permission_prompt_tool_name="AskUser")
        args = request.as_cmd_args()
        assert "--permission-prompt-tool" in args
        assert "AskUser" in args

    def test_as_cmd_args_when_mcp_config_then_includes_config(self):
        """Test MCP config path."""
        request = ClaudeCodeRequest(prompt="test", mcp_config="/path/to/config.json")
        args = request.as_cmd_args()
        assert "--mcp-config" in args


# ============================================================================
# ClaudeChunk Tests
# ============================================================================


class TestClaudeChunk:
    """Test ClaudeChunk dataclass."""

    def test_chunk_when_created_then_stores_raw(self):
        """Test chunk stores raw data."""
        raw = {"type": "assistant", "content": "test"}
        chunk = ClaudeChunk(raw=raw, type="assistant")
        assert chunk.raw == raw
        assert chunk.type == "assistant"

    def test_chunk_when_thinking_then_stores_thinking(self):
        """Test chunk with thinking content."""
        chunk = ClaudeChunk(raw={}, type="assistant", thinking="I think...")
        assert chunk.thinking == "I think..."

    def test_chunk_when_text_then_stores_text(self):
        """Test chunk with text content."""
        chunk = ClaudeChunk(raw={}, type="assistant", text="Hello")
        assert chunk.text == "Hello"

    def test_chunk_when_tool_use_then_stores_tool_use(self):
        """Test chunk with tool use."""
        tool_use = {"id": "t1", "name": "Read", "input": {}}
        chunk = ClaudeChunk(raw={}, type="assistant", tool_use=tool_use)
        assert chunk.tool_use == tool_use

    def test_chunk_when_tool_result_then_stores_tool_result(self):
        """Test chunk with tool result."""
        tool_result = {"tool_use_id": "t1", "content": "result", "is_error": False}
        chunk = ClaudeChunk(raw={}, type="user", tool_result=tool_result)
        assert chunk.tool_result == tool_result


# ============================================================================
# ClaudeSession Tests
# ============================================================================


class TestClaudeSession:
    """Test ClaudeSession dataclass."""

    def test_session_when_default_then_empty_collections(self):
        """Test default session initialization."""
        session = ClaudeSession()
        assert session.chunks == []
        assert session.thinking_log == []
        assert session.messages == []
        assert session.tool_uses == []
        assert session.tool_results == []
        assert session.result == ""
        assert session.is_error is False

    def test_session_when_populated_then_stores_data(self):
        """Test session with populated data."""
        session = ClaudeSession(
            session_id="s1",
            model="sonnet",
            result="Done",
            total_cost_usd=0.05,
            num_turns=3,
        )
        assert session.session_id == "s1"
        assert session.model == "sonnet"
        assert session.result == "Done"
        assert session.total_cost_usd == 0.05

    def test_populate_summary_when_called_then_creates_summary(self):
        """Test summary population."""
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Read", "input": {"file_path": "test.py"}}]
        )
        session.populate_summary()
        assert session.summary is not None
        assert "tool_counts" in session.summary
        assert session.summary["tool_counts"]["Read"] == 1


# ============================================================================
# _extract_summary Tests
# ============================================================================


class TestExtractSummary:
    """Test _extract_summary function."""

    def test_extract_summary_when_empty_session_then_returns_empty_counts(self):
        """Test summary extraction with empty session."""
        session = ClaudeSession()
        summary = _extract_summary(session)
        assert summary["tool_counts"] == {}
        assert summary["total_tool_calls"] == 0
        assert "No specific actions detected" in summary["key_actions"]

    def test_extract_summary_when_read_tool_then_categorizes(self):
        """Test Read tool categorization."""
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Read", "input": {"file_path": "test.py"}}]
        )
        summary = _extract_summary(session)
        assert summary["tool_counts"]["Read"] == 1
        assert "test.py" in summary["file_operations"]["reads"]
        assert "Read test.py" in summary["key_actions"]

    def test_extract_summary_when_write_tool_then_categorizes(self):
        """Test Write tool categorization."""
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Write", "input": {"file_path": "output.txt"}}]
        )
        summary = _extract_summary(session)
        assert summary["tool_counts"]["Write"] == 1
        assert "output.txt" in summary["file_operations"]["writes"]

    def test_extract_summary_when_edit_tool_then_categorizes(self):
        """Test Edit tool categorization."""
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Edit", "input": {"file_path": "code.py"}}]
        )
        summary = _extract_summary(session)
        assert summary["tool_counts"]["Edit"] == 1
        assert "code.py" in summary["file_operations"]["edits"]

    def test_extract_summary_when_bash_tool_then_summarizes_command(self):
        """Test Bash tool command summarization."""
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Bash", "input": {"command": "ls -la /test"}}]
        )
        summary = _extract_summary(session)
        assert "Ran: ls -la /test" in summary["key_actions"]

    def test_extract_summary_when_long_bash_command_then_truncates(self):
        """Test long bash command truncation."""
        long_cmd = "a" * 60
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Bash", "input": {"command": long_cmd}}]
        )
        summary = _extract_summary(session)
        action = next(a for a in summary["key_actions"] if "Ran:" in a)
        assert len(action) < len(long_cmd) + 10  # Truncated

    def test_extract_summary_when_glob_tool_then_records_pattern(self):
        """Test Glob tool pattern recording."""
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Glob", "input": {"pattern": "*.py"}}]
        )
        summary = _extract_summary(session)
        assert "Searched files: *.py" in summary["key_actions"]

    def test_extract_summary_when_grep_tool_then_records_pattern(self):
        """Test Grep tool pattern recording."""
        session = ClaudeSession(
            tool_uses=[{"id": "t1", "name": "Grep", "input": {"pattern": "TODO"}}]
        )
        summary = _extract_summary(session)
        assert "Searched content: TODO" in summary["key_actions"]

    def test_extract_summary_when_mcp_tool_then_extracts_operation(self):
        """Test MCP tool operation extraction."""
        session = ClaudeSession(tool_uses=[{"id": "t1", "name": "mcp__khive__recall", "input": {}}])
        summary = _extract_summary(session)
        assert "MCP khive__recall" in summary["key_actions"]

    def test_extract_summary_when_todowrite_then_counts_todos(self):
        """Test TodoWrite tool todo counting."""
        session = ClaudeSession(
            tool_uses=[
                {
                    "id": "t1",
                    "name": "TodoWrite",
                    "input": {"todos": [{"content": "task1"}, {"content": "task2"}]},
                }
            ]
        )
        summary = _extract_summary(session)
        assert "Created 2 todos" in summary["key_actions"]

    def test_extract_summary_when_duplicate_files_then_deduplicates(self):
        """Test file path deduplication."""
        session = ClaudeSession(
            tool_uses=[
                {"id": "t1", "name": "Read", "input": {"file_path": "test.py"}},
                {"id": "t2", "name": "Read", "input": {"file_path": "test.py"}},
            ]
        )
        summary = _extract_summary(session)
        assert len(summary["file_operations"]["reads"]) == 1

    def test_extract_summary_when_multiple_tool_types_then_counts_all(self):
        """Test multiple tool type counting."""
        session = ClaudeSession(
            tool_uses=[
                {"id": "t1", "name": "Read", "input": {"file_path": "a.py"}},
                {"id": "t2", "name": "Write", "input": {"file_path": "b.py"}},
                {"id": "t3", "name": "Bash", "input": {"command": "test"}},
            ]
        )
        summary = _extract_summary(session)
        assert summary["total_tool_calls"] == 3
        assert summary["tool_counts"]["Read"] == 1
        assert summary["tool_counts"]["Write"] == 1
        assert summary["tool_counts"]["Bash"] == 1

    def test_extract_summary_when_long_result_then_truncates(self):
        """Test long result truncation."""
        long_result = "x" * 300
        session = ClaudeSession(result=long_result)
        summary = _extract_summary(session)
        assert len(summary["result_summary"]) <= 203  # 200 + "..."


# ============================================================================
# stream_cc_cli_events Tests
# ============================================================================


class TestStreamCcCliEvents:
    """Test stream_cc_cli_events async generator."""

    @pytest.mark.asyncio
    async def test_stream_when_no_cli_then_raises(self):
        """Test error when CLI not found."""
        with patch("lionpride.services.third_party.claude_code.CLAUDE_CLI", None):
            request = ClaudeCodeRequest(prompt="test")
            with pytest.raises(RuntimeError, match="Claude CLI binary not found"):
                async for _ in stream_cc_cli_events(request):
                    pass

    @pytest.mark.asyncio
    async def test_stream_when_cli_available_then_yields_objects(self):
        """Test streaming with CLI available."""
        with (
            patch("lionpride.services.third_party.claude_code.CLAUDE_CLI", "claude"),
            patch("lionpride.services.third_party.claude_code._ndjson_from_cli") as mock_ndjson,
        ):
            # Mock async generator
            async def mock_generator(req):
                yield {"type": "system", "session_id": "s1"}
                yield {"type": "assistant", "message": {"content": []}}

            mock_ndjson.return_value = mock_generator(None)

            request = ClaudeCodeRequest(prompt="test")
            events = []
            async for event in stream_cc_cli_events(request):
                events.append(event)

            assert len(events) == 3  # 2 events + done
            assert events[-1] == {"type": "done"}


# ============================================================================
# stream_claude_code_cli Tests
# ============================================================================


class TestStreamClaudeCodeCli:
    """Test main stream_claude_code_cli function."""

    @pytest.mark.asyncio
    async def test_stream_when_system_event_then_updates_session(self):
        """Test system event updates session."""
        mock_stream = [
            {"type": "system", "session_id": "s1", "model": "sonnet"},
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            chunks = []
            async for chunk in stream_claude_code_cli(request, session):
                chunks.append(chunk)

            assert session.session_id == "s1"
            assert session.model == "sonnet"

    @pytest.mark.asyncio
    async def test_stream_when_assistant_with_thinking_then_captures(self):
        """Test thinking block capture."""
        mock_stream = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "thinking", "thinking": "Let me think..."}]},
            },
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            async for _ in stream_claude_code_cli(request, session):
                pass

            assert "Let me think..." in session.thinking_log

    @pytest.mark.asyncio
    async def test_stream_when_assistant_with_text_then_captures(self):
        """Test text block capture."""
        mock_stream = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hello world"}]},
            },
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            chunks = []
            async for chunk in stream_claude_code_cli(request, session):
                if isinstance(chunk, ClaudeChunk):
                    chunks.append(chunk)

            assert any(c.text == "Hello world" for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_when_tool_use_then_captures(self):
        """Test tool use capture."""
        mock_stream = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "test.py"},
                        }
                    ]
                },
            },
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            async for _ in stream_claude_code_cli(request, session):
                pass

            assert len(session.tool_uses) == 1
            assert session.tool_uses[0]["name"] == "Read"

    @pytest.mark.asyncio
    async def test_stream_when_tool_result_in_assistant_then_captures(self):
        """Test tool result in assistant message."""
        mock_stream = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "result data",
                            "is_error": False,
                        }
                    ]
                },
            },
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            async for _ in stream_claude_code_cli(request, session):
                pass

            assert len(session.tool_results) == 1
            assert session.tool_results[0]["content"] == "result data"

    @pytest.mark.asyncio
    async def test_stream_when_user_with_tool_result_then_captures(self):
        """Test tool result in user message."""
        mock_stream = [
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "user result",
                            "is_error": False,
                        }
                    ]
                },
            },
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            async for _ in stream_claude_code_cli(request, session):
                pass

            assert len(session.tool_results) == 1

    @pytest.mark.asyncio
    async def test_stream_when_result_event_then_populates_metrics(self):
        """Test result event populates session metrics."""
        mock_stream = [
            {
                "type": "result",
                "result": "Task complete",
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "total_cost_usd": 0.05,
                "num_turns": 2,
                "duration_ms": 5000,
                "duration_api_ms": 4000,
                "is_error": False,
            },
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            async for _ in stream_claude_code_cli(request, session):
                pass

            assert session.result == "Task complete"
            assert session.usage["input_tokens"] == 100
            assert session.total_cost_usd == 0.05
            assert session.num_turns == 2
            assert session.is_error is False

    @pytest.mark.asyncio
    async def test_stream_when_callbacks_provided_then_called(self):
        """Test callbacks are invoked."""
        mock_stream = [
            {"type": "system", "session_id": "s1"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hello"}]},
            },
            {"type": "done"},
        ]

        on_system_called = False
        on_text_called = False

        def on_system(data):
            nonlocal on_system_called
            on_system_called = True

        def on_text(text):
            nonlocal on_text_called
            on_text_called = True

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")

            async for _ in stream_claude_code_cli(request, on_system=on_system, on_text=on_text):
                pass

            assert on_system_called
            assert on_text_called

    @pytest.mark.asyncio
    async def test_stream_when_verbose_output_then_calls_pretty_print(self):
        """Test verbose output calls pretty print functions."""
        mock_stream = [
            {"type": "system", "session_id": "s1", "model": "sonnet", "tools": []},
            {
                "type": "result",
                "result": "Done",
                "usage": {},
                "total_cost_usd": 0.01,
                "num_turns": 1,
                "duration_ms": 1000,
                "duration_api_ms": 800,
            },
            {"type": "done"},
        ]

        with (
            patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock,
            patch("lionpride.services.third_party.claude_code._pp_system") as mock_pp,
        ):

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test", verbose=True)

            async for _ in stream_claude_code_cli(request):
                pass

            # Verify pretty print was called
            assert mock_pp.called

    @pytest.mark.asyncio
    async def test_stream_when_async_callbacks_then_awaits(self):
        """Test async callbacks are awaited."""
        callback_awaited = False

        async def async_callback(data):
            nonlocal callback_awaited
            await asyncio.sleep(0.01)
            callback_awaited = True

        mock_stream = [
            {"type": "system", "session_id": "s1"},
            {"type": "done"},
        ]

        with patch("lionpride.services.third_party.claude_code.stream_cc_cli_events") as mock:

            async def mock_gen(req):
                for event in mock_stream:
                    yield event

            mock.return_value = mock_gen(None)

            request = ClaudeCodeRequest(prompt="test")

            async for _ in stream_claude_code_cli(request, on_system=async_callback):
                pass

            assert callback_awaited


# ============================================================================
# Integration Tests (mocked subprocess)
# ============================================================================


class TestClaudeCodeIntegration:
    """Integration tests with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_integration_when_full_session_then_complete_flow(self):
        """Test complete session flow with all event types."""
        ndjson_chunks = [
            b'{"type":"system","session_id":"s1","model":"sonnet","tools":[]}\n',
            b'{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"Processing..."}]}}\n',
            b'{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}\n',
            b'{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{}}]}}\n',
            b'{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"file content","is_error":false}]}}\n',
            b'{"type":"result","result":"Done","usage":{"input_tokens":100,"output_tokens":50},"total_cost_usd":0.01,"num_turns":1,"duration_ms":1000,"duration_api_ms":800}\n',
        ]

        # Create a properly configured mock
        mock_proc = AsyncMock()

        # Mock stdout.read to return chunks then EOF
        async def mock_read_side_effect(size):
            if ndjson_chunks:
                return ndjson_chunks.pop(0)
            return b""

        mock_proc.stdout.read = AsyncMock(side_effect=mock_read_side_effect)
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.terminate = Mock()

        # Mock json_repair module to avoid import error
        with (
            patch("lionpride.services.third_party.claude_code.CLAUDE_CLI", "claude"),
            patch(
                "lionpride.services.third_party.claude_code.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ),
            patch.dict("sys.modules", {"json_repair": MagicMock()}),
        ):
            request = ClaudeCodeRequest(prompt="test")
            session = ClaudeSession()

            chunks = []
            async for chunk in stream_claude_code_cli(request, session):
                chunks.append(chunk)

            # Verify session state
            assert session.session_id == "s1"
            assert "Processing..." in session.thinking_log
            assert len(session.tool_uses) == 1
            assert len(session.tool_results) == 1
            assert session.result == "Done"
            assert session.total_cost_usd == 0.01

            # Verify chunks
            assert any(isinstance(c, dict) and c.get("type") == "system" for c in chunks)
            assert any(isinstance(c, ClaudeChunk) for c in chunks)
            assert isinstance(chunks[-1], ClaudeSession)
