# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gemini CLI integration security warnings (Issue #92)."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from lionpride.services.third_party.gemini_models import GeminiCodeRequest

# ============================================================================
# GeminiCodeRequest Security Warnings Tests (Issue #92)
# ============================================================================


class TestGeminiSecurityWarnings:
    """Test security warnings for dangerous Gemini CLI settings."""

    def test_yolo_true_emits_warning(self):
        """Test that yolo=True emits a security warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCodeRequest(prompt="test", yolo=True)

            # Should have at least one warning
            assert len(w) >= 1
            # Find the yolo warning
            yolo_warnings = [x for x in w if "yolo" in str(x.message).lower()]
            assert len(yolo_warnings) == 1
            assert issubclass(yolo_warnings[0].category, UserWarning)
            assert "auto-approval" in str(yolo_warnings[0].message).lower()

    def test_sandbox_false_emits_warning(self):
        """Test that sandbox=False emits a security warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCodeRequest(prompt="test", sandbox=False)

            # Should have at least one warning
            assert len(w) >= 1
            # Find the sandbox warning
            sandbox_warnings = [x for x in w if "sandbox" in str(x.message).lower()]
            assert len(sandbox_warnings) == 1
            assert issubclass(sandbox_warnings[0].category, UserWarning)
            assert "unrestricted" in str(sandbox_warnings[0].message).lower()

    def test_both_dangerous_settings_emit_both_warnings(self):
        """Test that both yolo=True and sandbox=False emit their respective warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCodeRequest(prompt="test", yolo=True, sandbox=False)

            # Should have at least two warnings
            assert len(w) >= 2
            # Find both warnings
            yolo_warnings = [x for x in w if "yolo" in str(x.message).lower()]
            sandbox_warnings = [x for x in w if "sandbox" in str(x.message).lower()]
            assert len(yolo_warnings) == 1
            assert len(sandbox_warnings) == 1

    def test_default_settings_no_warnings(self):
        """Test that default settings (yolo=False, sandbox=True) emit no warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCodeRequest(prompt="test")

            # Should have no security warnings (yolo or sandbox related)
            security_warnings = [
                x
                for x in w
                if "yolo" in str(x.message).lower() or "sandbox" in str(x.message).lower()
            ]
            assert len(security_warnings) == 0

    def test_yolo_false_no_yolo_warning(self):
        """Test that yolo=False (explicit) does not emit yolo warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCodeRequest(prompt="test", yolo=False)

            yolo_warnings = [x for x in w if "yolo" in str(x.message).lower()]
            assert len(yolo_warnings) == 0

    def test_sandbox_true_no_sandbox_warning(self):
        """Test that sandbox=True (explicit) does not emit sandbox warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCodeRequest(prompt="test", sandbox=True)

            sandbox_warnings = [x for x in w if "sandbox" in str(x.message).lower()]
            assert len(sandbox_warnings) == 0

    def test_warning_message_mentions_security_risk(self):
        """Test that warning messages mention security risk."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeminiCodeRequest(prompt="test", yolo=True, sandbox=False)

            for warning in w:
                msg = str(warning.message).lower()
                if "yolo" in msg or "sandbox" in msg:
                    assert (
                        "security" in msg
                        or "safety" in msg
                        or "risk" in msg
                        or "danger" in msg
                        or "unrestricted" in msg
                    )


# ============================================================================
# GeminiCodeRequest Model Tests
# ============================================================================


class TestGeminiCodeRequest:
    """Test GeminiCodeRequest model validation and methods."""

    def test_request_minimal_valid_prompt(self):
        """Test minimal request with only prompt."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test prompt")
            assert request.prompt == "test prompt"
            assert request.model == "gemini-2.5-pro"
            assert request.yolo is False
            assert request.sandbox is True

    def test_request_with_all_fields(self):
        """Test request with all optional fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(
                prompt="test",
                system_prompt="system",
                model="gemini-2.5-flash",
                yolo=True,
                sandbox=False,
                debug=True,
                mcp_tools=["tool1"],
            )
            assert request.prompt == "test"
            assert request.system_prompt == "system"
            assert request.model == "gemini-2.5-flash"
            assert request.yolo is True
            assert request.sandbox is False

    def test_request_missing_prompt_and_messages_raises(self):
        """Test missing prompt and messages raises error."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="messages or prompt required"):
                GeminiCodeRequest(messages=[])

    def test_cwd_with_no_workspace_returns_repo(self):
        """Test cwd returns repo when no workspace."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", repo=Path("/test/repo"))
            assert request.cwd() == Path("/test/repo")

    def test_cwd_with_relative_workspace(self):
        """Test cwd with relative workspace."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", repo=Path("/test/repo"), ws="sub")
            cwd = request.cwd()
            assert cwd.parts[-1] == "sub"

    def test_cwd_with_absolute_workspace_raises(self):
        """Test absolute workspace path raises error."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", repo=Path("/test/repo"), ws="/absolute/path")
            with pytest.raises(ValueError, match="must be relative"):
                request.cwd()

    def test_cwd_with_directory_traversal_raises(self):
        """Test directory traversal prevention."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", repo=Path("/test/repo"), ws="../escape")
            with pytest.raises(ValueError, match="Directory traversal detected"):
                request.cwd()

    def test_as_cmd_args_minimal(self):
        """Test command args generation with minimal config."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test prompt")
            args = request.as_cmd_args()
            assert "-p" in args
            assert "test prompt" in args
            assert "--output-format" in args
            assert "stream-json" in args
            assert "-m" in args
            assert "gemini-2.5-pro" in args

    def test_as_cmd_args_with_yolo(self):
        """Test yolo flag in command args."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", yolo=True)
            args = request.as_cmd_args()
            assert "--yolo" in args

    def test_as_cmd_args_with_no_sandbox(self):
        """Test no-sandbox flag in command args."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", sandbox=False)
            args = request.as_cmd_args()
            assert "--no-sandbox" in args

    def test_as_cmd_args_with_debug(self):
        """Test debug flag in command args."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", debug=True)
            args = request.as_cmd_args()
            assert "--debug" in args

    def test_as_cmd_args_with_approval_mode(self):
        """Test approval mode in command args."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(prompt="test", approval_mode="full_auto")
            args = request.as_cmd_args()
            assert "--approval-mode" in args
            assert "full_auto" in args


# ============================================================================
# Message Conversion Tests
# ============================================================================


class TestGeminiMessageConversion:
    """Test message to prompt conversion."""

    def test_messages_converted_to_prompt(self):
        """Test message conversion to prompt."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "how are you"},
                ]
            )
            assert "hello" in request.prompt
            assert "hi" in request.prompt
            assert "how are you" in request.prompt

    def test_system_message_extracted(self):
        """Test system message extraction."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(
                messages=[
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "hello"},
                ]
            )
            assert request.system_prompt == "You are helpful"
            assert "hello" in request.prompt
            assert "You are helpful" not in request.prompt

    def test_dict_content_serialized(self):
        """Test dictionary content serialization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            request = GeminiCodeRequest(
                messages=[{"role": "user", "content": {"type": "text", "text": "hello"}}]
            )
            assert "text" in request.prompt or "hello" in request.prompt
