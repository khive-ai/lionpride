# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for operations/operate/types.py coverage.

Coverage targets:
- GenerateParams.instruction_message property edge cases
- Line 94: Return None when instruction is sentinel
"""

import pytest

from lionpride.operations.operate.types import GenerateParams
from lionpride.session.messages import Message


class TestGenerateParamsInstructionMessage:
    """Test GenerateParams.instruction_message property."""

    def test_instruction_message_with_string_instruction(self):
        """Test instruction_message creates Message from string."""
        params = GenerateParams(instruction="test instruction")

        msg = params.instruction_message

        assert msg is not None
        assert isinstance(msg, Message)
        assert msg.content.instruction == "test instruction"

    def test_instruction_message_with_message_instruction(self):
        """Test instruction_message returns Message directly when given Message.

        Coverage: types.py lines 96-97 (isinstance check branch)
        """
        from lionpride.session.messages import InstructionContent

        # Create Message with instruction content
        content = InstructionContent.create(instruction="direct message")
        original_msg = Message(content=content)

        params = GenerateParams(instruction=original_msg)

        msg = params.instruction_message

        # Should return the same Message object
        assert msg is original_msg

    def test_instruction_is_required(self):
        """Test that instruction is a required parameter.

        GenerateParams.instruction is required - no default value.
        """
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'instruction'"
        ):
            GenerateParams()

    def test_instruction_message_with_context(self):
        """Test instruction_message includes context in Message."""
        params = GenerateParams(
            instruction="test instruction",
            context={"key": "value", "number": 42},
        )

        msg = params.instruction_message

        assert msg is not None
        assert msg.content.context == {"key": "value", "number": 42}

    def test_instruction_message_with_images(self):
        """Test instruction_message includes images in Message."""
        params = GenerateParams(
            instruction="describe this image",
            images=["https://example.com/image.png"],
        )

        msg = params.instruction_message

        assert msg is not None
        assert msg.content.images == ["https://example.com/image.png"]

    def test_instruction_message_with_tool_schemas(self):
        """Test instruction_message includes tool_schemas in Message."""
        tool_schema = {"name": "search", "description": "Search the web"}

        params = GenerateParams(
            instruction="use the search tool",
            tool_schemas=[tool_schema],
        )

        msg = params.instruction_message

        assert msg is not None
        assert msg.content.tool_schemas == [tool_schema]
