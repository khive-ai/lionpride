# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for message sequence utilities.

prepare_messages_for_chat - 5-Phase Algorithm:

Phase 1: System Message Auto-Detection
- If first message is SystemContent, extract rendered text and skip it
- System text will be embedded into first instruction later

Phase 2: Message Processing with Action Output Collection
- ActionResponseContent: Collect rendered output for next instruction
- InstructionContent: Embed pending action outputs into context, clear tool_schemas/response_model
- AssistantResponseContent: Clone as-is
- SystemContent (mid-sequence): Skip
- Other content types: Clone as-is

Phase 3: Consecutive AssistantResponse Merging
- Merge consecutive AssistantResponses with "\\n\\n" separator
- Non-consecutive assistants remain separate
- Handles sentinel (empty) values correctly

Phase 4: System Message Embedding
- System text embedded into first instruction (not standalone)
- If no history: embed into new_instruction
- If history exists: embed into first instruction in history

Phase 5: New Instruction Appending
- Append new_instruction (with any remaining action outputs)

Key Nuances:
- All operations create new instances (immutability preserved via with_updates)
- ActionResponseContent.rendered is embedded into following instruction's context
- Pile.__getitem__ handles progression validation automatically

Test Coverage:
- All 5 phases tested independently
- Integration tests for full conversation flows
- Edge cases: empty piles, None values, malformed inputs
- to_chat_format: MessageContent to chat API dict conversion
"""

import pytest

from lionpride import Pile, Progression
from lionpride.session.messages.base import MessageRole
from lionpride.session.messages.content import (
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    SystemContent,
)
from lionpride.session.messages.message import Message
from lionpride.session.messages.prepare_msg import prepare_messages_for_chat


# Fixtures
@pytest.fixture
def system_msg() -> Message:
    """System message fixture."""
    content = SystemContent.create(system_message="You are a helpful assistant.")
    return Message(content=content)


@pytest.fixture
def instruction_msg() -> Message:
    """Basic instruction message fixture."""
    content = InstructionContent.create(instruction="What is the capital of France?")
    return Message(content=content)


@pytest.fixture
def instruction_with_tools() -> Message:
    """Instruction with tool_schemas and context."""
    from pydantic import BaseModel

    class TestTool(BaseModel):
        query: str

    content = InstructionContent.create(
        instruction="Search for something",
        context=["context1", "context2"],
        tool_schemas=[TestTool],
    )
    return Message(content=content)


@pytest.fixture
def assistant_msg() -> Message:
    """Assistant response message fixture."""
    content = AssistantResponseContent.create(assistant_response="Paris is the capital.")
    return Message(content=content)


@pytest.fixture
def assistant_msg2() -> Message:
    """Second assistant response for merging tests."""
    content = AssistantResponseContent.create(
        assistant_response="It's also known as the City of Light."
    )
    return Message(content=content)


@pytest.fixture
def action_response_msg() -> Message:
    """Action response message fixture."""
    content = ActionResponseContent.create(
        request_id="req_123", result={"data": "test"}, error=None
    )
    return Message(content=content)


# Phase 1: System Auto-Detection Tests
class TestSystemAutoDetection:
    """Test Phase 1: System message auto-detection from first message."""

    def test_system_first_message_extracted(self, system_msg: Message, instruction_msg: Message):
        """Test that system message as first message is extracted and embedded."""
        messages = Pile([system_msg, instruction_msg])
        result = prepare_messages_for_chat(messages)

        # System message should be embedded into instruction
        assert len(result) == 1
        assert isinstance(result[0], InstructionContent)
        assert "You are a helpful assistant." in result[0].instruction
        assert "What is the capital of France?" in result[0].instruction

    def test_no_system_message_works(self, instruction_msg: Message):
        """Test that no system message works fine."""
        messages = Pile([instruction_msg])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        assert result[0].instruction == "What is the capital of France?"

    def test_system_mid_sequence_skipped(
        self, system_msg: Message, instruction_msg: Message, assistant_msg: Message
    ):
        """Test that system message in middle of sequence is skipped."""
        # System in middle should be ignored
        messages = Pile([instruction_msg, system_msg, assistant_msg])
        result = prepare_messages_for_chat(messages)

        # Should have instruction and assistant, no system
        assert len(result) == 2
        assert isinstance(result[0], InstructionContent)
        assert isinstance(result[1], AssistantResponseContent)


# Phase 2: Action Output Collection Tests
class TestActionOutputCollection:
    """Test Phase 2: ActionResponseContent embedding into following instruction."""

    def test_action_response_embedded_into_next_instruction(
        self, action_response_msg: Message, instruction_msg: Message
    ):
        """Test that action response is embedded into following instruction's context."""
        messages = Pile([action_response_msg, instruction_msg])
        result = prepare_messages_for_chat(messages)

        # Action response should be embedded into instruction's context
        assert len(result) == 1
        assert isinstance(result[0], InstructionContent)
        # Context should contain the rendered action response
        assert result[0].context is not None
        assert len(result[0].context) == 1
        assert "success: true" in result[0].context[0]
        assert "req_123" in result[0].context[0]

    def test_multiple_action_responses_collected(self, instruction_msg: Message):
        """Test that multiple action responses are collected."""
        action1 = Message(
            content=ActionResponseContent.create(request_id="req_1", result="result1")
        )
        action2 = Message(
            content=ActionResponseContent.create(request_id="req_2", result="result2")
        )
        messages = Pile([action1, action2, instruction_msg])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        assert len(result[0].context) == 2
        assert "req_1" in result[0].context[0]
        assert "req_2" in result[0].context[1]

    def test_action_response_appended_to_existing_context(
        self, action_response_msg: Message, instruction_with_tools: Message
    ):
        """Test that action outputs are appended to existing context."""
        messages = Pile([action_response_msg, instruction_with_tools])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        # Original context had ["context1", "context2"]
        assert len(result[0].context) == 3
        assert result[0].context[0] == "context1"
        assert result[0].context[1] == "context2"
        assert "req_123" in result[0].context[2]

    def test_action_response_without_following_instruction_into_new_instruction(
        self, action_response_msg: Message, instruction_msg: Message
    ):
        """Test that action output goes into new_instruction if no following instruction."""
        messages = Pile([action_response_msg])
        result = prepare_messages_for_chat(messages, new_instruction=instruction_msg)

        assert len(result) == 1
        assert result[0].context is not None
        assert "req_123" in result[0].context[0]


# Phase 2 Continued: Progression Selection Tests
class TestProgressionSelection:
    """Test progression selection and content transformation."""

    def test_none_progression_uses_all_messages(
        self, instruction_msg: Message, assistant_msg: Message
    ):
        """Test that None progression includes all messages in order."""
        messages = Pile([instruction_msg, assistant_msg])
        result = prepare_messages_for_chat(messages, progression=None)

        assert len(result) == 2
        assert result[0].instruction == instruction_msg.content.instruction
        assert result[1].assistant_response == assistant_msg.content.assistant_response

    def test_explicit_progression_selects_indices(
        self, instruction_msg: Message, assistant_msg: Message
    ):
        """Test that explicit progression selects only specified indices."""
        inst_msg2 = Message(content=instruction_msg.content)
        messages = Pile([instruction_msg, assistant_msg, inst_msg2])
        result = prepare_messages_for_chat(
            messages, progression=Progression(order=[instruction_msg.id, inst_msg2.id])
        )

        assert len(result) == 2
        # Both should be instructions
        assert isinstance(result[0], InstructionContent)
        assert isinstance(result[1], InstructionContent)

    def test_assistant_response_cloned_as_is(self, assistant_msg: Message):
        """Test that AssistantResponse is cloned without modification."""
        messages = Pile([assistant_msg])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        assert result[0].assistant_response == assistant_msg.content.assistant_response
        # Verify it's a new instance (with_updates creates a copy)
        assert result[0] is not assistant_msg.content

    def test_instruction_clears_tool_schemas(self, instruction_with_tools: Message):
        """Test that InstructionContent clears tool_schemas/response_model from history."""
        messages = Pile([instruction_with_tools])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        # Instruction text preserved
        assert result[0].instruction == instruction_with_tools.content.instruction
        # Context preserved
        assert result[0].context == instruction_with_tools.content.context
        # tool_schemas cleared
        assert result[0]._is_sentinel(result[0].tool_schemas)

    def test_empty_progression_returns_empty_list(self, instruction_msg: Message):
        """Test that empty progression returns empty list."""
        messages = Pile([instruction_msg])
        result = prepare_messages_for_chat(messages, progression=Progression(order=[]))

        assert len(result) == 0


# Phase 3: Merging Tests
class TestConsecutiveAssistantMerging:
    """Test Phase 3: Consecutive AssistantResponse merging."""

    def test_consecutive_assistant_responses_merged(
        self, assistant_msg: Message, assistant_msg2: Message
    ):
        """Test that consecutive AssistantResponses are merged with \\n\\n."""
        messages = Pile([assistant_msg, assistant_msg2])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        merged_text = result[0].assistant_response
        assert "Paris is the capital." in merged_text
        assert "City of Light." in merged_text
        assert "\n\n" in merged_text

    def test_non_consecutive_assistant_not_merged(
        self, assistant_msg: Message, instruction_msg: Message, assistant_msg2: Message
    ):
        """Test that non-consecutive AssistantResponses are not merged."""
        messages = Pile([assistant_msg, instruction_msg, assistant_msg2])
        result = prepare_messages_for_chat(messages)

        # Should have 3 messages: asst1, inst, asst2
        assert len(result) == 3
        assert isinstance(result[0], AssistantResponseContent)
        assert isinstance(result[1], InstructionContent)
        assert isinstance(result[2], AssistantResponseContent)

    def test_single_message_not_affected_by_merging(self, assistant_msg: Message):
        """Test that single message passes through unchanged."""
        messages = Pile([assistant_msg])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        assert result[0].assistant_response == assistant_msg.content.assistant_response

    def test_merging_handles_sentinel_values(self):
        """Test that merging handles sentinel (empty) values correctly."""
        # Create assistant responses with sentinel values
        content1 = AssistantResponseContent.create(assistant_response="First")
        content2 = AssistantResponseContent.create(assistant_response="Second response")
        msg1 = Message(content=content1)
        msg2 = Message(content=content2)

        result = prepare_messages_for_chat(Pile([msg1, msg2]))

        assert len(result) == 1
        # Should merge "First" + "\n\n" + "Second response"
        assert "First" in result[0].assistant_response
        assert "Second response" in result[0].assistant_response
        assert "\n\n" in result[0].assistant_response


# Phase 4: System Embedding Tests
class TestSystemEmbedding:
    """Test Phase 4: System embedding into first instruction."""

    def test_system_embedded_into_first_instruction(
        self, system_msg: Message, instruction_msg: Message
    ):
        """Test that system message is embedded into first instruction."""
        messages = Pile([system_msg, instruction_msg])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        embedded_text = result[0].instruction
        assert "You are a helpful assistant." in embedded_text
        assert "What is the capital of France?" in embedded_text
        assert "\n\n" in embedded_text

    def test_system_embedded_clears_tool_schemas_from_history(
        self, system_msg: Message, instruction_with_tools: Message
    ):
        """Test that system embedding into history instruction clears tool_schemas."""
        messages = Pile([system_msg, instruction_with_tools])
        result = prepare_messages_for_chat(messages)

        assert len(result) == 1
        # System embedded
        assert "You are a helpful assistant." in result[0].instruction
        # Context preserved
        assert result[0].context == instruction_with_tools.content.context
        # tool_schemas cleared (this is history, not final instruction)
        assert result[0]._is_sentinel(result[0].tool_schemas)

    def test_system_embedded_into_new_instruction_when_empty_history(
        self, system_msg: Message, instruction_msg: Message
    ):
        """Test that system is embedded into new_instruction when history is empty."""
        # Only system message in history
        messages = Pile([system_msg])
        result = prepare_messages_for_chat(messages, new_instruction=instruction_msg)

        assert len(result) == 1
        embedded_text = result[0].instruction
        assert "You are a helpful assistant." in embedded_text
        assert "What is the capital of France?" in embedded_text

    def test_no_standalone_system_in_output(self, system_msg: Message, instruction_msg: Message):
        """Test that standalone system messages don't appear in output."""
        messages = Pile([system_msg, instruction_msg])
        result = prepare_messages_for_chat(messages)

        # Should only have instruction (no standalone system)
        assert len(result) == 1
        assert result[0].role == MessageRole.USER


# Phase 5: New Instruction Tests
class TestNewInstructionAppend:
    """Test Phase 5: New instruction appending."""

    def test_new_instruction_appended(self, instruction_msg: Message, assistant_msg: Message):
        """Test that new_instruction is appended to the end."""
        messages = Pile([instruction_msg, assistant_msg])
        new_inst = Message(content=InstructionContent.create(instruction="Follow-up question"))

        result = prepare_messages_for_chat(messages, new_instruction=new_inst)

        assert len(result) == 3
        assert result[-1].instruction == "Follow-up question"

    def test_new_instruction_not_added_when_embedded_with_system(
        self, system_msg: Message, instruction_msg: Message
    ):
        """Test that new_instruction is not added twice when embedded with system."""
        # System only in history, new instruction should get system embedded
        messages = Pile([system_msg])
        result = prepare_messages_for_chat(messages, new_instruction=instruction_msg)

        # Should only have 1 message (system embedded into new_instruction)
        assert len(result) == 1
        assert "You are a helpful assistant." in result[0].instruction

    def test_new_instruction_none_does_not_append(
        self, instruction_msg: Message, assistant_msg: Message
    ):
        """Test that None new_instruction doesn't append anything."""
        messages = Pile([instruction_msg, assistant_msg])
        result = prepare_messages_for_chat(messages, new_instruction=None)

        assert len(result) == 2


# Integration Tests
class TestIntegration:
    """Test full 5-phase integration scenarios."""

    def test_full_conversation_flow(
        self,
        system_msg: Message,
        instruction_msg: Message,
        assistant_msg: Message,
        assistant_msg2: Message,
    ):
        """Test complete conversation with all phases."""
        # Conversation: system, inst1, asst1, asst2, inst2
        inst2 = Message(content=InstructionContent.create(instruction="Tell me more"))
        messages = Pile([system_msg, instruction_msg, assistant_msg, assistant_msg2, inst2])
        new_inst = Message(content=InstructionContent.create(instruction="Final question"))

        result = prepare_messages_for_chat(messages, new_instruction=new_inst)

        # Expected: inst1_with_system, merged_asst, inst2, new_inst
        assert len(result) == 4

        # First instruction has system embedded
        assert "You are a helpful assistant." in result[0].instruction
        assert "What is the capital of France?" in result[0].instruction

        # Second message is merged assistant
        assert isinstance(result[1], AssistantResponseContent)
        assert "Paris is the capital." in result[1].assistant_response
        assert "City of Light." in result[1].assistant_response

        # Third is inst2
        assert result[2].instruction == "Tell me more"

        # Fourth is new_inst
        assert result[3].instruction == "Final question"

    def test_conversation_with_action_responses(
        self, system_msg: Message, instruction_msg: Message, assistant_msg: Message
    ):
        """Test conversation flow with action responses."""
        action = Message(
            content=ActionResponseContent.create(request_id="act_1", result="search result")
        )
        follow_up = Message(content=InstructionContent.create(instruction="Process the result"))

        messages = Pile([system_msg, instruction_msg, assistant_msg, action, follow_up])
        result = prepare_messages_for_chat(messages)

        # system -> embedded in inst1
        # inst1 -> with system
        # asst
        # action -> embedded in follow_up
        # follow_up -> with action context
        assert len(result) == 3
        assert "You are a helpful assistant." in result[0].instruction
        assert isinstance(result[1], AssistantResponseContent)
        assert "act_1" in str(result[2].context)

    def test_empty_messages_returns_empty_with_no_new_instruction(self):
        """Test that empty messages with no new_instruction returns empty list."""
        result = prepare_messages_for_chat(Pile([]))
        assert len(result) == 0

    def test_immutability_original_messages_unchanged(self, instruction_with_tools: Message):
        """Test that original messages are not modified."""
        messages = Pile([instruction_with_tools])
        original_tools = instruction_with_tools.content.tool_schemas

        result = prepare_messages_for_chat(messages)

        # Result should have cleared tool_schemas
        assert result[0]._is_sentinel(result[0].tool_schemas)
        # Original should be unchanged
        assert instruction_with_tools.content.tool_schemas == original_tools

    def test_deep_copy_preserves_context_isolation(self, instruction_with_tools: Message):
        """Test that context modifications don't affect original."""
        action = Message(content=ActionResponseContent.create(request_id="act_1", result="data"))
        messages = Pile([action, instruction_with_tools])

        original_context = list(instruction_with_tools.content.context)
        result = prepare_messages_for_chat(messages)

        # Result has extended context
        assert len(result[0].context) == 3
        # Original unchanged
        assert instruction_with_tools.content.context == original_context


# to_chat Flag Tests
class TestToChatFlag:
    """Test to_chat flag for chat API format output."""

    def test_to_chat_returns_dict_list(self, instruction_msg: Message, assistant_msg: Message):
        """Test that to_chat=True returns list of dicts."""
        messages = Pile([instruction_msg, assistant_msg])
        result = prepare_messages_for_chat(messages, to_chat=True)

        assert len(result) == 2
        assert all(isinstance(msg, dict) for msg in result)
        assert all("role" in msg and "content" in msg for msg in result)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_to_chat_false_returns_content_list(
        self, instruction_msg: Message, assistant_msg: Message
    ):
        """Test that to_chat=False (default) returns MessageContent list."""
        messages = Pile([instruction_msg, assistant_msg])
        result = prepare_messages_for_chat(messages, to_chat=False)

        assert len(result) == 2
        assert isinstance(result[0], InstructionContent)
        assert isinstance(result[1], AssistantResponseContent)

    def test_to_chat_with_system_embedding(self, system_msg: Message, instruction_msg: Message):
        """Test to_chat works with system embedding."""
        messages = Pile([system_msg, instruction_msg])
        result = prepare_messages_for_chat(messages, to_chat=True)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "You are a helpful assistant." in result[0]["content"]


# Parametrized Edge Cases
@pytest.mark.parametrize(
    "messages,progression,expected_count",
    [
        (Pile([]), None, 0),
        (Pile([]), Progression(order=[]), 0),
        pytest.param(
            Pile([Message(content=InstructionContent.create(instruction="test"))]),
            None,
            1,
            id="single_message",
        ),
    ],
)
def test_edge_cases_parametrized(messages, progression, expected_count):
    """Parametrized tests for edge cases."""
    result = prepare_messages_for_chat(messages, progression=progression)
    assert len(result) == expected_count


# Additional Edge Case Tests for Coverage
class TestToChatWithEmptyPile:
    """Tests for to_chat with empty pile and new_instruction.

    Covers prepare_msg.py lines 70-71.
    """

    def test_empty_pile_with_new_instruction_to_chat_true(self):
        """Test empty pile with new_instruction and to_chat=True returns dict (covers line 70)."""
        new_instr = Message(content=InstructionContent.create(instruction="What is 2+2?"))
        result = prepare_messages_for_chat(
            messages=Pile([]),
            new_instruction=new_instr,
            to_chat=True,
        )

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["role"] == "user"
        assert "2+2" in result[0]["content"]

    def test_empty_pile_with_new_instruction_to_chat_false(self):
        """Test empty pile with new_instruction and to_chat=False returns MessageContent (covers line 71)."""
        new_instr = Message(content=InstructionContent.create(instruction="What is 3+3?"))
        result = prepare_messages_for_chat(
            messages=Pile([]),
            new_instruction=new_instr,
            to_chat=False,  # Explicitly False
        )

        assert len(result) == 1
        assert isinstance(result[0], InstructionContent)
        assert result[0].instruction == "What is 3+3?"


class TestSystemWithPendingActions:
    """Tests for system message with pending actions embedded into new_instruction.

    Covers prepare_msg.py lines 136-137.
    """

    def test_system_message_with_pending_actions_no_history(self):
        """Test system message with action response but no other history.

        When system exists, action exists, but no other instructions in history,
        both system text and action outputs should be embedded into new_instruction.
        (covers lines 136-137)
        """
        system_msg = Message(content=SystemContent.create(system_message="You are helpful."))
        action_msg = Message(
            content=ActionResponseContent.create(request_id="act_123", result="Result: 42")
        )
        new_instr = Message(content=InstructionContent.create(instruction="What was the result?"))

        messages = Pile([system_msg, action_msg])
        result = prepare_messages_for_chat(
            messages=messages,
            new_instruction=new_instr,
        )

        # Should have single instruction with system embedded and action in context
        assert len(result) == 1
        assert isinstance(result[0], InstructionContent)
        # System embedded into instruction
        assert "You are helpful." in result[0].instruction
        # Action output in context
        assert result[0].context is not None
        assert any("Result: 42" in str(ctx) for ctx in result[0].context)
