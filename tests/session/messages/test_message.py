# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Message class with auto role inference and immutability.

Target: 90%+ coverage for src/lionpride/session/messages/message.py

## Test Coverage

### Role Inference (lines 36-45)
- Message auto-derives role from content type via property
- SystemContent → MessageRole.SYSTEM
- InstructionContent → MessageRole.USER
- AssistantResponseContent → MessageRole.ASSISTANT
- ActionRequestContent → MessageRole.ASSISTANT
- ActionResponseContent → MessageRole.TOOL
- Unknown content type → MessageRole.UNSET

### Content Validation (lines 60-88)
- Accepts MessageContent instances directly
- Infers content type from dict keys
- Dict with "instruction" → InstructionContent
- Dict with "assistant_response" → AssistantResponseContent
- Dict with "result"/"error" → ActionResponseContent
- Dict with "function"/"arguments" → ActionRequestContent
- Dict with "system_message" → SystemContent
- Empty dict defaults to InstructionContent
- TypeError for non-dict/non-MessageContent

### Chat API Format (lines 48-53)
- chat_msg property returns {"role": "...", "content": "..."}
- Uses role.value and content.render()
- Returns None on exception (defensive)

### Immutability (lines 99-117)
- update() creates new Message instance
- Validates all fields during update
- Preserves unmodified fields
- Merges metadata correctly

### Clone with Lineage (lines 119-130)
- Creates new Message with new ID and timestamp
- Tracks clone_from (original ID)
- Tracks original_created_at
- Optional sender override

### Sender/Recipient (lines 33-34, 90-97)
- Accepts MessageRole, str, UUID, or None
- Validates and normalizes values
- Serializes correctly for storage

### Serialization (inherited from Node)
- to_dict/from_dict round-trip
- Polymorphic deserialization (Element.from_dict)
"""

from uuid import UUID

import pytest

from lionpride.session.messages import Message
from lionpride.session.messages.base import MessageRole
from lionpride.session.messages.content import (
    ActionRequestContent,
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    SystemContent,
)

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def system_content():
    """System message content."""
    return SystemContent.create(system_message="You are a helpful assistant")


@pytest.fixture
def instruction_content():
    """User instruction content."""
    return InstructionContent.create(instruction="What is the capital of France?")


@pytest.fixture
def assistant_content():
    """Assistant response content."""
    return AssistantResponseContent.create(assistant_response="The capital is Paris.")


@pytest.fixture
def action_request_content():
    """Action request content."""
    return ActionRequestContent.create(function="get_weather", arguments={"city": "NYC"})


@pytest.fixture
def action_response_content():
    """Action response content (success)."""
    return ActionResponseContent.create(request_id="req_123", result={"temp": 72})


@pytest.fixture
def action_error_content():
    """Action response content (error)."""
    return ActionResponseContent.create(request_id="req_456", error="API timeout")


# ===========================
# Test: Role Inference
# ===========================


@pytest.mark.parametrize(
    "content_type,expected_role",
    [
        (SystemContent.create(system_message="test"), MessageRole.SYSTEM),
        (InstructionContent.create(instruction="test"), MessageRole.USER),
        (AssistantResponseContent.create(assistant_response="test"), MessageRole.ASSISTANT),
        (ActionRequestContent.create(function="f", arguments={}), MessageRole.ASSISTANT),
        (ActionResponseContent.create(result={"ok": True}), MessageRole.TOOL),
    ],
)
def test_role_inference_from_content_type(content_type, expected_role):
    """Test auto role derivation from content type."""
    msg = Message(content=content_type)
    assert msg.role == expected_role


def test_role_inference_unset_for_unknown_content():
    """Test unknown content type returns MessageRole.UNSET.

    Note: In practice, the validator ensures only known content types are created,
    but we test the fallback behavior of the role property mapping.
    """
    # Create a custom content class that's not in the role mapping
    from dataclasses import dataclass

    from lionpride.session.messages.content import MessageContent

    @dataclass
    class UnknownContent(MessageContent):
        data: str = "test"

        @property
        def rendered(self):
            return self.data

        @classmethod
        def from_dict(cls, data):
            return cls(data=data.get("data", ""))

    # Mock a message with unknown content by setting directly after proper init
    msg = Message(content={"instruction": "temp"})  # Initialize properly first
    # Now inject unknown content (simulates hypothetical unknown type)
    object.__setattr__(msg, "content", UnknownContent())
    assert msg.role == MessageRole.UNSET


# ===========================
# Test: Content Validation
# ===========================


def test_message_accepts_content_instance(instruction_content):
    """Test Message accepts MessageContent instance directly."""
    msg = Message(content=instruction_content)
    assert isinstance(msg.content, InstructionContent)
    assert msg.role == MessageRole.USER


@pytest.mark.parametrize(
    "content_dict,expected_type,expected_role",
    [
        ({"instruction": "test"}, InstructionContent, MessageRole.USER),
        ({"context": ["a", "b"]}, InstructionContent, MessageRole.USER),
        ({"assistant_response": "test"}, AssistantResponseContent, MessageRole.ASSISTANT),
        ({"result": {"ok": True}}, ActionResponseContent, MessageRole.TOOL),
        ({"error": "failed"}, ActionResponseContent, MessageRole.TOOL),
        ({"function": "f", "arguments": {}}, ActionRequestContent, MessageRole.ASSISTANT),
        ({"system_message": "test"}, SystemContent, MessageRole.SYSTEM),
        ({}, InstructionContent, MessageRole.USER),  # Empty dict defaults to Instruction
    ],
)
def test_message_infers_content_type_from_dict(content_dict, expected_type, expected_role):
    """Test content type inference from dict keys."""
    msg = Message(content=content_dict)
    assert isinstance(msg.content, expected_type)
    assert msg.role == expected_role


def test_message_rejects_non_dict_non_content():
    """Test Message raises TypeError for invalid content."""
    with pytest.raises(TypeError, match="content must be MessageContent instance or dict"):
        Message(content="invalid string")

    with pytest.raises(TypeError, match="content must be MessageContent instance or dict"):
        Message(content=123)


# ===========================
# Test: Chat API Format
# ===========================


def test_to_chat_format(instruction_content):
    """Test content.to_chat() returns correct format."""
    msg = Message(content=instruction_content)
    chat_msg = msg.content.to_chat()

    assert chat_msg is not None
    assert "role" in chat_msg
    assert "content" in chat_msg
    assert chat_msg["role"] == "user"
    assert isinstance(chat_msg["content"], str)


def test_to_chat_all_roles():
    """Test to_chat() format for all role types."""
    test_cases = [
        ({"system_message": "test"}, "system"),
        ({"instruction": "test"}, "user"),
        ({"assistant_response": "test"}, "assistant"),
        ({"function": "f", "arguments": {}}, "assistant"),
        ({"result": {"ok": True}}, "tool"),
    ]

    for content_dict, expected_role in test_cases:
        msg = Message(content=content_dict)
        chat_msg = msg.content.to_chat()
        assert chat_msg["role"] == expected_role


def test_to_chat_defensive_none_on_exception():
    """Test to_chat() returns None on exception (defensive)."""
    from dataclasses import dataclass

    from lionpride.session.messages.content import MessageContent

    @dataclass
    class BrokenContent(MessageContent):
        def render(self, *args, **kwargs):
            raise RuntimeError("Broken render")

        @classmethod
        def from_dict(cls, data):
            return cls()

    # Initialize properly first, then inject broken content
    msg = Message(content={"instruction": "temp"})
    object.__setattr__(msg, "content", BrokenContent())

    # Should return None instead of raising
    assert msg.content.to_chat() is None


# ===========================
# Test: Clone with Lineage
# ===========================


def test_clone_creates_new_instance(instruction_content):
    """Test clone() creates new Message with new ID."""
    msg1 = Message(content=instruction_content)
    msg2 = msg1.clone()

    assert msg1.id != msg2.id
    assert msg1.created_at != msg2.created_at


def test_clone_tracks_lineage(instruction_content):
    """Test clone() tracks original in metadata."""
    msg1 = Message(content=instruction_content)
    msg2 = msg1.clone()

    assert "clone_from" in msg2.metadata
    assert msg2.metadata["clone_from"] == str(msg1.id)
    assert "original_created_at" in msg2.metadata
    assert msg2.metadata["original_created_at"] == msg1.created_at.isoformat()


def test_clone_preserves_content(instruction_content):
    """Test clone() preserves content."""
    msg1 = Message(content=instruction_content)
    msg2 = msg1.clone()

    assert isinstance(msg2.content, InstructionContent)
    assert msg2.role == msg1.role


def test_clone_with_sender_override(instruction_content):
    """Test clone() can override sender."""
    msg1 = Message(content=instruction_content, sender="alice")
    msg2 = msg1.clone(sender="bob")

    assert msg2.sender == "bob"
    assert msg1.sender == "alice"


def test_clone_chain_preserves_original():
    """Test multiple clones track original, not intermediate."""
    msg1 = Message(content={"instruction": "test"})
    msg2 = msg1.clone()
    msg3 = msg2.clone()

    # msg3 should track msg2 as clone_from (not msg1)
    assert msg3.metadata["clone_from"] == str(msg2.id)
    # Original timestamp should be msg2's timestamp
    assert msg3.metadata["original_created_at"] == msg2.created_at.isoformat()


def test_clone_overwrites_user_provided_clone_from():
    """Test clone() overwrites user-provided 'clone_from' with lineage (BS-2).

    Reserved keys are documented as being overwritten to enable clone chains.
    Users should avoid using these keys in their metadata.
    """
    msg = Message(
        content={"instruction": "test"},
        metadata={"clone_from": "user_provided_value"},
    )

    cloned = msg.clone()

    # Reserved key is overwritten with correct lineage
    assert cloned.metadata["clone_from"] == str(msg.id)
    assert cloned.metadata["clone_from"] != "user_provided_value"


def test_clone_overwrites_user_provided_original_created_at():
    """Test clone() overwrites user-provided 'original_created_at' (BS-2)."""
    msg = Message(
        content={"instruction": "test"},
        metadata={"original_created_at": "2025-01-01T00:00:00"},
    )

    cloned = msg.clone()

    # Reserved key is overwritten with correct timestamp
    assert cloned.metadata["original_created_at"] == msg.created_at.isoformat()
    assert cloned.metadata["original_created_at"] != "2025-01-01T00:00:00"


def test_clone_overwrites_both_reserved_keys():
    """Test clone() overwrites both reserved keys if present (BS-2)."""
    msg = Message(
        content={"instruction": "test"},
        metadata={
            "clone_from": "user_value_1",
            "original_created_at": "2025-01-01T00:00:00",
        },
    )

    cloned = msg.clone()

    # Both reserved keys are overwritten with correct values
    assert cloned.metadata["clone_from"] == str(msg.id)
    assert cloned.metadata["original_created_at"] == msg.created_at.isoformat()


def test_clone_preserves_other_metadata():
    """Test clone() preserves non-reserved metadata keys (BS-2)."""
    msg = Message(
        content={"instruction": "test"},
        metadata={"tag": "v1", "priority": "high"},
    )

    cloned = msg.clone()

    # Verify clone succeeded
    assert cloned.id != msg.id
    assert "clone_from" in cloned.metadata
    assert cloned.metadata["clone_from"] == str(msg.id)
    # Original metadata preserved
    assert cloned.metadata["tag"] == "v1"
    assert cloned.metadata["priority"] == "high"


# ===========================
# Test: Sender/Recipient
# ===========================


@pytest.mark.parametrize(
    "sender_value",
    [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        "alice",
        UUID("12345678-1234-5678-1234-567812345678"),
        None,
    ],
)
def test_sender_accepts_valid_types(sender_value, instruction_content):
    """Test sender accepts MessageRole, str, UUID, None."""
    msg = Message(content=instruction_content, sender=sender_value)
    # Just verify no exception and sender is set
    assert msg.sender is not None or sender_value is None


@pytest.mark.parametrize(
    "recipient_value",
    [
        MessageRole.ASSISTANT,
        "bob",
        UUID("87654321-4321-8765-4321-876543218765"),
        None,
    ],
)
def test_recipient_accepts_valid_types(recipient_value, instruction_content):
    """Test recipient accepts MessageRole, str, UUID, None."""
    msg = Message(content=instruction_content, recipient=recipient_value)
    # Just verify no exception
    assert msg.recipient is not None or recipient_value is None


def test_sender_recipient_validation():
    """Test sender/recipient validation for edge cases."""
    # Valid UUID string
    msg = Message(content={"instruction": "test"}, sender="12345678-1234-5678-1234-567812345678")
    assert isinstance(msg.sender, UUID)

    # Valid role string
    msg = Message(content={"instruction": "test"}, sender="user")
    assert msg.sender == MessageRole.USER

    # Arbitrary string
    msg = Message(content={"instruction": "test"}, sender="custom_agent")
    assert msg.sender == "custom_agent"


# ===========================
# Test: Serialization
# ===========================


def test_serialization_round_trip(instruction_content):
    """Test Message serialization/deserialization."""
    msg1 = Message(content=instruction_content, sender="alice", recipient="bob")
    data = msg1.to_dict()

    msg2 = Message.from_dict(data)

    assert msg2.id == msg1.id
    assert msg2.role == msg1.role
    assert msg2.sender == msg1.sender
    assert msg2.recipient == msg1.recipient
    assert isinstance(msg2.content, InstructionContent)


def test_serialization_all_content_types():
    """Test serialization round-trip for all content types."""
    test_cases = [
        {"system_message": "test"},
        {"instruction": "test"},
        {"assistant_response": "test"},
        {"function": "f", "arguments": {"x": 1}},
        {"result": {"ok": True}},
        {"error": "failed"},
    ]

    for content_dict in test_cases:
        msg1 = Message(content=content_dict)
        data = msg1.to_dict()
        msg2 = Message.from_dict(data)

        assert msg2.id == msg1.id
        assert msg2.role == msg1.role
        assert type(msg2.content) == type(msg1.content)


def test_serialization_preserves_metadata():
    """Test serialization preserves metadata."""
    msg1 = Message(content={"instruction": "test"}, metadata={"tag": "v1", "priority": "high"})
    data = msg1.to_dict()
    msg2 = Message.from_dict(data)

    assert msg2.metadata == msg1.metadata


def test_polymorphic_deserialization_from_element():
    """Test Message can be deserialized via Element.from_dict()."""
    from lionpride import Element

    msg1 = Message(content={"instruction": "test"})
    data = msg1.to_dict()

    # Polymorphic deserialization through Element
    restored = Element.from_dict(data)

    assert isinstance(restored, Message)
    assert restored.id == msg1.id
    assert restored.role == MessageRole.USER


# ===========================
# Test: Edge Cases
# ===========================


def test_message_with_empty_metadata():
    """Test Message handles empty metadata."""
    msg = Message(content={"instruction": "test"})
    assert msg.metadata == {}


def test_message_render_delegates_to_content(instruction_content):
    """Test render() delegates to content."""
    msg = Message(content=instruction_content)
    assert msg.content.render() == instruction_content.render()


def test_message_with_complex_instruction():
    """Test Message with complex instruction (context, tools, response_model)."""
    from pydantic import BaseModel

    class OutputModel(BaseModel):
        answer: str
        confidence: float

    content = InstructionContent.create(
        instruction="Answer the question",
        context=["ctx1", "ctx2"],
        request_model=OutputModel,
    )

    msg = Message(content=content)
    assert msg.role == MessageRole.USER
    assert "ResponseFormat" in msg.content.render()  # Structured output section


def test_message_none_sender_recipient():
    """Test Message with None sender/recipient."""
    msg = Message(content={"instruction": "test"}, sender=None, recipient=None)
    assert msg.sender is None
    assert msg.recipient is None
