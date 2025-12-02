# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Message and MessageContent types.

Focus areas:
    1. Content type inference priority when dict has multiple type indicators
    2. Empty/minimal content handling
    3. ActionResponseContent.success edge cases (None, empty string, both result+error)
    4. Message.clone() lineage tracking edge cases
    5. to_chat() for each content type with edge cases
    6. render() with structure_format='custom' vs 'json'
    7. Sender/recipient edge cases in Message context
    8. Serialization round-trip edge cases

These tests are designed to expose potential bugs in edge cases.

DISCOVERED ISSUES (these tests document current behavior):
    - BUG-001: Ambiguous dicts pass ALL keys to content create(), causing TypeError
    - BUG-002: ActionResponseContent.success treats empty string error as True (no error)
    - BUG-003: SystemContent with datetime_factory doesn't roundtrip (becomes InstructionContent)
    - BUG-004: URL validation doesn't catch null bytes in URLs
    - BUG-005: ActionRequestContent.render() omits function key when function is empty string
"""

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, Field

from lionpride.session.messages import Message
from lionpride.session.messages.base import (
    MessageRole,
    serialize_sender_recipient,
    validate_sender_recipient,
)
from lionpride.session.messages.content import (
    ActionRequestContent,
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    MessageContent,
    SystemContent,
)
from lionpride.types import Unset

# =============================================================================
# Test: Content Type Inference Priority (Ambiguous Dicts)
# =============================================================================


class TestContentTypeInferencePriority:
    """Test content type inference when dict has multiple type indicators.

    The current implementation checks keys in this order:
    1. instruction/context/request_model/tool_schemas/images -> InstructionContent
    2. assistant_response -> AssistantResponseContent
    3. result/error -> ActionResponseContent
    4. function/arguments -> ActionRequestContent
    5. system_message/system_datetime -> SystemContent
    6. Empty dict -> InstructionContent (default)

    KNOWN ISSUE (BUG-001): When dict has keys from multiple content types,
    ALL keys are passed to create(), causing TypeError.
    """

    @pytest.mark.xfail(
        reason="BUG-001: Ambiguous dict passes all keys to create(), "
        "causing TypeError for unexpected keyword arguments"
    )
    def test_when_dict_has_instruction_and_assistant_response_then_instruction_wins(
        self,
    ):
        """Dict with both instruction and assistant_response should become InstructionContent.

        This tests the priority: instruction fields are checked first.
        """
        ambiguous_dict = {
            "instruction": "Do something",
            "assistant_response": "I did something",  # This should be ignored
        }
        msg = Message(content=ambiguous_dict)

        assert isinstance(msg.content, InstructionContent)
        assert msg.role == MessageRole.USER
        assert msg.content.instruction == "Do something"

    @pytest.mark.xfail(
        reason="BUG-001: Ambiguous dict passes all keys to create(), "
        "causing TypeError for unexpected keyword arguments"
    )
    def test_when_dict_has_context_and_result_then_instruction_wins(self):
        """Dict with context (InstructionContent key) and result (ActionResponseContent key).

        Context is checked before result, so InstructionContent should win.
        """
        ambiguous_dict = {
            "context": ["some context"],
            "result": {"data": "value"},
        }
        msg = Message(content=ambiguous_dict)

        assert isinstance(msg.content, InstructionContent)
        assert msg.role == MessageRole.USER
        assert msg.content.context == ["some context"]

    @pytest.mark.xfail(
        reason="BUG-001: Ambiguous dict passes all keys to create(), "
        "causing TypeError for unexpected keyword arguments"
    )
    def test_when_dict_has_assistant_response_and_function_then_assistant_wins(self):
        """Dict with assistant_response and function should become AssistantResponseContent.

        assistant_response is checked before function.
        """
        ambiguous_dict = {
            "assistant_response": "response",
            "function": "some_function",
        }
        msg = Message(content=ambiguous_dict)

        assert isinstance(msg.content, AssistantResponseContent)
        assert msg.role == MessageRole.ASSISTANT

    @pytest.mark.xfail(
        reason="BUG-001: Ambiguous dict passes all keys to create(), "
        "causing TypeError for unexpected keyword arguments"
    )
    def test_when_dict_has_result_and_function_then_action_response_wins(self):
        """Dict with result and function should become ActionResponseContent.

        result/error are checked before function/arguments.
        """
        ambiguous_dict = {
            "result": {"data": "value"},
            "function": "some_function",
        }
        msg = Message(content=ambiguous_dict)

        assert isinstance(msg.content, ActionResponseContent)
        assert msg.role == MessageRole.TOOL

    @pytest.mark.xfail(
        reason="BUG-001: Ambiguous dict passes all keys to create(), "
        "causing TypeError for unexpected keyword arguments"
    )
    def test_when_dict_has_error_and_system_message_then_action_response_wins(self):
        """Dict with error and system_message should become ActionResponseContent.

        error is checked before system_message.
        """
        ambiguous_dict = {
            "error": "something failed",
            "system_message": "system info",
        }
        msg = Message(content=ambiguous_dict)

        assert isinstance(msg.content, ActionResponseContent)
        assert msg.role == MessageRole.TOOL
        assert msg.content.error == "something failed"

    def test_when_dict_has_only_images_then_instruction_content(self):
        """Dict with only 'images' key should become InstructionContent.

        images is one of the InstructionContent keys.
        """
        msg = Message(content={"images": ["https://example.com/img.png"]})

        assert isinstance(msg.content, InstructionContent)
        assert msg.role == MessageRole.USER
        assert msg.content.images == ["https://example.com/img.png"]

    def test_when_dict_has_only_request_model_then_instruction_content(self):
        """Dict with only 'request_model' key should become InstructionContent."""

        class OutputModel(BaseModel):
            answer: str

        msg = Message(content={"request_model": OutputModel})

        assert isinstance(msg.content, InstructionContent)
        assert msg.role == MessageRole.USER

    def test_when_dict_has_only_tool_schemas_then_instruction_content(self):
        """Dict with only 'tool_schemas' key should become InstructionContent."""
        msg = Message(content={"tool_schemas": ["interface Tool { x: string }"]})

        assert isinstance(msg.content, InstructionContent)
        assert msg.role == MessageRole.USER

    def test_when_dict_has_function_and_arguments_then_action_request(self):
        """Dict with function and arguments should become ActionRequestContent."""
        msg = Message(content={"function": "search", "arguments": {"q": "test"}})

        assert isinstance(msg.content, ActionRequestContent)
        assert msg.role == MessageRole.ASSISTANT

    def test_when_dict_has_only_arguments_then_action_request(self):
        """Dict with only 'arguments' key should become ActionRequestContent."""
        msg = Message(content={"arguments": {"param": "value"}})

        assert isinstance(msg.content, ActionRequestContent)
        assert msg.role == MessageRole.ASSISTANT

    def test_when_dict_has_only_system_datetime_then_system_content(self):
        """Dict with only 'system_datetime' key should become SystemContent."""
        msg = Message(content={"system_datetime": "2025-01-01T00:00:00Z"})

        assert isinstance(msg.content, SystemContent)
        assert msg.role == MessageRole.SYSTEM


# =============================================================================
# Test: Empty/Minimal Content Edge Cases
# =============================================================================


class TestEmptyMinimalContent:
    """Test edge cases with empty or minimal content."""

    def test_empty_dict_creates_instruction_content(self):
        """Empty dict should default to InstructionContent with all fields Unset."""
        msg = Message(content={})

        assert isinstance(msg.content, InstructionContent)
        assert msg.role == MessageRole.USER
        # All fields should be Unset
        assert InstructionContent._is_sentinel(msg.content.instruction)
        assert InstructionContent._is_sentinel(msg.content.context)

    def test_empty_dict_content_renders_empty_task(self):
        """Empty InstructionContent should render minimal output."""
        msg = Message(content={})
        rendered = msg.content.render()

        # Should not fail and should produce some output
        assert isinstance(rendered, str)

    def test_instruction_content_with_none_instruction(self):
        """InstructionContent.create(instruction=None) should set as sentinel."""
        content = InstructionContent.create(instruction=None)

        assert InstructionContent._is_sentinel(content.instruction)

    def test_assistant_response_content_with_empty_string(self):
        """AssistantResponseContent with empty string should render empty string."""
        content = AssistantResponseContent.create(assistant_response="")

        # Empty string is NOT a sentinel, it's a valid value
        assert content.assistant_response == ""
        assert content.render() == ""

    def test_system_content_all_none(self):
        """SystemContent with all None fields should render empty string."""
        content = SystemContent.create(
            system_message=None, system_datetime=None, datetime_factory=None
        )

        assert content.render() == ""


# =============================================================================
# Test: ActionResponseContent.success Edge Cases
# =============================================================================


class TestActionResponseContentSuccess:
    """Test ActionResponseContent.success property edge cases.

    success = True when error is Unset (sentinel)
    success = False when error is set (any value including empty string)

    KNOWN ISSUE (BUG-002): Empty string "" is treated as falsy, so success=True.
    Design expectation: empty string should mean "error present but empty message".
    """

    def test_success_when_only_result_then_true(self):
        """When only result is set, success should be True."""
        content = ActionResponseContent.create(result={"data": "value"})

        assert content.success is True

    def test_success_when_only_error_then_false(self):
        """When only error is set, success should be False."""
        content = ActionResponseContent.create(error="something failed")

        assert content.success is False

    @pytest.mark.xfail(
        reason="BUG-002: Empty string is NOT a sentinel but _is_sentinel uses config "
        "that may treat empty strings as sentinel. Current behavior: success=True"
    )
    def test_success_when_error_is_empty_string_then_false(self):
        """When error is empty string (not Unset), success should be False.

        Empty string is a valid error value, not a sentinel.
        This could be a bug if code assumes empty string = no error.
        """
        content = ActionResponseContent.create(error="")

        # Empty string is NOT Unset, so success should be False
        assert content.success is False

    def test_success_when_both_result_and_error_then_false(self):
        """When both result and error are set, success should be False.

        Error takes precedence - if error is set, it's a failure.
        """
        content = ActionResponseContent.create(result={"data": "value"}, error="partial failure")

        assert content.success is False

    def test_success_when_result_is_none_then_true(self):
        """When result is None (not Unset) and no error, success should be True.

        None is a valid result value.
        """
        content = ActionResponseContent.create(result=None)

        assert content.success is True

    def test_success_when_result_is_false_then_true(self):
        """When result is False (falsy but valid) and no error, success should be True."""
        content = ActionResponseContent.create(result=False)

        assert content.success is True

    def test_success_when_result_is_empty_dict_then_true(self):
        """When result is empty dict and no error, success should be True."""
        content = ActionResponseContent.create(result={})

        assert content.success is True

    def test_success_when_nothing_set_then_true(self):
        """When neither result nor error is set, success should be True.

        This represents an action that completed but returned no data.
        """
        content = ActionResponseContent.create()

        assert content.success is True

    def test_render_when_success_with_none_result(self):
        """Render should handle None result correctly."""
        content = ActionResponseContent.create(result=None)
        rendered = content.render()

        assert "success: true" in rendered or "success: True" in rendered

    @pytest.mark.xfail(
        reason="BUG-002: Empty string error is treated as sentinel, "
        "so success=True and render shows success: true"
    )
    def test_render_when_error_empty_string(self):
        """Render should show error even if empty string."""
        content = ActionResponseContent.create(error="")
        rendered = content.render()

        assert "success: false" in rendered or "success: False" in rendered
        assert "error:" in rendered


# =============================================================================
# Test: Message.clone() Lineage Edge Cases
# =============================================================================


class TestMessageCloneLineage:
    """Test Message.clone() lineage tracking edge cases."""

    def test_clone_deep_chain_tracks_immediate_parent(self):
        """Deep clone chain should track immediate parent, not original."""
        msg1 = Message(content={"instruction": "original"})
        msg2 = msg1.clone()
        msg3 = msg2.clone()
        msg4 = msg3.clone()

        # Each clone tracks its immediate parent
        assert msg4.metadata["clone_from"] == str(msg3.id)
        assert msg3.metadata["clone_from"] == str(msg2.id)
        assert msg2.metadata["clone_from"] == str(msg1.id)

    def test_clone_preserves_original_metadata_with_reserved_keys(self):
        """Clone preserves user metadata but overwrites reserved keys."""
        msg = Message(
            content={"instruction": "test"},
            metadata={
                "user_tag": "important",
                "clone_from": "should_be_overwritten",
                "original_created_at": "should_be_overwritten",
            },
        )

        cloned = msg.clone()

        # User metadata preserved
        assert cloned.metadata["user_tag"] == "important"
        # Reserved keys overwritten
        assert cloned.metadata["clone_from"] == str(msg.id)
        assert cloned.metadata["original_created_at"] == msg.created_at.isoformat()

    def test_clone_with_sender_override_preserves_recipient(self):
        """Clone with sender override should preserve recipient."""
        msg = Message(content={"instruction": "test"}, sender="alice", recipient="bob")

        cloned = msg.clone(sender="charlie")

        assert cloned.sender == "charlie"
        assert cloned.recipient == "bob"

    def test_clone_of_clone_preserves_content_type(self):
        """Clone chain should preserve content type throughout."""
        msg1 = Message(content={"system_message": "you are helpful"})
        msg2 = msg1.clone()
        msg3 = msg2.clone()

        assert isinstance(msg3.content, SystemContent)
        assert msg3.role == MessageRole.SYSTEM

    def test_clone_creates_unique_ids(self):
        """Multiple clones should have unique IDs."""
        msg = Message(content={"instruction": "test"})

        clones = [msg.clone() for _ in range(10)]
        clone_ids = [c.id for c in clones]

        # All IDs should be unique
        assert len(set(clone_ids)) == 10

    def test_clone_metadata_is_independent(self):
        """Modifying clone metadata should not affect original."""
        msg = Message(content={"instruction": "test"}, metadata={"mutable": {"key": "value"}})

        cloned = msg.clone()
        cloned.metadata["new_key"] = "new_value"

        assert "new_key" not in msg.metadata


# =============================================================================
# Test: to_chat() for Each Content Type
# =============================================================================


class TestToChatForAllContentTypes:
    """Test to_chat() format for each content type with edge cases."""

    def test_system_content_to_chat_with_datetime_true(self):
        """SystemContent with system_datetime=True should render timestamp."""
        content = SystemContent.create(system_datetime=True)
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "system"
        assert "System Time:" in chat["content"]

    def test_system_content_to_chat_with_factory(self):
        """SystemContent with datetime_factory should call factory."""
        content = SystemContent.create(datetime_factory=lambda: "CUSTOM_TIME")
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "system"
        assert "CUSTOM_TIME" in chat["content"]

    def test_instruction_content_to_chat_with_images(self):
        """InstructionContent with images should return list in content."""
        content = InstructionContent.create(
            instruction="analyze",
            images=["https://example.com/img.png"],
        )
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "user"
        # Content should be a list when images present
        assert isinstance(chat["content"], list)

    def test_instruction_content_to_chat_without_images(self):
        """InstructionContent without images should return string in content."""
        content = InstructionContent.create(instruction="do something")
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "user"
        assert isinstance(chat["content"], str)

    def test_assistant_response_to_chat_empty(self):
        """AssistantResponseContent with Unset response should render empty."""
        content = AssistantResponseContent.create()
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "assistant"
        assert chat["content"] == ""

    def test_action_request_to_chat(self):
        """ActionRequestContent should render as YAML."""
        content = ActionRequestContent.create(function="search", arguments={"query": "test"})
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "assistant"
        assert "function" in chat["content"]
        assert "arguments" in chat["content"]

    def test_action_response_to_chat_success(self):
        """ActionResponseContent success case should render correctly."""
        content = ActionResponseContent.create(request_id="req-123", result={"data": "value"})
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "tool"
        assert "success: true" in chat["content"] or "success: True" in chat["content"]

    def test_action_response_to_chat_error(self):
        """ActionResponseContent error case should render correctly."""
        content = ActionResponseContent.create(request_id="req-123", error="something failed")
        chat = content.to_chat()

        assert chat is not None
        assert chat["role"] == "tool"
        assert "success: false" in chat["content"] or "success: False" in chat["content"]


# =============================================================================
# Test: render() with structure_format
# =============================================================================


class TestRenderStructureFormat:
    """Test InstructionContent.render() with different structure_format values."""

    def test_render_json_format_includes_response_format(self):
        """render(structure_format='json') should include ResponseFormat section."""

        class OutputModel(BaseModel):
            answer: str = Field(..., description="The answer")
            confidence: float = Field(..., description="Confidence 0-1")

        content = InstructionContent.create(instruction="Answer", request_model=OutputModel)
        rendered = content.render(structure_format="json")

        assert "## ResponseFormat" in rendered
        assert "MUST RETURN VALID JSON" in rendered

    def test_render_custom_format_excludes_json_structure(self):
        """render(structure_format='custom') should NOT include JSON structure section."""

        class OutputModel(BaseModel):
            answer: str = Field(..., description="The answer")

        content = InstructionContent.create(instruction="Answer", request_model=OutputModel)
        rendered = content.render(structure_format="custom")

        # custom format should still include the schema but NOT the JSON structure
        # The _format_json_response_structure is only called for structure_format="json"
        assert "## ResponseFormat" not in rendered

    def test_render_default_is_json(self):
        """render() with no args should default to json format."""

        class OutputModel(BaseModel):
            answer: str

        content = InstructionContent.create(instruction="Answer", request_model=OutputModel)
        rendered_default = content.render()
        rendered_json = content.render(structure_format="json")

        assert rendered_default == rendered_json

    def test_render_without_request_model_ignores_format(self):
        """render() without request_model should work regardless of format."""
        content = InstructionContent.create(instruction="Simple instruction")

        rendered_json = content.render(structure_format="json")
        rendered_custom = content.render(structure_format="custom")

        # Both should work and produce similar output (no ResponseFormat section)
        assert "Instruction:" in rendered_json
        assert "Instruction:" in rendered_custom


# =============================================================================
# Test: Sender/Recipient Edge Cases in Message Context
# =============================================================================


class TestMessageSenderRecipientEdgeCases:
    """Test sender/recipient handling in Message context."""

    def test_message_with_uuid_sender(self):
        """Message with UUID sender should preserve UUID."""
        uid = uuid4()
        msg = Message(content={"instruction": "test"}, sender=uid)

        assert msg.sender == uid
        assert isinstance(msg.sender, UUID)

    def test_message_with_uuid_string_sender_converts(self):
        """Message with UUID string sender should convert to UUID."""
        uid_str = "12345678-1234-5678-1234-567812345678"
        msg = Message(content={"instruction": "test"}, sender=uid_str)

        assert isinstance(msg.sender, UUID)
        assert str(msg.sender) == uid_str

    def test_message_with_role_sender(self):
        """Message with MessageRole sender should preserve role."""
        msg = Message(content={"instruction": "test"}, sender=MessageRole.USER)

        assert msg.sender == MessageRole.USER

    def test_message_with_role_string_sender_converts(self):
        """Message with role string sender should convert to MessageRole."""
        msg = Message(content={"instruction": "test"}, sender="assistant")

        assert msg.sender == MessageRole.ASSISTANT

    def test_message_with_custom_string_sender(self):
        """Message with custom string sender should preserve string."""
        msg = Message(content={"instruction": "test"}, sender="agent_007")

        assert msg.sender == "agent_007"
        assert isinstance(msg.sender, str)

    def test_message_serialization_preserves_uuid_sender(self):
        """Serialization round-trip should preserve UUID sender."""
        uid = uuid4()
        msg1 = Message(content={"instruction": "test"}, sender=uid)

        data = msg1.to_dict()
        msg2 = Message.from_dict(data)

        assert msg2.sender == uid
        assert isinstance(msg2.sender, UUID)

    def test_message_serialization_preserves_role_sender(self):
        """Serialization round-trip should preserve role sender."""
        msg1 = Message(content={"instruction": "test"}, sender=MessageRole.ASSISTANT)

        data = msg1.to_dict()
        msg2 = Message.from_dict(data)

        assert msg2.sender == MessageRole.ASSISTANT


# =============================================================================
# Test: Serialization Round-Trip Edge Cases
# =============================================================================


class TestSerializationRoundTripEdgeCases:
    """Test serialization round-trip for edge cases."""

    @pytest.mark.xfail(
        reason="BUG-003: SystemContent with datetime_factory doesn't roundtrip. "
        "After serialization, the callable is lost and content becomes InstructionContent"
    )
    def test_roundtrip_system_content_with_datetime_factory(self):
        """SystemContent with datetime_factory cannot be fully serialized.

        datetime_factory is a callable, which may not serialize properly.
        This test documents expected behavior.
        """
        content = SystemContent.create(datetime_factory=lambda: "custom")
        msg = Message(content=content)

        data = msg.to_dict()
        # The datetime_factory should be lost or None after serialization
        msg2 = Message.from_dict(data)

        # Content type should be preserved
        assert isinstance(msg2.content, SystemContent)
        # But factory should be gone (serialization limitation)
        assert (
            SystemContent._is_sentinel(msg2.content.datetime_factory)
            or msg2.content.datetime_factory is None
        )

    def test_roundtrip_system_content_with_message_only(self):
        """SystemContent with only system_message should roundtrip correctly."""
        content = SystemContent.create(system_message="You are helpful")
        msg = Message(content=content)

        data = msg.to_dict()
        msg2 = Message.from_dict(data)

        assert isinstance(msg2.content, SystemContent)
        assert msg2.content.system_message == "You are helpful"

    def test_roundtrip_instruction_with_request_model(self):
        """InstructionContent with request_model may have serialization issues.

        request_model is a type, not an instance - may not serialize.
        """

        class OutputModel(BaseModel):
            answer: str

        content = InstructionContent.create(instruction="test", request_model=OutputModel)
        msg = Message(content=content)

        data = msg.to_dict()
        # After roundtrip, request_model may be lost
        msg2 = Message.from_dict(data)

        assert isinstance(msg2.content, InstructionContent)
        # instruction should be preserved
        assert msg2.content.instruction == "test"

    def test_roundtrip_preserves_all_action_response_fields(self):
        """ActionResponseContent should fully roundtrip."""
        content = ActionResponseContent.create(
            request_id="req-abc",
            result={"nested": {"data": [1, 2, 3]}},
        )
        msg = Message(content=content)

        data = msg.to_dict()
        msg2 = Message.from_dict(data)

        assert isinstance(msg2.content, ActionResponseContent)
        assert msg2.content.request_id == "req-abc"
        assert msg2.content.result == {"nested": {"data": [1, 2, 3]}}

    def test_roundtrip_preserves_metadata_with_nested_structures(self):
        """Message metadata with nested structures should roundtrip."""
        msg1 = Message(
            content={"instruction": "test"},
            metadata={
                "tags": ["a", "b", "c"],
                "config": {"nested": {"deep": {"value": 42}}},
            },
        )

        data = msg1.to_dict()
        msg2 = Message.from_dict(data)

        assert msg2.metadata == msg1.metadata

    def test_roundtrip_with_both_sender_and_recipient(self):
        """Message with both sender and recipient should roundtrip."""
        msg1 = Message(
            content={"instruction": "test"},
            sender="alice",
            recipient=MessageRole.ASSISTANT,
        )

        data = msg1.to_dict()
        msg2 = Message.from_dict(data)

        assert msg2.sender == "alice"
        assert msg2.recipient == MessageRole.ASSISTANT


# =============================================================================
# Test: SystemContent datetime_factory vs system_datetime Conflict
# =============================================================================


class TestSystemContentDatetimeConflict:
    """Test SystemContent datetime_factory vs system_datetime mutual exclusion."""

    def test_create_with_both_datetime_options_raises(self):
        """Setting both system_datetime and datetime_factory should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            SystemContent.create(
                system_datetime="2025-01-01T00:00:00Z",
                datetime_factory=lambda: "custom",
            )

    def test_create_with_datetime_true_and_factory_raises(self):
        """Setting system_datetime=True and datetime_factory should raise."""
        with pytest.raises(ValueError, match="Cannot set both"):
            SystemContent.create(system_datetime=True, datetime_factory=lambda: "custom")

    def test_from_dict_does_not_validate_conflict(self):
        """from_dict may not validate mutual exclusion - potential bug.

        This test documents current behavior. If from_dict uses create(),
        it should also raise. If it doesn't, this is a gap.
        """
        data = {
            "system_message": "test",
            "system_datetime": "2025-01-01T00:00:00Z",
            # datetime_factory can't be in dict (not serializable)
        }
        # This should work - no conflict
        content = SystemContent.from_dict(data)
        assert content.system_datetime == "2025-01-01T00:00:00Z"


# =============================================================================
# Test: InstructionContent Image URL Security Edge Cases
# =============================================================================


class TestInstructionContentImageUrlSecurity:
    """Additional security tests for image URL validation."""

    @pytest.mark.xfail(
        reason="BUG-004: URL validation doesn't catch null bytes. "
        "Null bytes can be used for path truncation attacks."
    )
    def test_reject_url_with_null_byte(self):
        """URL with null byte should be rejected."""
        # Null bytes can be used for path truncation attacks
        with pytest.raises(ValueError):
            InstructionContent.create(
                instruction="test",
                images=["https://example.com/image\x00.png"],
            )

    def test_reject_url_with_newline(self):
        """URL with newline should be handled safely."""
        # Newlines can cause header injection in some contexts
        # This may or may not be rejected depending on implementation
        try:
            content = InstructionContent.create(
                instruction="test",
                images=["https://example.com/ima\nge.png"],
            )
            # If it doesn't raise, the URL should be stored as-is
            assert "\n" in content.images[0]
        except ValueError:
            # Also acceptable - reject malformed URL
            pass

    def test_accept_url_with_unicode_domain(self):
        """URL with unicode domain (IDN) should be accepted or rejected consistently."""
        # Internationalized domain names
        try:
            content = InstructionContent.create(
                instruction="test",
                images=["https://xn--nxasmq5b.com/image.png"],  # Punycode
            )
            assert content.images == ["https://xn--nxasmq5b.com/image.png"]
        except ValueError:
            # Also acceptable
            pass

    def test_accept_url_with_port(self):
        """URL with port number should be accepted."""
        content = InstructionContent.create(
            instruction="test",
            images=["https://example.com:8080/image.png"],
        )
        assert content.images == ["https://example.com:8080/image.png"]

    def test_reject_url_with_backslash_path(self):
        """URL with backslash in path may be platform-dependent."""
        # Backslashes can be interpreted differently on Windows
        try:
            content = InstructionContent.create(
                instruction="test",
                images=["https://example.com/path\\image.png"],
            )
            # If accepted, just verify it's stored
            assert "\\" in content.images[0]
        except ValueError:
            pass


# =============================================================================
# Test: Message Content Update Behavior
# =============================================================================


class TestMessageContentUpdateBehavior:
    """Test that content cannot be mutated directly (immutability)."""

    def test_message_content_is_stored_reference(self):
        """Verify content is stored, not deep-copied."""
        original_content = InstructionContent.create(instruction="original")
        msg = Message(content=original_content)

        # Content should be the same object
        assert msg.content is original_content

    def test_message_content_changes_not_reflected_if_reconstructed(self):
        """Creating message from dict vs instance may differ."""
        content_dict = {"instruction": "test"}
        msg = Message(content=content_dict)

        # Modifying the original dict should NOT affect message
        content_dict["instruction"] = "modified"
        assert msg.content.instruction == "test"


# =============================================================================
# Test: ActionRequestContent Edge Cases
# =============================================================================


class TestActionRequestContentEdgeCases:
    """Test ActionRequestContent edge cases."""

    @pytest.mark.xfail(
        reason="BUG-005: ActionRequestContent.render() omits function key when "
        "function is empty string, because empty string is treated as sentinel"
    )
    def test_render_with_empty_function_name(self):
        """ActionRequestContent with empty function name should render."""
        content = ActionRequestContent.create(function="", arguments={"x": 1})
        rendered = content.render()

        assert isinstance(rendered, str)
        # Empty function name should still be included
        assert "function:" in rendered

    def test_render_with_valid_function_name(self):
        """ActionRequestContent with valid function name should render correctly."""
        content = ActionRequestContent.create(function="my_func", arguments={"x": 1})
        rendered = content.render()

        assert isinstance(rendered, str)
        assert "function:" in rendered
        assert "my_func" in rendered

    def test_render_with_nested_arguments(self):
        """ActionRequestContent with deeply nested arguments should render."""
        content = ActionRequestContent.create(
            function="complex",
            arguments={
                "level1": {
                    "level2": {
                        "level3": {"value": [1, 2, {"nested": True}]},
                    },
                },
            },
        )
        rendered = content.render()

        assert isinstance(rendered, str)
        assert "complex" in rendered

    def test_render_with_special_characters_in_arguments(self):
        """ActionRequestContent with special chars in arguments should render."""
        content = ActionRequestContent.create(
            function="special",
            arguments={
                "yaml_special": "colons: here",
                "quotes": 'single\' and "double"',
                "newlines": "line1\nline2",
            },
        )
        rendered = content.render()

        assert isinstance(rendered, str)


# =============================================================================
# Test: MessageContent Base Class Behavior
# =============================================================================


class TestMessageContentBaseClass:
    """Test MessageContent base class behavior."""

    def test_base_class_render_raises_not_implemented(self):
        """MessageContent.render() should raise NotImplementedError."""

        @dataclass
        class BareContent(MessageContent):
            pass

        content = BareContent()

        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            content.render()

    def test_base_class_from_dict_raises_not_implemented(self):
        """MessageContent.from_dict() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            MessageContent.from_dict({})

    def test_base_class_has_unset_role(self):
        """MessageContent base class should have role = UNSET."""
        assert MessageContent.role == MessageRole.UNSET


# =============================================================================
# Test: Message with Non-Dict Non-MessageContent Rejection
# =============================================================================


class TestMessageContentTypeRejection:
    """Test that Message rejects invalid content types."""

    def test_reject_string_content(self):
        """Message should reject string content."""
        with pytest.raises(TypeError, match="content must be MessageContent instance or dict"):
            Message(content="just a string")

    def test_reject_list_content(self):
        """Message should reject list content."""
        with pytest.raises(TypeError, match="content must be MessageContent instance or dict"):
            Message(content=["item1", "item2"])

    def test_reject_int_content(self):
        """Message should reject int content."""
        with pytest.raises(TypeError, match="content must be MessageContent instance or dict"):
            Message(content=42)

    def test_reject_none_content(self):
        """Message should reject None content."""
        with pytest.raises((TypeError, ValueError)):
            Message(content=None)

    def test_reject_basemodel_content(self):
        """Message should reject BaseModel instance (not dict, not MessageContent)."""

        class SomeModel(BaseModel):
            field: str = "value"

        with pytest.raises(TypeError, match="content must be MessageContent instance or dict"):
            Message(content=SomeModel())
