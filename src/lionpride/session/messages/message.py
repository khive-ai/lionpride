# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from pydantic import field_serializer, field_validator

from lionpride.core import Node

from .base import (
    MessageRole,
    SenderRecipient,
    serialize_sender_recipient,
    validate_sender_recipient,
)
from .content import (
    ActionRequestContent,
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    MessageContent,
    SystemContent,
)

__all__ = ("Message",)

# Valid keys for each content type's create() method
# Used to filter ambiguous dicts and prevent TypeError on unexpected kwargs
_INSTRUCTION_KEYS = frozenset(
    {
        "instruction",
        "context",
        "tool_schemas",
        "request_model",
        "images",
        "image_detail",
    }
)
_ASSISTANT_RESPONSE_KEYS = frozenset({"assistant_response"})
_ACTION_RESPONSE_KEYS = frozenset({"request_id", "result", "error"})
_ACTION_REQUEST_KEYS = frozenset({"function", "arguments"})
_SYSTEM_KEYS = frozenset({"system_message", "system_datetime", "datetime_factory"})


class Message(Node):
    """Message container with auto-derived role from content type.

    Attributes:
        content: MessageContent variant (auto-inferred from dict keys)
        sender: Optional sender identifier
        recipient: Optional recipient identifier
        role: Auto-derived from content.role (read-only)
        chat_msg: Chat API format {"role": "...", "content": "..."}
    """

    content: MessageContent
    sender: SenderRecipient | None = None
    recipient: SenderRecipient | None = None

    @property
    def role(self) -> MessageRole:
        """Auto-derive role from content type via ClassVar."""
        return self.content.role

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, v: Any) -> MessageContent:
        """Infer and construct MessageContent from dict.

        When a dict has keys from multiple content types (ambiguous dict),
        only the keys recognized by the inferred content type are passed
        to create(). Extra keys are silently ignored to prevent TypeError.
        """
        if isinstance(v, MessageContent):
            return v

        if not isinstance(v, dict):
            raise TypeError(
                f"content must be MessageContent instance or dict, got {type(v).__name__}"
            )

        # Infer content type from dict keys and filter to valid keys only
        # This prevents TypeError when dict has keys from multiple content types
        if any(
            k in v
            for k in (
                "instruction",
                "context",
                "request_model",
                "tool_schemas",
                "images",
            )
        ):
            filtered = {k: v[k] for k in v if k in _INSTRUCTION_KEYS}
            return InstructionContent.create(**filtered)
        if "assistant_response" in v:
            filtered = {k: v[k] for k in v if k in _ASSISTANT_RESPONSE_KEYS}
            return AssistantResponseContent.create(**filtered)
        if "result" in v or "error" in v:
            filtered = {k: v[k] for k in v if k in _ACTION_RESPONSE_KEYS}
            return ActionResponseContent.create(**filtered)
        if "function" in v or "arguments" in v:
            filtered = {k: v[k] for k in v if k in _ACTION_REQUEST_KEYS}
            return ActionRequestContent.create(**filtered)
        if "system_message" in v or "system_datetime" in v:
            filtered = {k: v[k] for k in v if k in _SYSTEM_KEYS}
            return SystemContent.create(**filtered)

        # Default to InstructionContent for empty dict
        return InstructionContent.create()

    @field_serializer("sender", "recipient")
    def _serialize_sender_recipient(self, value: SenderRecipient) -> str | None:
        return serialize_sender_recipient(value)

    @field_validator("sender", "recipient", mode="before")
    @classmethod
    def _validate_sender_recipient(cls, v):
        return validate_sender_recipient(v) if v is not None else None

    def clone(self, *, sender: SenderRecipient | None = None) -> "Message":
        """Create copy with new ID and lineage tracking in metadata."""
        current = self.to_dict(exclude={"id", "created_at"})
        metadata = current.get("metadata", {})
        metadata["clone_from"] = str(self.id)
        metadata["original_created_at"] = self.created_at.isoformat()
        current["metadata"] = metadata
        if sender is not None:
            current["sender"] = sender
        return cast("Message", self.from_dict(current))
