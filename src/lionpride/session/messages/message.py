# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any

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
        """Infer and construct MessageContent from dict."""
        if isinstance(v, MessageContent):
            return v

        if not isinstance(v, dict):
            raise TypeError(
                f"content must be MessageContent instance or dict, got {type(v).__name__}"
            )

        # Infer content type from dict keys
        if any(
            k in v for k in ("instruction", "context", "request_model", "tool_schemas", "images")
        ):
            return InstructionContent.create(**v)
        if "assistant_response" in v:
            return AssistantResponseContent.create(**v)
        if "result" in v or "error" in v:
            return ActionResponseContent.create(**v)
        if "function" in v or "arguments" in v:
            return ActionRequestContent.create(**v)
        if "system_message" in v or "system_datetime" in v:
            return SystemContent.create(**v)

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
        return self.from_dict(current)
