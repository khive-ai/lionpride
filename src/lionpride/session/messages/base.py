# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias
from uuid import UUID

from lionpride.types import Enum

__all__ = (
    "MessageRole",
    "SenderRecipient",
    "serialize_sender_recipient",
    "validate_sender_recipient",
)


class MessageRole(Enum):
    """Roles for message sender/recipient in chat interactions."""

    SYSTEM = "system"
    """System/Developer instructions defining model behavior"""

    USER = "user"
    """Direct message sender to the assistant"""

    ASSISTANT = "assistant"
    """Assistant response (model-generated)"""

    TOOL = "tool"
    """Tool result returned after tool_call execution"""

    UNSET = "unset"
    """No role specified (fallback/unknown)"""


SenderRecipient: TypeAlias = MessageRole | str | UUID
"""Sender/recipient: role, string, or UUID."""


def validate_sender_recipient(value) -> SenderRecipient:
    """Normalize sender/recipient."""
    if isinstance(value, MessageRole):
        return value
    if isinstance(value, UUID):
        return value
    if value is None:
        return MessageRole.UNSET
    if value in MessageRole.allowed():
        return MessageRole(value)
    if isinstance(value, str):
        try:
            return UUID(value)
        except (ValueError, AttributeError):
            return value
    raise ValueError(f"Invalid sender or recipient: {value}")


def serialize_sender_recipient(value: SenderRecipient) -> str | None:
    """Serialize for storage."""
    if not value:
        return None
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, MessageRole):
        return value.value
    if isinstance(value, str):
        return value
    return str(value)
