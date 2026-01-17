# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Mail: A message between communicatable entities.

Mail is the basic unit of communication in the exchange system.
It carries content from a sender to a recipient, optionally within a channel.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from lionpride.core import Element

__all__ = ("Mail",)


class Mail(Element):
    """A message that can be sent between communicatable entities.

    Mail provides a simple envelope for content routing:
    - sender: who sent it
    - recipient: who should receive it (None = broadcast)
    - content: the actual payload
    - channel: optional namespace for grouping/filtering

    The content can be any serializable value - the receiver decides
    how to interpret it. Mail doesn't impose structure on content.

    Attributes:
        sender: UUID of the sending entity
        recipient: UUID of the receiving entity (None for broadcast)
        content: The message payload (any serializable value)
        channel: Optional namespace for message grouping
    """

    sender: UUID = Field(
        description="UUID of the sending entity",
    )
    recipient: UUID | None = Field(
        default=None,
        description="UUID of receiving entity (None = broadcast)",
    )
    content: Any = Field(
        default=None,
        description="The message payload",
    )
    channel: str | None = Field(
        default=None,
        description="Optional namespace for message grouping",
    )

    @field_validator("sender", mode="before")
    @classmethod
    def _coerce_sender(cls, v) -> UUID:
        return cls._coerce_id(v)

    @field_validator("recipient", mode="before")
    @classmethod
    def _coerce_recipient(cls, v: Any) -> UUID | None:
        """Coerce recipient to UUID or None."""
        if v is None:
            return None
        return cls._coerce_id(v)

    @property
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message (no specific recipient)."""
        return self.recipient is None

    @property
    def is_direct(self) -> bool:
        """Check if this is a direct message (has specific recipient)."""
        return self.recipient is not None

    def __repr__(self) -> str:
        target = str(self.recipient)[:8] if self.recipient else "broadcast"
        channel_str = f", channel={self.channel}" if self.channel else ""
        return f"Mail({str(self.sender)[:8]} -> {target}{channel_str})"
