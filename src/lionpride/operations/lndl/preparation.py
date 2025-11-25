# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL message preparation for chat API calls.

Functions for preparing messages with LNDL system prompts injected,
handling system message merging and spec format injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.session.messages import Message, SystemContent
from lionpride.session.messages.utils import prepare_messages_for_chat

from .formatting import generate_lndl_spec_format

if TYPE_CHECKING:
    from lionpride.session import Branch, Session
    from lionpride.types import Operable

    from ..operate.operative import Operative


def prepare_lndl_messages(
    session: Session,
    branch: Branch,
    ins_msg: Message,
    operable: Operable | Operative,
) -> list[dict[str, Any]]:
    """Prepare messages with LNDL system prompt injection.

    Creates a message list suitable for chat API, with LNDL system
    prompt injected at the beginning. Merges with existing system
    message if present.

    Args:
        session: Session containing messages
        branch: Branch with conversation history
        ins_msg: New instruction message
        operable: Operable or Operative with specs

    Returns:
        List of chat-formatted messages with LNDL system prompt
    """
    from lionpride.lndl import get_lndl_system_prompt

    # Get base LNDL prompt
    lndl_prompt = get_lndl_system_prompt()

    # Add spec-specific format guidance
    spec_format = generate_lndl_spec_format(operable)
    if spec_format:
        lndl_prompt = f"{lndl_prompt}\n\n{spec_format}"

    # Create LNDL system message (merged with existing if present)
    sender_str = str(ins_msg.sender) if ins_msg.sender is not None else "system"
    lndl_system_msg = create_lndl_system_message(
        lndl_prompt,
        session,
        branch,
        sender_str,
    )

    # Get branch messages and prepare for chat
    branch_msgs = session.messages[branch]
    messages = prepare_messages_for_chat(
        messages=branch_msgs,
        progression=branch,
        new_instruction=ins_msg,
        to_chat=True,
    )

    # Insert LNDL system message at the beginning
    result: list[dict[str, Any]] = []
    if lndl_system_msg.chat_msg is not None:
        result.append(lndl_system_msg.chat_msg)
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
    return result


def create_lndl_system_message(
    lndl_prompt: str,
    session: Session,
    branch: Branch,
    recipient: str,
) -> Message:
    """Create LNDL system message, merging with existing if present.

    If the branch already has a system message, the LNDL prompt is
    appended to it. Otherwise, creates a new system message.

    Args:
        lndl_prompt: LNDL system prompt text
        session: Session containing messages
        branch: Branch to check for existing system message
        recipient: Message recipient

    Returns:
        System message with LNDL prompt
    """
    system_msg = session.get_branch_system(branch)

    if system_msg:
        existing_message = (
            system_msg.content.system_message
            if hasattr(system_msg.content, "system_message")
            else str(system_msg.content)
        )
        content = SystemContent(system_message=f"{existing_message}\n\n{lndl_prompt}")
    else:
        content = SystemContent(system_message=lndl_prompt)

    return Message(
        content=content,
        sender="system",
        recipient=recipient,
    )


__all__ = (
    "create_lndl_system_message",
    "prepare_lndl_messages",
)
