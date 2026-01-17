# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import JsonValue

from lionpride.core import Pile, Progression
from lionpride.types import not_sentinel

if TYPE_CHECKING:
    from pydantic import BaseModel

from .content import (
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    MessageContent,
    SystemContent,
)
from .message import Message

__all__ = ("prepare_messages_for_chat",)


def _get_text(content: MessageContent, attr: str) -> str:
    """Get text from content attr, returning '' if sentinel."""
    val = getattr(content, attr)
    return "" if content._is_sentinel(val) else val


def _build_context(content: InstructionContent, action_outputs: list[str]) -> list[JsonValue]:
    """Build context list by appending action outputs to existing context."""
    existing = content.context
    if content._is_sentinel(existing):
        return cast(list[JsonValue], list(action_outputs))
    return cast(list[JsonValue], list(cast(list[JsonValue], existing)) + action_outputs)


def prepare_messages_for_chat(
    messages: Pile[Message],
    progression: Progression | None = None,
    new_instruction: Message | None = None,
    to_chat: bool = False,
    structure_format: Literal["json", "custom"] = "json",
    custom_renderer: Callable[["BaseModel"], str] | None = None,
) -> list[MessageContent] | list[dict[str, Any]]:
    """Prepare messages for chat API with intelligent content organization.

    Algorithm:
    1. Auto-detect system message from first message (if SystemContent)
    2. Collect ActionResponseContent and embed into following instruction's context
    3. Merge consecutive AssistantResponses
    4. Embed system into first instruction
    5. Append new_instruction

    Args:
        messages: Pile of messages
        progression: Progression or list of UUIDs (None = all messages)
        new_instruction: New instruction to append
        to_chat: If True, return list[dict] chat format instead of MessageContent

    Returns:
        List of prepared MessageContent instances (immutable copies), or
        List of chat API dicts if to_chat=True
    """
    to_use: Pile[Message] = messages if progression is None else messages[progression]

    if len(to_use) == 0:
        if new_instruction:
            new_content: MessageContent = new_instruction.content.with_updates(
                copy_containers="deep"
            )
            if to_chat:
                chat_msg = new_content.to_chat(structure_format, custom_renderer)
                if not_sentinel(chat_msg, True, True):
                    return [cast(dict[str, Any], chat_msg)]
                return []
            return [new_content]
        return []

    # Phase 1: Extract system message (auto-detect from first message)
    system_text: str | None = None
    start_idx = 0

    first_msg = to_use[0]
    if isinstance(first_msg.content, SystemContent):
        system_text = first_msg.content.render()
        start_idx = 1

    # Phase 2: Process messages - collect action outputs for next instruction
    _use_msgs: list[MessageContent] = []
    pending_actions: list[str] = []

    for i, msg in enumerate(to_use):
        if i < start_idx:
            continue

        content = msg.content

        # ActionResponseContent: collect rendered output
        if isinstance(content, ActionResponseContent):
            pending_actions.append(content.render())
            continue

        # SystemContent in middle: skip
        if isinstance(content, SystemContent):
            continue

        # InstructionContent: embed pending action outputs
        if isinstance(content, InstructionContent):
            updates: dict[str, Any] = {"tool_schemas": None, "request_model": None}
            if pending_actions:
                updates["context"] = _build_context(content, pending_actions)
                pending_actions = []
            _use_msgs.append(content.with_updates(copy_containers="deep", **updates))
            continue

        # Other (AssistantResponse, ActionRequest): copy as-is
        _use_msgs.append(content.with_updates(copy_containers="deep"))

    # Phase 3: Merge consecutive AssistantResponses
    if len(_use_msgs) > 1:
        merged: list[MessageContent] = [_use_msgs[0]]
        for content in _use_msgs[1:]:
            if isinstance(content, AssistantResponseContent) and isinstance(
                merged[-1], AssistantResponseContent
            ):
                prev = _get_text(merged[-1], "assistant_response")
                curr = _get_text(content, "assistant_response")
                merged[-1] = AssistantResponseContent.create(assistant_response=f"{prev}\n\n{curr}")
            else:
                merged.append(content)
        _use_msgs = merged

    # Phase 4: Embed system message into first instruction
    if system_text:
        if len(_use_msgs) == 0 and new_instruction:
            # No history: embed into new_instruction
            if isinstance(new_instruction.content, InstructionContent):
                curr = _get_text(new_instruction.content, "instruction")
                system_updates: dict[str, Any] = {"instruction": f"{system_text}\n\n{curr}"}
                if pending_actions:
                    system_updates["context"] = _build_context(
                        new_instruction.content, pending_actions
                    )
                    pending_actions = []
                _use_msgs.append(
                    new_instruction.content.with_updates(copy_containers="deep", **system_updates)
                )
                new_instruction = None
        elif _use_msgs and isinstance(_use_msgs[0], InstructionContent):
            curr = _get_text(_use_msgs[0], "instruction")
            _use_msgs[0] = _use_msgs[0].with_updates(instruction=f"{system_text}\n\n{curr}")

    # Phase 5: Append new_instruction (with any remaining action outputs)
    if new_instruction:
        final_updates: dict[str, Any] = {}
        if pending_actions and isinstance(new_instruction.content, InstructionContent):
            final_updates["context"] = _build_context(new_instruction.content, pending_actions)
        _use_msgs.append(
            new_instruction.content.with_updates(copy_containers="deep", **final_updates)
        )

    if to_chat:
        result: list[dict[str, Any]] = []
        for m in _use_msgs:
            chat_msg = m.to_chat(structure_format, custom_renderer)
            if not_sentinel(chat_msg, True, True):
                result.append(cast(dict[str, Any], chat_msg))
        return result
    return _use_msgs
