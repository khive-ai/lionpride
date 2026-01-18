# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Session module with lazy loading for fast import."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # mail
    "Exchange": ("lionpride.session.mail", "Exchange"),
    "Mail": ("lionpride.session.mail", "Mail"),
    "OUTBOX": ("lionpride.session.mail", "OUTBOX"),
    # log_adapter
    "LogAdapter": ("lionpride.session.log_adapter", "LogAdapter"),
    "LogAdapterConfig": ("lionpride.session.log_adapter", "LogAdapterConfig"),
    "PostgresLogAdapter": ("lionpride.session.log_adapter", "PostgresLogAdapter"),
    "SQLiteWALLogAdapter": ("lionpride.session.log_adapter", "SQLiteWALLogAdapter"),
    # log_broadcaster
    "LogBroadcaster": ("lionpride.session.log_broadcaster", "LogBroadcaster"),
    "LogBroadcasterConfig": (
        "lionpride.session.log_broadcaster",
        "LogBroadcasterConfig",
    ),
    "LogSubscriber": ("lionpride.session.log_broadcaster", "LogSubscriber"),
    "PostgresLogSubscriber": (
        "lionpride.session.log_broadcaster",
        "PostgresLogSubscriber",
    ),
    "S3LogSubscriber": ("lionpride.session.log_broadcaster", "S3LogSubscriber"),
    "WebhookLogSubscriber": (
        "lionpride.session.log_broadcaster",
        "WebhookLogSubscriber",
    ),
    # logs
    "Log": ("lionpride.session.logs", "Log"),
    "LogStore": ("lionpride.session.logs", "LogStore"),
    "LogStoreConfig": ("lionpride.session.logs", "LogStoreConfig"),
    "LogType": ("lionpride.session.logs", "LogType"),
    # messages
    "ActionRequestContent": ("lionpride.session.messages", "ActionRequestContent"),
    "ActionResponseContent": ("lionpride.session.messages", "ActionResponseContent"),
    "AssistantResponseContent": (
        "lionpride.session.messages",
        "AssistantResponseContent",
    ),
    "InstructionContent": ("lionpride.session.messages", "InstructionContent"),
    "Message": ("lionpride.session.messages", "Message"),
    "MessageContent": ("lionpride.session.messages", "MessageContent"),
    "MessageRole": ("lionpride.session.messages", "MessageRole"),
    "SenderRecipient": ("lionpride.session.messages", "SenderRecipient"),
    "SystemContent": ("lionpride.session.messages", "SystemContent"),
    "prepare_messages_for_chat": (
        "lionpride.session.messages",
        "prepare_messages_for_chat",
    ),
    # session
    "Branch": ("lionpride.session.session", "Branch"),
    "Session": ("lionpride.session.session", "Session"),
}

_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazy import attributes on first access."""
    if name in _LOADED:
        return _LOADED[name]

    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value

    raise AttributeError(f"module 'lionpride.session' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from .log_adapter import (
        LogAdapter,
        LogAdapterConfig,
        PostgresLogAdapter,
        SQLiteWALLogAdapter,
    )
    from .log_broadcaster import (
        LogBroadcaster,
        LogBroadcasterConfig,
        LogSubscriber,
        PostgresLogSubscriber,
        S3LogSubscriber,
        WebhookLogSubscriber,
    )
    from .logs import Log, LogStore, LogStoreConfig, LogType
    from .mail import OUTBOX, Exchange, Mail
    from .messages import (
        ActionRequestContent,
        ActionResponseContent,
        AssistantResponseContent,
        InstructionContent,
        Message,
        MessageContent,
        MessageRole,
        SenderRecipient,
        SystemContent,
        prepare_messages_for_chat,
    )
    from .session import Branch, Session

__all__ = (
    "OUTBOX",
    "ActionRequestContent",
    "ActionResponseContent",
    "AssistantResponseContent",
    "Branch",
    "Exchange",
    "InstructionContent",
    "Log",
    "LogAdapter",
    "LogAdapterConfig",
    "LogBroadcaster",
    "LogBroadcasterConfig",
    "LogStore",
    "LogStoreConfig",
    "LogSubscriber",
    "LogType",
    "Mail",
    "Message",
    "MessageContent",
    "MessageRole",
    "PostgresLogAdapter",
    "PostgresLogSubscriber",
    "S3LogSubscriber",
    "SQLiteWALLogAdapter",
    "SenderRecipient",
    "Session",
    "SystemContent",
    "WebhookLogSubscriber",
    "prepare_messages_for_chat",
)
