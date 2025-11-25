# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Streaming module for lionpride operations.

Provides streaming abstractions for LLM response handling:
- StreamChannel: Buffered streaming with fan-out
- Consumers: Real-time chunk processing (text, LNDL)
- OutputSink: Streaming output destinations
"""

from .channel import (
    SourcedStreamChannel,
    StreamChannel,
    StreamChunk,
)
from .consumer import (
    LNDLConsumer,
    PrintConsumer,
    StreamConsumer,
    TextConsumer,
)
from .output import (
    BufferSink,
    CallbackSink,
    FileSink,
    MultiSink,
    OutputSink,
    SyncCallbackSink,
)

__all__ = (
    # Output sinks
    "BufferSink",
    "CallbackSink",
    "FileSink",
    # Consumers
    "LNDLConsumer",
    "MultiSink",
    "OutputSink",
    "PrintConsumer",
    # Channel
    "SourcedStreamChannel",
    "StreamChannel",
    "StreamChunk",
    "StreamConsumer",
    "SyncCallbackSink",
    "TextConsumer",
)
