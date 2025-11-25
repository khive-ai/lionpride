# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Stream consumers for processing streamed content.

Provides consumer classes that can be attached to StreamChannel
for real-time processing of streamed content.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .channel import StreamChunk


class StreamConsumer(ABC):
    """Abstract base for stream consumers.

    Consumers process StreamChunks as they arrive, performing
    real-time analysis or transformation.
    """

    @abstractmethod
    def consume(self, chunk: StreamChunk) -> None:
        """Process a stream chunk.

        Args:
            chunk: StreamChunk to process
        """
        ...

    def __call__(self, chunk: StreamChunk) -> None:
        """Allow consumer to be used directly as callback."""
        self.consume(chunk)


class TextConsumer(StreamConsumer):
    """Simple text accumulator consumer.

    Accumulates text chunks and optionally calls a callback
    for each chunk or on completion.
    """

    def __init__(
        self,
        on_chunk: Callable[[str], None] | None = None,
        on_complete: Callable[[str], None] | None = None,
    ):
        """Initialize text consumer.

        Args:
            on_chunk: Optional callback for each chunk content
            on_complete: Optional callback when stream completes
        """
        self._buffer: list[str] = []
        self._on_chunk = on_chunk
        self._on_complete = on_complete

    def consume(self, chunk: StreamChunk) -> None:
        """Accumulate chunk and notify callbacks."""
        if chunk.is_final:
            if self._on_complete:
                self._on_complete(self.get_text())
            return

        self._buffer.append(chunk.content)
        if self._on_chunk:
            self._on_chunk(chunk.content)

    def get_text(self) -> str:
        """Get accumulated text."""
        return "".join(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


class LNDLConsumer(StreamConsumer):
    """Consumer for detecting LNDL constructs in streamed content.

    Monitors streamed content for LNDL v2 cognitive constructs
    like <yield/>, <context/>, and <send/>.
    """

    def __init__(
        self,
        on_yield: Callable[[dict[str, Any]], None] | None = None,
        on_context: Callable[[dict[str, Any]], None] | None = None,
        on_send: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize LNDL consumer.

        Args:
            on_yield: Callback when <yield> detected
            on_context: Callback when <context> detected
            on_send: Callback when <send> detected
        """
        self._buffer = ""
        self._on_yield = on_yield
        self._on_context = on_context
        self._on_send = on_send

    def consume(self, chunk: StreamChunk) -> None:
        """Buffer content and check for LNDL markers."""
        if chunk.is_final:
            # Final parse of complete buffer
            self._check_markers()
            return

        self._buffer += chunk.content
        self._check_markers()

    def _check_markers(self) -> None:
        """Check buffer for complete LNDL markers."""
        import re

        # Check for yield markers
        if self._on_yield:
            yield_pattern = r"<yield\s+([^>]*)/>"
            for match in re.finditer(yield_pattern, self._buffer):
                attrs = self._parse_attrs(match.group(1))
                self._on_yield(attrs)

        # Check for context markers
        if self._on_context:
            context_pattern = r"<context>(.*?)</context>"
            for match in re.finditer(context_pattern, self._buffer, re.DOTALL):
                self._on_context({"content": match.group(1)})

        # Check for send markers
        if self._on_send:
            send_pattern = r'<send\s+to="([^"]+)"[^>]*>(.*?)</send>'
            for match in re.finditer(send_pattern, self._buffer, re.DOTALL):
                self._on_send({"to": match.group(1), "content": match.group(2)})

    def _parse_attrs(self, attr_str: str) -> dict[str, Any]:
        """Parse attribute string into dict."""
        import re

        attrs = {}
        for match in re.finditer(r'(\w+)="([^"]*)"', attr_str):
            attrs[match.group(1)] = match.group(2)
        return attrs

    def get_buffer(self) -> str:
        """Get current buffer content."""
        return self._buffer

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = ""


class PrintConsumer(StreamConsumer):
    """Consumer that prints chunks to stdout.

    Useful for debugging or CLI applications.
    """

    def __init__(self, prefix: str = "", end: str = "", flush: bool = True):
        """Initialize print consumer.

        Args:
            prefix: Prefix to print before each chunk
            end: String to print after each chunk (default: no newline)
            flush: Whether to flush stdout after each chunk
        """
        self._prefix = prefix
        self._end = end
        self._flush = flush

    def consume(self, chunk: StreamChunk) -> None:
        """Print chunk content."""
        if chunk.is_final:
            print()  # Final newline
            return

        print(f"{self._prefix}{chunk.content}", end=self._end, flush=self._flush)


__all__ = (
    "LNDLConsumer",
    "PrintConsumer",
    "StreamConsumer",
    "TextConsumer",
)
