# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""StreamChannel abstraction for buffered streaming with fan-out.

Provides a channel that buffers streaming content and supports
multiple consumers for fan-out processing patterns.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class StreamChunk:
    """A chunk of streamed content.

    Attributes:
        content: The content string
        index: Sequential index of this chunk
        is_final: Whether this is the last chunk
        metadata: Optional metadata (e.g., finish_reason, usage)
    """

    content: str
    index: int
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamChannel:
    """Async stream with buffering and fan-out to multiple consumers.

    Wraps an async iterator of strings and provides:
    - Buffering: accumulates all chunks for replay
    - Fan-out: notifies multiple consumers as chunks arrive
    - Accumulation: get full content after streaming completes

    Usage:
        async def process():
            channel = StreamChannel(response_stream)
            channel.add_consumer(lambda chunk: print(chunk.content))
            async for chunk in channel:
                # Process each chunk
                pass
            full_text = channel.get_accumulated()
    """

    def __init__(self, source: AsyncIterator[str]):
        """Initialize with an async iterator source.

        Args:
            source: Async iterator yielding string chunks
        """
        self._source = source
        self._buffer: list[StreamChunk] = []
        self._consumers: list[Callable[[StreamChunk], None]] = []
        self._accumulated = ""
        self._index = 0
        self._completed = False

    def add_consumer(self, consumer: Callable[[StreamChunk], None]) -> None:
        """Add a consumer callback for chunk notifications.

        Consumers are called synchronously as each chunk arrives.

        Args:
            consumer: Callable that receives StreamChunk
        """
        self._consumers.append(consumer)

    def remove_consumer(self, consumer: Callable[[StreamChunk], None]) -> bool:
        """Remove a consumer callback.

        Args:
            consumer: Consumer to remove

        Returns:
            True if consumer was found and removed
        """
        try:
            self._consumers.remove(consumer)
            return True
        except ValueError:
            return False

    def get_accumulated(self) -> str:
        """Get all accumulated content.

        Returns:
            Full accumulated string from all chunks
        """
        return self._accumulated

    def get_buffer(self) -> list[StreamChunk]:
        """Get the buffer of all chunks.

        Returns:
            List of all StreamChunks received
        """
        return self._buffer.copy()

    @property
    def is_completed(self) -> bool:
        """Whether the stream has completed."""
        return self._completed

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over chunks, buffering and notifying consumers.

        Yields:
            StreamChunk for each piece of content
        """
        async for raw_content in self._source:
            # Parse SSE content (skip empty lines, handle [DONE])
            content = self._parse_sse_content(raw_content)
            if content is None:
                continue

            # Create chunk
            chunk = StreamChunk(
                content=content,
                index=self._index,
                is_final=False,
            )

            # Buffer and accumulate
            self._buffer.append(chunk)
            self._accumulated += content
            self._index += 1

            # Notify consumers
            for consumer in self._consumers:
                consumer(chunk)

            yield chunk

        # Mark completion
        self._completed = True

        # Send final chunk notification
        if self._buffer:
            final_chunk = StreamChunk(
                content="",
                index=self._index,
                is_final=True,
                metadata={"total_chunks": len(self._buffer)},
            )
            for consumer in self._consumers:
                consumer(final_chunk)

    def _parse_sse_content(self, line: str) -> str | None:
        """Parse SSE line to extract content.

        Args:
            line: Raw SSE line

        Returns:
            Extracted content or None if not content line
        """
        import json

        if not line or not line.strip():
            return None

        # Handle "data: " prefix
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None

            try:
                parsed = json.loads(data)
                # OpenAI format
                if "choices" in parsed:
                    choices = parsed["choices"]
                    if choices and "delta" in choices[0]:
                        return choices[0]["delta"].get("content", "")
                    elif choices and "text" in choices[0]:
                        return choices[0]["text"]
                # Anthropic format
                elif "delta" in parsed:
                    return parsed["delta"].get("text", "")
                elif "content" in parsed:
                    return parsed["content"]
            except json.JSONDecodeError:
                # Return raw data if not JSON
                return data

        return line


class SourcedStreamChannel(StreamChannel):
    """StreamChannel with source event tracking.

    Extends StreamChannel to track which event the stream belongs to,
    useful for Processor-level fan-out where multiple streams may be active.
    """

    def __init__(self, source: AsyncIterator[str], event_id: UUID):
        """Initialize with source and event ID.

        Args:
            source: Async iterator yielding string chunks
            event_id: UUID of the event this stream belongs to
        """
        super().__init__(source)
        self.event_id = event_id


__all__ = (
    "SourcedStreamChannel",
    "StreamChannel",
    "StreamChunk",
)
