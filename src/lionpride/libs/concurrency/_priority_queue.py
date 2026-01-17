# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
from typing import Any, Generic, TypeVar

from ._primitives import Condition

# heapq requires items to be comparable, so we use Any for the internal list
T = TypeVar("T")

__all__ = ("PriorityQueue", "QueueEmpty", "QueueFull")


class QueueEmpty(Exception):  # noqa: N818
    """Exception raised when queue.get_nowait() is called on empty queue."""


class QueueFull(Exception):  # noqa: N818
    """Exception raised when queue.put_nowait() is called on full queue."""


class PriorityQueue(Generic[T]):
    """Async priority queue (heapq + anyio.Condition).

    API: Similar to asyncio.PriorityQueue, but nowait methods are async.

    Items are stored internally as (priority, sequence_number, item) to ensure
    stable ordering when priorities are equal, preventing TypeError when items
    aren't directly comparable.

    Attributes:
        maxsize: Maximum queue size (0 = unlimited)
    """

    def __init__(self, maxsize: int = 0):
        """Initialize priority queue.

        Args:
            maxsize: Max size (0 = unlimited)
        """
        if maxsize < 0:
            raise ValueError("maxsize must be >= 0")
        self.maxsize = maxsize
        self._queue: list[Any] = []  # heapq entries: (priority, seq, item)
        self._seq = 0  # Tie-breaker to ensure stable ordering
        self._condition = Condition()

    @staticmethod
    def _get_priority(item: Any) -> Any:
        """Extract priority from item.

        Convention: if item is a tuple/list, first element is priority.
        Otherwise, the item itself is the priority.
        """
        if isinstance(item, (tuple, list)) and item:
            return item[0]
        return item

    async def put(self, item: T) -> None:
        """Put item into queue (blocks if full).

        Args:
            item: Item (tuple with priority as first element)
        """
        async with self._condition:
            # Wait if queue is full
            while self.maxsize > 0 and len(self._queue) >= self.maxsize:
                await self._condition.wait()

            priority = self._get_priority(item)
            entry = (priority, self._seq, item)
            self._seq += 1
            heapq.heappush(self._queue, entry)
            self._condition.notify()

    async def put_nowait(self, item: T) -> None:
        """Put item without blocking (async, unlike asyncio).

        Args:
            item: Item (tuple with priority as first element)

        Raises:
            QueueFull: If queue is at maxsize
        """
        async with self._condition:
            if self.maxsize > 0 and len(self._queue) >= self.maxsize:
                raise QueueFull("Queue is full")

            priority = self._get_priority(item)
            entry = (priority, self._seq, item)
            self._seq += 1
            heapq.heappush(self._queue, entry)
            # Notify waiting getters that item is available
            self._condition.notify()

    async def get(self) -> T:
        """Get highest priority item (blocks if empty).

        Returns:
            Highest priority item (lowest value first)
        """
        async with self._condition:
            # Wait if queue is empty
            while not self._queue:
                await self._condition.wait()

            _priority, _seq, item = heapq.heappop(self._queue)
            self._condition.notify()
            return item

    async def get_nowait(self) -> T:
        """Get item without blocking (async, unlike asyncio).

        Returns:
            Highest priority item

        Raises:
            QueueEmpty: If queue is empty
        """
        async with self._condition:
            if not self._queue:
                raise QueueEmpty("Queue is empty")

            _priority, _seq, item = heapq.heappop(self._queue)
            # Notify waiting putters that space is available
            self._condition.notify()
            return item

    def qsize(self) -> int:
        """Approximate queue size (unlocked, racy).

        Note: Value may be stale immediately. Use for monitoring only.

        Returns:
            Number of items in queue
        """
        return len(self._queue)

    def empty(self) -> bool:
        """Check if queue is empty (unlocked, racy).

        Note: Value may be stale immediately. Use for monitoring only.

        Returns:
            True if queue is empty
        """
        return len(self._queue) == 0

    def full(self) -> bool:
        """Check if queue is full (unlocked, racy).

        Note: Value may be stale immediately. Use for monitoring only.

        Returns:
            True if queue is at maxsize
        """
        return self.maxsize > 0 and len(self._queue) >= self.maxsize
