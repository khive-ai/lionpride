# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable

__all__ = ("OperationRegistry",)


class OperationRegistry:
    """Simple operation registry for async function lookup.

    Pattern matches ServiceRegistry and lionagi OperationManager.
    Stores operation functions with name-based O(1) lookup.
    """

    def __init__(self):
        self._registry: dict[str, Callable] = {}

    def register(self, name: str, func: Callable, update: bool = False) -> None:
        """Register async operation function.

        Args:
            name: Operation name
            func: Async function with signature (session, branch, params) -> result
            update: If True, replaces existing operation

        Raises:
            ValueError: If name exists and update=False, or func not async
        """
        if name in self._registry and not update:
            raise ValueError(f"Operation '{name}' already registered")

        if not asyncio.iscoroutinefunction(func):
            raise ValueError(f"Operation '{name}' must be async function")

        self._registry[name] = func

    def get(self, name: str) -> Callable:
        """Get operation function by name."""
        if name not in self._registry:
            raise KeyError(f"Operation '{name}' not found")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def list_names(self) -> list[str]:
        return list(self._registry.keys())

    def clear(self) -> None:
        self._registry.clear()

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"OperationRegistry(count={len(self)})"
