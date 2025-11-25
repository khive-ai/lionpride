# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable

__all__ = ("OperationRegistry",)


class OperationRegistry:
    """Simple operation registry: name → async function.

    Matches ServiceRegistry pattern.
    """

    def __init__(self):
        self._registry: dict[str, Callable] = {}
        self._required_resources: dict[str, set[str]] = {}
        self._required_capabilities: dict[str, set[str]] = {}

    def register(
        self,
        name: str,
        func: Callable,
        update: bool = False,
        required_resources: set[str] | None = None,
        required_capabilities: set[str] | None = None,
    ) -> None:
        """Register async operation function.

        Args:
            name: Operation name
            func: Async function (session, branch, params) -> result
            update: If True, replaces existing operation
            required_resources: Service names required (branch.resources)
            required_capabilities: Schema names required (branch.capabilities)

        Raises:
            ValueError: If name exists and update=False, or func not async
        """
        if name in self._registry and not update:
            raise ValueError(f"Operation '{name}' already registered")

        if not asyncio.iscoroutinefunction(func):
            raise ValueError(f"Operation '{name}' must be async")

        self._registry[name] = func
        self._required_resources[name] = required_resources or set()
        self._required_capabilities[name] = required_capabilities or set()

    def get(self, name: str) -> Callable:
        """Get operation function by name."""
        if name not in self._registry:
            raise KeyError(f"Operation '{name}' not found")
        return self._registry[name]

    def get_metadata(self, name: str) -> dict[str, set[str]]:
        """Get operation access control requirements.

        Returns:
            Dict with required_resources and required_capabilities sets
        """
        if name not in self._registry:
            raise KeyError(f"Operation '{name}' not found")
        return {
            "required_resources": self._required_resources[name],
            "required_capabilities": self._required_capabilities[name],
        }

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def list_names(self) -> list[str]:
        return list(self._registry.keys())

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"OperationRegistry(count={len(self)})"
