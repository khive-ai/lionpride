# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("OperationDispatcher", "get_dispatcher", "register_operation")

# Factory signature: (session, branch, parameters) -> result
OperationFactory = Callable[
    ["Session", "Branch | str", dict[str, Any] | Any],
    Awaitable[Any],
]


class OperationDispatcher:
    """Registry mapping operation types to factory functions."""

    def __init__(self):
        """Initialize empty registry."""
        self._factories: dict[str, OperationFactory] = {}

    def register(
        self,
        operation_type: str,
        factory: OperationFactory,
        *,
        override: bool = False,
    ) -> None:
        """Register operation type with factory function."""
        if operation_type in self._factories and not override:
            raise ValueError(
                f"Operation type '{operation_type}' already registered. "
                f"Use override=True to replace."
            )

        self._factories[operation_type] = factory

    def get_factory(self, operation_type: str) -> OperationFactory | None:
        """Get factory for operation type, or None if not registered."""
        return self._factories.get(operation_type)

    def unregister(self, operation_type: str) -> bool:
        """Unregister operation type. Returns True if removed."""
        if operation_type in self._factories:
            del self._factories[operation_type]
            return True
        return False

    def list_types(self) -> list[str]:
        """List all registered operation types."""
        return list(self._factories.keys())

    def is_registered(self, operation_type: str) -> bool:
        """Check if operation type is registered."""
        return operation_type in self._factories

    def clear(self) -> None:
        """Clear all registrations. Primarily for testing."""
        self._factories.clear()

    def __repr__(self) -> str:
        return f"OperationDispatcher(registered={len(self._factories)})"


# Global dispatcher singleton
_dispatcher: OperationDispatcher | None = None


def get_dispatcher() -> OperationDispatcher:
    """Get global operation dispatcher (singleton)."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = OperationDispatcher()
    return _dispatcher


def register_operation(
    operation_type: str,
    *,
    override: bool = False,
) -> Callable[[OperationFactory], OperationFactory]:
    """Decorator for registering operation factories."""

    def decorator(factory: OperationFactory) -> OperationFactory:
        dispatcher = get_dispatcher()
        dispatcher.register(operation_type, factory, override=override)
        return factory

    return decorator
