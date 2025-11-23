# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""OperationDispatcher - Registry mapping operation types to factory functions."""

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
    """Registry mapping operation types to factory functions.

    Fixes GAP-002: Provides dispatch layer connecting operation type strings
    ("generate", "operate") to implementation functions.

    Thread-safe singleton pattern ensures consistent registration across modules.

    Example:
        >>> dispatcher = get_dispatcher()
        >>> dispatcher.register("generate", generate_factory)
        >>> factory = dispatcher.get_factory("generate")
        >>> result = await factory(session, branch, {"instruction": "Hi"})
    """

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
        """Register operation type with factory function.

        Args:
            operation_type: Operation type string ("generate", "operate", etc.)
            factory: Async factory function with signature:
                     (session, branch, parameters) -> result
            override: Allow overriding existing registration (default: False)

        Raises:
            ValueError: If operation_type already registered and override=False

        Example:
            >>> async def my_factory(session, branch, params):
            ...     return await do_something(params)
            >>> dispatcher.register("custom", my_factory)
        """
        if operation_type in self._factories and not override:
            raise ValueError(
                f"Operation type '{operation_type}' already registered. "
                f"Use override=True to replace."
            )

        self._factories[operation_type] = factory

    def get_factory(self, operation_type: str) -> OperationFactory | None:
        """Get factory for operation type.

        Args:
            operation_type: Operation type string

        Returns:
            Factory function or None if not registered

        Example:
            >>> factory = dispatcher.get_factory("generate")
            >>> if factory:
            ...     result = await factory(session, branch, params)
        """
        return self._factories.get(operation_type)

    def unregister(self, operation_type: str) -> bool:
        """Unregister operation type.

        Args:
            operation_type: Operation type to remove

        Returns:
            True if removed, False if not registered

        Example:
            >>> dispatcher.unregister("obsolete_op")
        """
        if operation_type in self._factories:
            del self._factories[operation_type]
            return True
        return False

    def list_types(self) -> list[str]:
        """List all registered operation types.

        Returns:
            List of operation type strings

        Example:
            >>> dispatcher.list_types()
            ['generate', 'operate', 'chat']
        """
        return list(self._factories.keys())

    def is_registered(self, operation_type: str) -> bool:
        """Check if operation type is registered.

        Args:
            operation_type: Operation type to check

        Returns:
            True if registered, False otherwise

        Example:
            >>> if dispatcher.is_registered("generate"):
            ...     # Use it
        """
        return operation_type in self._factories

    def clear(self) -> None:
        """Clear all registrations.

        Primarily for testing - use with caution in production.

        Example:
            >>> dispatcher.clear()  # Remove all registered operations
        """
        self._factories.clear()

    def __repr__(self) -> str:
        return f"OperationDispatcher(registered={len(self._factories)})"


# Global dispatcher singleton
_dispatcher: OperationDispatcher | None = None


def get_dispatcher() -> OperationDispatcher:
    """Get global operation dispatcher (singleton).

    Returns:
        Global OperationDispatcher instance

    Example:
        >>> dispatcher = get_dispatcher()
        >>> dispatcher.register("custom", my_factory)
    """
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = OperationDispatcher()
    return _dispatcher


def register_operation(
    operation_type: str,
    *,
    override: bool = False,
) -> Callable[[OperationFactory], OperationFactory]:
    """Decorator for registering operation factories.

    Args:
        operation_type: Operation type string
        override: Allow overriding existing registration

    Returns:
        Decorator function

    Example:
        >>> @register_operation("generate")
        ... async def generate_factory(session, branch, params):
        ...     return await do_generation(params)
    """

    def decorator(factory: OperationFactory) -> OperationFactory:
        dispatcher = get_dispatcher()
        dispatcher.register(operation_type, factory, override=override)
        return factory

    return decorator
