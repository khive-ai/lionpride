# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("OperationRegistry",)

# Factory signature: (session, branch, parameters) -> result
OperationFactory = Callable[
    ["Session", "Branch | str", dict[str, Any]],
    Awaitable[Any],
]


class OperationRegistry:
    """Per-session registry mapping operation names to factory functions.

    Unlike the global dispatcher pattern, OperationRegistry is
    instantiated per-Session for:
    - Better isolation between sessions
    - Easier testing (no global state)
    - Per-session operation customization

    Default operations are auto-registered on creation:
    - operate: Structured output with optional actions
    - react: Multi-step reasoning with tool calling
    - communicate: Stateful chat with optional structured output
    - generate: Stateless text generation
    - parse: JSON extraction with LLM fallback
    - interpret: Instruction refinement

    Usage:
        # Session creates registry automatically
        session = Session(default_imodel=iModel(...))

        # Conduct operations through Session
        result = await session.conduct("operate", branch, instruction="...", ...)

        # Custom operations can be registered per-session
        session.operations.register("my_op", my_operation_factory)
    """

    def __init__(self, *, auto_register_defaults: bool = True):
        """Initialize operation registry.

        Args:
            auto_register_defaults: If True, register default operations
                (operate, react, communicate, generate) on creation.
        """
        self._factories: dict[str, OperationFactory] = {}

        if auto_register_defaults:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default operations.

        Imports are deferred to avoid circular imports.
        Factories are registered directly - they expect typed Params objects.
        """
        from .operate import (
            communicate,
            generate,
            interpret,
            operate,
            parse,
            react,
        )

        # Register factories directly - they expect typed params
        self._factories["operate"] = operate
        self._factories["react"] = react
        self._factories["communicate"] = communicate
        self._factories["generate"] = generate
        self._factories["parse"] = parse
        self._factories["interpret"] = interpret

    def register(
        self,
        operation_name: str,
        factory: OperationFactory,
        *,
        override: bool = False,
    ) -> None:
        """Register operation with factory function.

        Args:
            operation_name: Name for the operation
            factory: Async factory function (session, branch, params) -> result
            override: If True, allow replacing existing registration
        """
        if operation_name in self._factories and not override:
            raise ValueError(
                f"Operation '{operation_name}' already registered. Use override=True to replace."
            )
        self._factories[operation_name] = factory

    def get(self, operation_name: str) -> OperationFactory:
        """Get factory for operation name.

        Raises:
            KeyError: If operation not registered
        """
        if operation_name not in self._factories:
            raise KeyError(
                f"Operation '{operation_name}' not registered. Available: {self.list_names()}"
            )
        return self._factories[operation_name]

    def has(self, operation_name: str) -> bool:
        """Check if operation is registered."""
        return operation_name in self._factories

    def unregister(self, operation_name: str) -> bool:
        """Unregister operation. Returns True if removed."""
        if operation_name in self._factories:
            del self._factories[operation_name]
            return True
        return False

    def list_names(self) -> list[str]:
        """List all registered operation names."""
        return list(self._factories.keys())

    def __contains__(self, operation_name: str) -> bool:
        """Support 'in' operator."""
        return operation_name in self._factories

    def __len__(self) -> int:
        """Return number of registered operations."""
        return len(self._factories)

    def __repr__(self) -> str:
        return f"OperationRegistry(operations={self.list_names()})"
