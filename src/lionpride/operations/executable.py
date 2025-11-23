# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""ExecutableOperation - Composition of Operation (data) + Event (execution)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from lionpride.core import Event

from .node import Operation

if TYPE_CHECKING:
    pass

__all__ = ("ExecutableOperation",)


class ExecutableOperation(Event):
    """Composes Operation (data) + Event (lifecycle) for execution.

    Attributes:
        operation: Operation data node
        session: Session for services and context
        branch: Branch for message context
    """

    operation: Operation = Field(..., description="Operation data node")
    session: Any = Field(
        ..., description="Session for services"
    )  # Use Any to avoid circular import
    branch: Any = Field(
        ..., description="Branch for message context"
    )  # Use Any to avoid circular import

    async def _invoke(self) -> Any:
        """Execute operation via dispatcher. Called by Event.invoke()."""
        # Import here to avoid circular dependency
        from .dispatcher import get_dispatcher

        dispatcher = get_dispatcher()

        # Get factory for operation type
        factory = dispatcher.get_factory(self.operation.content.operation)
        if factory is None:
            raise ValueError(
                f"Operation type '{self.operation.content.operation}' not registered. "
                f"Available: {list(dispatcher.list_types())}"
            )

        # Execute operation via factory
        # Factory signature: (session, branch, parameters) -> result
        try:
            result = await factory(
                self.session,
                self.branch,
                self.operation.content.parameters,
            )
            return result
        except Exception as e:
            # Add context to the error
            raise RuntimeError(
                f"Operation '{self.operation.content.operation}' failed in factory: {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"ExecutableOperation("
            f"operation={self.operation.content.operation}, "
            f"status={self.status}, "
            f"id={self.id})"
        )
