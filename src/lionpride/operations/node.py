# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PrivateAttr

from lionpride.core import Event, Node

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("Operation", "OperationType", "create_operation")

OperationType = Literal[
    "generate",
    "communicate",
    "operate",
    "react",
]


class Operation(Node, Event):
    """Unified operation: Node (graph) + Event (lifecycle).

    Multiple inheritance pattern:
    - Node: Graph membership, UUID identity, content storage
    - Event: Lifecycle tracking (PENDING→PROCESSING→COMPLETED→FAILED)

    Why both?
    - As Node: Can be added to operation graphs for workflow execution
    - As Event: Has invoke() for execution, status tracking, result storage

    Usage:
        # Create and execute directly
        op = Operation(
            operation_type="communicate",
            parameters={"instruction": "Hello"},
        )
        op.bind(session, branch)
        await op.invoke()
        print(op.status)  # EventStatus.COMPLETED
        print(op.response)  # "Hello! How can I help?"

        # Via Session.conduct() (preferred)
        op = await session.conduct("communicate", branch, instruction="Hello")
        print(op.response)  # Already invoked, result available

        # As graph node
        graph = Graph()
        graph.add_node(op)  # Works because Operation is a Node
    """

    # Operation specification
    operation_type: OperationType | str = Field(
        ..., description="Operation type (communicate, operate, react, generate)"
    )
    parameters: dict[str, Any] | Any = Field(
        default_factory=dict,
        description="Operation parameters (dict or Pydantic model)",
    )

    # Execution context (private to avoid serialization issues)
    _session: Any = PrivateAttr(default=None)
    _branch: Any = PrivateAttr(default=None)

    def bind(self, session: Session, branch: Branch) -> Operation:
        """Bind session and branch for execution.

        Must be called before invoke() if not using Session.conduct().

        Args:
            session: Session with operations registry and services
            branch: Branch for message context

        Returns:
            Self for chaining
        """
        self._session = session
        self._branch = branch
        return self

    def _require_binding(self) -> tuple[Session, Branch]:
        """Get bound session/branch or raise error."""
        if self._session is None or self._branch is None:
            raise RuntimeError(
                "Operation not bound to session/branch. "
                "Use operation.bind(session, branch) or session.conduct(...)"
            )
        return self._session, self._branch

    async def _invoke(self) -> Any:
        """Execute operation via session's registry.

        Called by Event.invoke() during execution.

        Returns:
            Operation result from factory

        Raises:
            RuntimeError: If not bound to session/branch
            KeyError: If operation type not registered
        """
        session, branch = self._require_binding()

        # Get factory from session's operation registry
        factory = session.operations.get(self.operation_type)

        # Execute operation via factory
        # Factory signature: (session, branch, params) -> result
        return await factory(session, branch, self.parameters)

    def __repr__(self) -> str:
        bound = "bound" if self._session is not None else "unbound"
        return f"Operation(type={self.operation_type}, status={self.status.value}, {bound})"


def create_operation(
    operation_type: OperationType | str | None = None,
    parameters: dict[str, Any] | None = None,
    *,
    operation: OperationType | str | None = None,  # Legacy kwarg for backward compat
    **kwargs,
) -> Operation:
    """Helper to create Operation nodes.

    Args:
        operation_type: Operation type (communicate, operate, react, generate)
        parameters: Operation parameters dict
        operation: Legacy alias for operation_type (backward compatibility)
        **kwargs: Additional Operation fields (metadata, etc.)

    Returns:
        Operation node ready for binding and invocation
    """
    # Support legacy `operation=` kwarg
    op_type = operation_type or operation
    if op_type is None:
        raise ValueError("operation_type (or operation) is required")

    return Operation(
        operation_type=op_type,
        parameters=parameters or {},
        **kwargs,
    )
