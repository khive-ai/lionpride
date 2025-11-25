# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import Field

from lionpride.core import Event, Node

if TYPE_CHECKING:
    pass

__all__ = ("Operation", "OperationType")

OperationType = Literal["communicate", "operate", "react", "generate"]


class Operation(Node, Event):
    """Operation execution combining Node (graph) + Event (lifecycle).

    Multiple inheritance pattern from lionagi v0:
    - Node: Graph membership, content storage, UUID identity, embedding
    - Event: Lifecycle tracking (PENDING→PROCESSING→COMPLETED→FAILED)

    Why both?
    - Node: Can be added to operation graphs, has content field for metadata
    - Event: Has invoke() execution, status tracking, EventBus integration

    Usage:
        # Create operation
        op = Operation(
            operation_type="communicate",
            parameters={"instruction": "..."},
            session_id=session.id,
            branch_id=branch.id,
        )

        # As Node: add to graph
        graph.add_node(op)

        # As Event: execute
        result = await op.invoke()
        print(op.status)  # EventStatus.COMPLETED

        # Access results
        print(op.execution.output)
    """

    operation_type: OperationType | str = Field(..., description="Operation type to execute")
    parameters: Any = Field(default=None, description="Operation parameters (typed Param or dict)")

    # Execution context (IDs only, no circular refs)
    session_id: UUID = Field(..., description="Session ID for services")
    branch_id: UUID = Field(..., description="Branch ID for message context")

    async def _invoke(self) -> Any:
        """Execute operation via IPU context.

        Called by Event.invoke() during execution.
        Gets session from IPU's thread-local context.

        Returns:
            Operation result

        Raises:
            RuntimeError: If no IPU in context or operation execution fails
        """
        from lionpride.ipu import get_current_ipu

        # Get IPU from context (set by IPU.execute())
        try:
            ipu = get_current_ipu()
        except RuntimeError as e:
            raise RuntimeError(
                "Operation must be executed within IPU context. "
                "Use ipu.execute(operation) or session.conduct(..., ipu=ipu)"
            ) from e

        # Get session from IPU registry
        session = ipu.get_session(self.session_id)
        branch = session.get_branch(self.branch_id)

        # Get operation function from session registry
        operation_func = session.operations.get(self.operation_type)

        # Execute operation
        try:
            result = await operation_func(session, branch, self.parameters)
            return result
        except Exception as e:
            raise RuntimeError(f"Operation '{self.operation_type}' failed: {e}") from e

    def __repr__(self) -> str:
        return f"Operation(type={self.operation_type}, status={self.status}, id={str(self.id)[:8]})"
