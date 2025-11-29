# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from lionpride.core import EventStatus, Graph
from lionpride.libs import concurrency
from lionpride.libs.concurrency import CapacityLimiter, CompletionStream

from .node import Operation

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("DependencyAwareExecutor", "OperationResult", "flow", "flow_stream")


@dataclass
class OperationResult:
    """Result from a completed operation in streaming execution."""

    name: str
    """Operation name"""
    result: Any
    """Operation result (None if failed)"""
    error: Exception | None = None
    """Exception if operation failed"""
    completed: int = 0
    """Number of operations completed so far"""
    total: int = 0
    """Total number of operations"""

    @property
    def success(self) -> bool:
        """Whether the operation succeeded."""
        return self.error is None


class DependencyAwareExecutor:
    """Executes operation graphs with dependency management.

    This executor:
    - Waits for predecessor operations to complete before executing
    - Respects concurrency limits
    - Tracks results and errors

    Note: Context management is NOT handled here. Operations should
    have their context set before being added to the graph, or the
    caller should use a higher-level abstraction like flow_report.
    """

    def __init__(
        self,
        session: Session,
        graph: Graph,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        verbose: bool = False,
        default_branch: Branch | str | None = None,
    ):
        """Initialize executor."""
        self.session = session
        self.graph = graph
        self.max_concurrent = max_concurrent
        self.stop_on_error = stop_on_error
        self.verbose = verbose
        self._default_branch = default_branch

        # Track results and completion
        self.results: dict[UUID, Any] = {}
        self.errors: dict[UUID, Exception] = {}
        self.completion_events: dict[UUID, concurrency.Event] = {}
        self.operation_branches: dict[UUID, Branch] = {}

        # Concurrency limiter - acquired AFTER dependencies resolve
        self._limiter: CapacityLimiter | None = (
            CapacityLimiter(max_concurrent) if max_concurrent else None
        )

        # Initialize completion events for all operations
        for node in graph.nodes:
            if isinstance(node, Operation):
                self.completion_events[node.id] = concurrency.Event()

    async def execute(self) -> dict[str, Any]:
        """Execute the operation graph with dependency coordination."""
        # Validate graph is acyclic
        if not self.graph.is_acyclic():
            raise ValueError("Operation graph has cycles - must be a DAG")

        # Validate all nodes are Operations
        for node in self.graph.nodes:
            if not isinstance(node, Operation):
                raise ValueError(
                    f"Graph contains non-Operation node: {node} ({type(node).__name__})"
                )

        # Pre-allocate branches to avoid locking during execution
        await self._preallocate_branches()

        # Execute operations with dependency coordination
        operations = [node for node in self.graph.nodes if isinstance(node, Operation)]

        # Create operation tasks (they wait on dependencies internally)
        tasks = [self._execute_operation(op) for op in operations]

        # Use CompletionStream to process results as they arrive
        # Concurrency is handled by self._limiter AFTER dependency resolution
        # This ensures limiter slots are only held by tasks ready to execute
        async with CompletionStream(tasks, limit=None) as stream:
            async for idx, _ in stream:
                op = operations[idx]
                if self.verbose:
                    name = op.metadata.get("name", str(op.id)[:8])
                    if op.id in self.errors:
                        print(f"Operation '{name}' failed")
                    elif op.id in self.results:
                        print(f"Operation '{name}' completed")

        # Compile results keyed by operation name for user-friendly access
        results_by_name = {}
        for node in self.graph.nodes:
            if isinstance(node, Operation):
                name = node.metadata.get("name", str(node.id))
                if node.id in self.results:
                    results_by_name[name] = self.results[node.id]

        return results_by_name

    async def stream_execute(self) -> AsyncGenerator[OperationResult, None]:
        """Execute the operation graph, yielding results as operations complete."""
        # Validate graph is acyclic
        if not self.graph.is_acyclic():
            raise ValueError("Operation graph has cycles - must be a DAG")

        # Validate all nodes are Operations
        for node in self.graph.nodes:
            if not isinstance(node, Operation):
                raise ValueError(
                    f"Graph contains non-Operation node: {node} ({type(node).__name__})"
                )

        # Pre-allocate branches
        await self._preallocate_branches()

        # Execute operations with dependency coordination
        operations = [node for node in self.graph.nodes if isinstance(node, Operation)]
        total = len(operations)

        # Create operation tasks
        tasks = [self._execute_operation(op) for op in operations]

        # Stream results as they complete
        # Concurrency is handled by self._limiter AFTER dependency resolution
        completed = 0
        async with CompletionStream(tasks, limit=None) as stream:
            async for idx, _ in stream:
                completed += 1
                op = operations[idx]
                name = op.metadata.get("name", str(op.id))

                # Build result
                if op.id in self.errors:
                    yield OperationResult(
                        name=name,
                        result=None,
                        error=self.errors[op.id],
                        completed=completed,
                        total=total,
                    )
                else:
                    yield OperationResult(
                        name=name,
                        result=self.results.get(op.id),
                        error=None,
                        completed=completed,
                        total=total,
                    )

    def _resolve_operation_branch(self, branch_spec: Any) -> Branch | None:
        """Resolve branch specification to a Branch object.

        Args:
            branch_spec: Can be a Branch, UUID, string name, or None

        Returns:
            Resolved Branch or None if unresolvable
        """
        if branch_spec is None:
            return None

        # Already a Branch
        if hasattr(branch_spec, "id") and hasattr(branch_spec, "order"):
            return branch_spec

        # UUID or string - use session.get_branch
        if isinstance(branch_spec, (UUID, str)):
            try:
                return self.session.get_branch(branch_spec)
            except Exception:
                # Branch not found - return None
                return None

        return None

    async def _preallocate_branches(self) -> None:
        """Pre-allocate branches for all operations."""
        # Resolve default branch
        default_branch = self._resolve_operation_branch(self._default_branch)
        if default_branch is None:
            default_branch = getattr(self.session, "default_branch", None)

        # Assign branches to operations
        # Operations can specify branch in metadata, otherwise use default
        for node in self.graph.nodes:
            if isinstance(node, Operation):
                # Check if operation has a branch specified in metadata
                op_branch = node.metadata.get("branch")
                if op_branch is not None:
                    resolved = self._resolve_operation_branch(op_branch)
                    self.operation_branches[node.id] = (
                        resolved if resolved is not None else default_branch
                    )
                else:
                    self.operation_branches[node.id] = default_branch

        if self.verbose:
            print(f"Pre-allocated branches for {len(self.operation_branches)} operations")

    async def _execute_operation(
        self,
        operation: Operation,
    ) -> Operation:
        """Execute single operation with dependency coordination."""
        try:
            # Wait for all dependencies to complete (no limiter held yet)
            await self._wait_for_dependencies(operation)

            # Acquire limiter slot ONLY when ready to execute
            if self._limiter:
                await self._limiter.acquire()

            try:
                # Execute the operation (no context injection - use params as-is)
                await self._invoke_operation(operation)
            finally:
                if self._limiter:
                    self._limiter.release()

        except Exception as e:
            self.errors[operation.id] = e
            if self.verbose:
                import traceback

                print(f"Operation {str(operation.id)[:8]} failed: {e}")
                print(f"Traceback: {traceback.format_exc()}")

            if self.stop_on_error:
                self.completion_events[operation.id].set()
                raise

        finally:
            self.completion_events[operation.id].set()

        return operation

    async def _wait_for_dependencies(self, operation: Operation) -> None:
        """Wait for all predecessor operations to complete."""
        predecessors = self.graph.get_predecessors(operation)

        if self.verbose and predecessors:
            print(f"Operation {str(operation.id)[:8]} waiting for {len(predecessors)} dependencies")

        for pred in predecessors:
            if pred.id in self.completion_events:
                await self.completion_events[pred.id].wait()

    async def _invoke_operation(self, operation: Operation) -> None:
        """Invoke operation and store result."""
        if self.verbose:
            name = operation.metadata.get("name", str(operation.id)[:8])
            print(f"Executing operation: {name}")

        branch = self.operation_branches.get(operation.id)
        if branch is None:
            raise ValueError(f"No branch allocated for operation {operation.id}")

        operation.bind(self.session, branch)
        await operation.invoke()

        if operation.status == EventStatus.COMPLETED:
            self.results[operation.id] = operation.response
            if self.verbose:
                name = operation.metadata.get("name", str(operation.id)[:8])
                print(f"Completed operation: {name}")
        else:
            error_msg = f"Execution status: {operation.status}"
            if hasattr(operation.execution, "error") and operation.execution.error:
                error_msg += f" - {operation.execution.error}"
            self.errors[operation.id] = RuntimeError(error_msg)
            if self.verbose:
                name = operation.metadata.get("name", str(operation.id)[:8])
                print(f"Operation {name} failed: {error_msg}")


async def flow(
    session: Session,
    graph: Graph,
    *,
    branch: Branch | str | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute operation graph with dependency-aware scheduling.

    Operations are executed with their given parameters - no context injection.
    For context passing between operations, use flow_report or manage context
    explicitly before adding operations to the graph.

    Args:
        session: Session for services and branches.
        graph: Operation graph (DAG) to execute.
        branch: Default branch (operations can override via metadata).
        max_concurrent: Max concurrent operations (None = unlimited).
        stop_on_error: Stop on first error.
        verbose: Print progress.

    Returns:
        Dictionary mapping operation names to their results.
    """
    executor = DependencyAwareExecutor(
        session=session,
        graph=graph,
        max_concurrent=max_concurrent,
        stop_on_error=stop_on_error,
        verbose=verbose,
        default_branch=branch,
    )

    return await executor.execute()


async def flow_stream(
    session: Session,
    graph: Graph,
    *,
    branch: Branch | str | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
) -> AsyncGenerator[OperationResult, None]:
    """Execute operation graph, yielding results as operations complete."""
    executor = DependencyAwareExecutor(
        session=session,
        graph=graph,
        max_concurrent=max_concurrent,
        stop_on_error=stop_on_error,
        verbose=False,
        default_branch=branch,
    )

    async for result in executor.stream_execute():
        yield result
