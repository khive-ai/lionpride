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
    """Executes operation graphs with dependency management and context inheritance."""

    def __init__(
        self,
        session: Session,
        graph: Graph,
        context: dict[str, Any] | None = None,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        verbose: bool = False,
        default_branch: Branch | str | None = None,
    ):
        """Initialize executor."""
        self.session = session
        self.graph = graph
        self.context = context or {}
        self.max_concurrent = max_concurrent
        self.stop_on_error = stop_on_error
        self.verbose = verbose
        self._default_branch = default_branch

        # Track results and completion
        self.results: dict[UUID, Any] = {}
        self.errors: dict[UUID, Exception] = {}
        self.completion_events: dict[UUID, concurrency.Event] = {}
        self.operation_branches: dict[UUID, Branch] = {}
        self.skipped_operations: set[UUID] = set()

        # Concurrency limiter - acquired AFTER dependencies resolve
        # This ensures only ready-to-execute tasks hold limiter slots
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

    async def _preallocate_branches(self) -> None:
        """Pre-allocate branches for all operations."""
        # Resolve default branch
        default_branch = self._default_branch
        if isinstance(default_branch, str):
            default_branch = self.session.conversations.get_progression(default_branch)
        elif default_branch is None:
            default_branch = getattr(self.session, "default_branch", None)

        # For now, all operations use the same branch
        # In future, support per-operation branch assignment
        for node in self.graph.nodes:
            if isinstance(node, Operation):
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
            # This prevents blocked tasks from holding slots
            if self._limiter:
                await self._limiter.acquire()

            try:
                # Prepare operation context with predecessor results
                self._prepare_operation_context(operation)

                # Execute the operation
                await self._invoke_operation(operation)
            finally:
                # Release limiter slot
                if self._limiter:
                    self._limiter.release()

        except Exception as e:
            # Store error
            self.errors[operation.id] = e
            if self.verbose:
                import traceback

                print(f"Operation {str(operation.id)[:8]} failed: {e}")
                print(f"Traceback: {traceback.format_exc()}")

            # Re-raise if stop_on_error is enabled
            if self.stop_on_error:
                # Still signal completion before re-raising
                self.completion_events[operation.id].set()
                raise

        finally:
            # Signal completion regardless of success/failure
            self.completion_events[operation.id].set()

        return operation

    async def _wait_for_dependencies(self, operation: Operation) -> None:
        """Wait for all predecessor operations to complete."""
        # Check for aggregation sources (special handling)
        is_aggregation = operation.metadata.get("aggregation", False)
        if is_aggregation and isinstance(operation.parameters, dict):
            aggregation_sources = operation.parameters.get("aggregation_sources", [])
            if aggregation_sources:
                if self.verbose:
                    print(
                        f"Aggregation {str(operation.id)[:8]} waiting for "
                        f"{len(aggregation_sources)} sources"
                    )

                # Wait for all aggregation sources
                for source_id_str in aggregation_sources:
                    # Find matching operation by ID string
                    for op_id, event in self.completion_events.items():
                        if str(op_id) == source_id_str:
                            await event.wait()
                            break

        # Wait for graph predecessors (normal dependency edges)
        predecessors = self.graph.get_predecessors(operation)

        if self.verbose and predecessors:
            print(
                f"Operation {str(operation.id)[:8]} waiting for "
                f"{len(predecessors)} graph dependencies"
            )

        # Wait for all predecessors to complete
        for pred in predecessors:
            if pred.id in self.completion_events:
                await self.completion_events[pred.id].wait()

    def _prepare_operation_context(self, operation: Operation) -> None:
        """Prepare operation parameters with predecessor results."""
        predecessors = self.graph.get_predecessors(operation)

        if not predecessors:
            # No dependencies - just add shared context if present
            if self.context and "context" not in operation.parameters:
                operation.parameters["context"] = self.context.copy()
            return

        # Build context from predecessor results
        pred_context = {}
        for pred in predecessors:
            # Skip if predecessor was skipped or failed
            if pred.id in self.skipped_operations or pred.id in self.errors:
                continue

            # Add predecessor result to context
            if pred.id in self.results:
                result = self.results[pred.id]
                # Use a clean key name
                pred_name = pred.metadata.get("name", str(pred.id))
                pred_context[f"{pred_name}_result"] = result

        # Merge with shared execution context
        if self.context:
            pred_context.update(self.context)

        # Update operation parameters
        if isinstance(operation.parameters, dict):
            # If parameters is a dict, merge context
            if "context" not in operation.parameters:
                operation.parameters["context"] = pred_context
            else:
                # Merge with existing context
                existing = operation.parameters["context"]
                if isinstance(existing, dict):
                    existing.update(pred_context)
                else:
                    # Existing context is not a dict - wrap it
                    operation.parameters["context"] = {
                        "original_context": existing,
                        **pred_context,
                    }

        if self.verbose:
            print(
                f"Operation {str(operation.id)[:8]} prepared with {len(pred_context)} context items"
            )

    async def _invoke_operation(self, operation: Operation) -> None:
        """Invoke operation and store result.

        Operation is both Node (graph) and Event (lifecycle), so it
        can invoke itself directly after binding to session/branch.
        """
        if self.verbose:
            print(f"Executing operation: {str(operation.id)[:8]}")

        # Get branch for this operation
        branch = self.operation_branches.get(operation.id)
        if branch is None:
            raise ValueError(f"No branch allocated for operation {operation.id}")

        # Bind operation to session/branch and invoke
        # Operation is both Node and Event, so it handles its own execution
        operation.bind(self.session, branch)
        await operation.invoke()

        # Check execution status
        if self.verbose:
            print(f"Operation {str(operation.id)[:8]} status after invoke: {operation.status}")
            if hasattr(operation.execution, "error"):
                print(f"  Execution error: {operation.execution.error}")

        if operation.status == EventStatus.COMPLETED:
            # Event stores result in response property
            result = operation.response
            self.results[operation.id] = result

            # Update shared context if result contains context
            if isinstance(result, dict) and "context" in result:
                self.context.update(result["context"])

            if self.verbose:
                print(f"Completed operation: {str(operation.id)[:8]}")
        else:
            # Execution failed
            error_msg = f"Execution status: {operation.status}"
            if hasattr(operation.execution, "error") and operation.execution.error:
                error_msg += f" - {operation.execution.error}"
            error = RuntimeError(error_msg)
            self.errors[operation.id] = error
            if self.verbose:
                print(f"Operation {str(operation.id)[:8]} failed: {error_msg}")


async def flow(
    session: Session,
    branch: Branch | str,
    graph: Graph,
    *,
    context: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute operation graph with dependency-aware scheduling.

    Args:
        graph: Operation graph (DAG) to execute.
        max_concurrent: Max concurrent operations (None = unlimited).

    Returns:
        Dictionary mapping operation names to their results.
    """
    executor = DependencyAwareExecutor(
        session=session,
        graph=graph,
        context=context,
        max_concurrent=max_concurrent,
        stop_on_error=stop_on_error,
        verbose=verbose,
        default_branch=branch,
    )

    return await executor.execute()


async def flow_stream(
    session: Session,
    branch: Branch | str,
    graph: Graph,
    *,
    context: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
) -> AsyncGenerator[OperationResult, None]:
    """Execute operation graph, yielding results as operations complete."""
    executor = DependencyAwareExecutor(
        session=session,
        graph=graph,
        context=context,
        max_concurrent=max_concurrent,
        stop_on_error=stop_on_error,
        verbose=False,
        default_branch=branch,
    )

    async for result in executor.stream_execute():
        yield result
