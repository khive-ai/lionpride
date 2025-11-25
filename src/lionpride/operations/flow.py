# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Flow execution for operation graphs.

Executes operation graphs built by OperationGraphBuilder.
Creates Operation instances at execution time with session/branch context.
Uses IPU for validated execution.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from lionpride.core import Graph, Node
from lionpride.libs import concurrency
from lionpride.libs.concurrency import CapacityLimiter, CompletionStream

from .operation import Operation

if TYPE_CHECKING:
    from lionpride.ipu import IPU
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
    """Executes operation graphs with dependency management and IPU validation.

    Creates Operation instances from graph nodes at execution time,
    providing session/branch context. Uses IPU for validated execution.
    """

    def __init__(
        self,
        session: Session,
        ipu: IPU,
        graph: Graph,
        context: dict[str, Any] | None = None,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        verbose: bool = False,
        default_branch: Branch | str | None = None,
    ):
        """Initialize executor."""
        self.session = session
        self.ipu = ipu
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
        self._limiter: CapacityLimiter | None = (
            CapacityLimiter(max_concurrent) if max_concurrent else None
        )

        # Initialize completion events for all nodes
        for node in graph.nodes:
            self.completion_events[node.id] = concurrency.Event()

    async def execute(self) -> dict[str, Any]:
        """Execute the operation graph with dependency coordination."""
        # Validate graph is acyclic
        if not self.graph.is_acyclic():
            raise ValueError("Operation graph has cycles - must be a DAG")

        # Pre-allocate branches
        await self._preallocate_branches()

        # Execute operations with dependency coordination
        nodes = list(self.graph.nodes)

        # Create operation tasks (they wait on dependencies internally)
        tasks = [self._execute_node(node) for node in nodes]

        # Use CompletionStream to process results as they arrive
        async with CompletionStream(tasks, limit=None) as stream:
            async for idx, _ in stream:
                node = nodes[idx]
                if self.verbose:
                    name = node.metadata.get("name", str(node.id)[:8])
                    if node.id in self.errors:
                        print(f"Operation '{name}' failed")
                    elif node.id in self.results:
                        print(f"Operation '{name}' completed")

        # Compile results keyed by operation name
        results_by_name = {}
        for node in self.graph.nodes:
            name = node.metadata.get("name", str(node.id))
            if node.id in self.results:
                results_by_name[name] = self.results[node.id]

        return results_by_name

    async def stream_execute(self) -> AsyncGenerator[OperationResult, None]:
        """Execute the operation graph, yielding results as operations complete."""
        # Validate graph is acyclic
        if not self.graph.is_acyclic():
            raise ValueError("Operation graph has cycles - must be a DAG")

        # Pre-allocate branches
        await self._preallocate_branches()

        # Execute operations with dependency coordination
        nodes = list(self.graph.nodes)
        total = len(nodes)

        # Create operation tasks
        tasks = [self._execute_node(node) for node in nodes]

        # Stream results as they complete
        completed = 0
        async with CompletionStream(tasks, limit=None) as stream:
            async for idx, _ in stream:
                completed += 1
                node = nodes[idx]
                name = node.metadata.get("name", str(node.id))

                # Build result
                if node.id in self.errors:
                    yield OperationResult(
                        name=name,
                        result=None,
                        error=self.errors[node.id],
                        completed=completed,
                        total=total,
                    )
                else:
                    yield OperationResult(
                        name=name,
                        result=self.results.get(node.id),
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
        for node in self.graph.nodes:
            if default_branch is not None:
                self.operation_branches[node.id] = default_branch

        if self.verbose:
            print(f"Pre-allocated branches for {len(self.operation_branches)} operations")

    async def _execute_node(self, node: Node) -> Node:
        """Execute single node with dependency coordination."""
        try:
            # Wait for all dependencies to complete (no limiter held yet)
            await self._wait_for_dependencies(node)

            # Acquire limiter slot ONLY when ready to execute
            if self._limiter:
                await self._limiter.acquire()

            try:
                # Prepare operation context with predecessor results
                self._prepare_operation_context(node)

                # Create and execute Operation
                await self._invoke_operation(node)
            finally:
                # Release limiter slot
                if self._limiter:
                    self._limiter.release()

        except Exception as e:
            # Store error
            self.errors[node.id] = e
            if self.verbose:
                import traceback

                print(f"Operation {str(node.id)[:8]} failed: {e}")
                print(f"Traceback: {traceback.format_exc()}")

            # Re-raise if stop_on_error is enabled
            if self.stop_on_error:
                self.completion_events[node.id].set()
                raise

        finally:
            # Signal completion regardless of success/failure
            self.completion_events[node.id].set()

        return node

    async def _wait_for_dependencies(self, node: Node) -> None:
        """Wait for all predecessor operations to complete."""
        # Check for aggregation sources (special handling)
        is_aggregation = node.metadata.get("aggregation", False)
        params = node.metadata.get("parameters", {})

        if is_aggregation and isinstance(params, dict):
            aggregation_sources = params.get("aggregation_sources", [])
            if aggregation_sources:
                if self.verbose:
                    print(
                        f"Aggregation {str(node.id)[:8]} waiting for "
                        f"{len(aggregation_sources)} sources"
                    )

                # Wait for all aggregation sources
                for source_id_str in aggregation_sources:
                    for n_id, event in self.completion_events.items():
                        if str(n_id) == source_id_str:
                            await event.wait()
                            break

        # Wait for graph predecessors (normal dependency edges)
        predecessors = self.graph.get_predecessors(node)

        if self.verbose and predecessors:
            print(
                f"Operation {str(node.id)[:8]} waiting for {len(predecessors)} graph dependencies"
            )

        # Wait for all predecessors to complete
        for pred in predecessors:
            if pred.id in self.completion_events:
                await self.completion_events[pred.id].wait()

    def _prepare_operation_context(self, node: Node) -> None:
        """Prepare operation parameters with predecessor results."""
        predecessors = self.graph.get_predecessors(node)
        params = node.metadata.get("parameters", {})

        if not predecessors:
            # No dependencies - just add shared context if present
            if self.context and "context" not in params:
                params["context"] = self.context.copy()
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
                pred_name = pred.metadata.get("name", str(pred.id))
                pred_context[f"{pred_name}_result"] = result

        # Merge with shared execution context
        if self.context:
            pred_context.update(self.context)

        # Update operation parameters
        if "context" not in params:
            params["context"] = pred_context
        else:
            # Merge with existing context
            existing = params["context"]
            if isinstance(existing, dict):
                existing.update(pred_context)
            else:
                params["context"] = {
                    "original_context": existing,
                    **pred_context,
                }

        # Update node metadata
        node.metadata["parameters"] = params

        if self.verbose:
            print(f"Operation {str(node.id)[:8]} prepared with {len(pred_context)} context items")

    async def _invoke_operation(self, node: Node) -> None:
        """Create Operation from node and execute via IPU."""
        if self.verbose:
            print(f"Executing operation: {str(node.id)[:8]}")

        # Get branch for this operation
        branch = self.operation_branches.get(node.id)
        if branch is None:
            raise ValueError(f"No branch allocated for operation {node.id}")

        # Extract operation spec from node metadata
        operation_type = node.metadata.get("operation_type")
        parameters = node.metadata.get("parameters", {})

        if not operation_type:
            raise ValueError(f"Node {node.id} missing operation_type in metadata")

        # Strip flow-internal metadata from parameters before converting to typed Param
        flow_internal_fields = {"aggregation_sources", "aggregation_count"}
        if isinstance(parameters, dict):
            parameters = {k: v for k, v in parameters.items() if k not in flow_internal_fields}

        # Convert parameters to typed Param if possible
        typed_params = self.session._kwargs_to_param(operation_type, parameters)

        # Create Operation with session/branch context
        operation = Operation(  # type: ignore[call-arg]
            operation_type=operation_type,
            parameters=typed_params,
            session_id=self.session.id,
            branch_id=branch.id,
        )

        # Execute via IPU (provides context and validation)
        result = await self.ipu.execute(operation)

        # Store result
        self.results[node.id] = result

        if self.verbose:
            print(f"Completed operation: {str(node.id)[:8]}")


async def flow(
    session: Session,
    branch: Branch | str,
    graph: Graph,
    ipu: IPU,
    *,
    context: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute operation graph with dependency-aware scheduling.

    Args:
        session: Session for operation execution
        branch: Branch for message context
        graph: Operation graph (DAG) to execute
        ipu: IPU for validated execution
        context: Shared context for all operations
        max_concurrent: Max concurrent operations (None = unlimited)
        stop_on_error: Stop on first error
        verbose: Print progress

    Returns:
        Dictionary mapping operation names to their results.
    """
    executor = DependencyAwareExecutor(
        session=session,
        ipu=ipu,
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
    ipu: IPU,
    *,
    context: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
) -> AsyncGenerator[OperationResult, None]:
    """Execute operation graph, yielding results as operations complete."""
    executor = DependencyAwareExecutor(
        session=session,
        ipu=ipu,
        graph=graph,
        context=context,
        max_concurrent=max_concurrent,
        stop_on_error=stop_on_error,
        verbose=False,
        default_branch=branch,
    )

    async for result in executor.stream_execute():
        yield result
