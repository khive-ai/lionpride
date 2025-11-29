# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from lionpride.core import Edge, Graph
from lionpride.core._utils import to_uuid

from .node import Operation, OperationType

if TYPE_CHECKING:
    from lionpride.session import Branch

__all__ = ("Builder", "OperationGraphBuilder")


def _resolve_branch_ref(branch: Any) -> UUID | str:
    """Resolve branch reference to UUID or name string.

    Args:
        branch: Branch object, UUID, or name string

    Returns:
        UUID if resolvable, otherwise the branch name string
    """
    # If it's already a UUID, return it
    if isinstance(branch, UUID):
        return branch

    # If it has an id attribute (Branch object), get its UUID
    if hasattr(branch, "id"):
        return branch.id

    # Try to convert string to UUID
    try:
        return to_uuid(branch)
    except (ValueError, TypeError):
        pass

    # Keep as name string if non-empty
    if isinstance(branch, str) and branch.strip():
        return branch.strip()

    raise ValueError(f"Invalid branch reference: {branch}")


class OperationGraphBuilder:
    """Fluent builder for operation graphs (DAGs)."""

    def __init__(self, graph: Graph | None = None):
        """Initialize builder with optional existing graph."""
        self.graph = graph or Graph()
        self._nodes: dict[str, Operation] = {}
        self._executed: set[UUID] = set()  # Track executed operations
        self._current_heads: list[str] = []  # Current head nodes for linking

    def add(
        self,
        name: str,
        operation: OperationType | str,
        parameters: dict[str, Any] | Any | None = None,
        depends_on: list[str] | None = None,
        branch: str | UUID | Branch | None = None,
        inherit_context: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> OperationGraphBuilder:
        """Add operation to graph. Returns self for chaining.

        Args:
            name: Operation name for reference
            operation: Operation type (communicate, operate, react, generate)
            parameters: Operation parameters (typed params or dict)
            depends_on: List of operation names this depends on
            branch: Branch object, UUID, or name to execute this operation on (None = default branch)
            inherit_context: Whether to inherit context from dependencies
            metadata: Optional metadata dict for the Operation node
        """
        if name in self._nodes:
            raise ValueError(f"Operation with name '{name}' already exists")

        # Pass typed params directly - no conversion
        # Typed params (GenerateParams, OperateParams, etc.) are expected
        params = parameters

        # Create unified Operation node (Node + Event)
        op = Operation(
            operation_type=operation,
            parameters=params,
            metadata=metadata or {},
        )

        # Store name in metadata for reference
        op.metadata["name"] = name

        # Store branch assignment for execution (UUID or name string)
        if branch is not None:
            op.metadata["branch"] = _resolve_branch_ref(branch)

        # Store context inheritance strategy
        if inherit_context and depends_on:
            op.metadata["inherit_context"] = True
            op.metadata["primary_dependency"] = self._nodes[depends_on[0]].id

        # Add to graph
        self.graph.add_node(op)

        # Track by name
        self._nodes[name] = op

        # Handle dependencies
        # Empty list [] = no dependencies (independent operation)
        # None = auto-link to current heads (sequential execution)
        if depends_on is not None:
            for dep_name in depends_on:
                if dep_name not in self._nodes:
                    raise ValueError(f"Dependency '{dep_name}' not found")
                dep_node = self._nodes[dep_name]
                edge = Edge(head=dep_node.id, tail=op.id, label=["depends_on"])
                self.graph.add_edge(edge)
        elif self._current_heads:
            # Auto-link from current heads only if depends_on is None
            for head_name in self._current_heads:
                if head_name in self._nodes:
                    head_node = self._nodes[head_name]
                    edge = Edge(head=head_node.id, tail=op.id, label=["sequential"])
                    self.graph.add_edge(edge)

        # Update current heads
        self._current_heads = [name]

        return self

    def depends_on(
        self,
        target: str,
        *dependencies: str,
        label: list[str] | None = None,
    ) -> OperationGraphBuilder:
        """Add dependency relationships. Returns self for chaining."""
        if target not in self._nodes:
            raise ValueError(f"Target operation '{target}' not found")

        target_node = self._nodes[target]

        for dep_name in dependencies:
            if dep_name not in self._nodes:
                raise ValueError(f"Dependency operation '{dep_name}' not found")

            dep_node = self._nodes[dep_name]

            # Create edge: dependency -> target
            edge = Edge(
                head=dep_node.id,
                tail=target_node.id,
                label=label or [],
            )
            self.graph.add_edge(edge)

        return self

    def sequence(self, *operations: str, label: list[str] | None = None) -> OperationGraphBuilder:
        """Create sequential dependency chain. Returns self for chaining."""
        if len(operations) < 2:
            raise ValueError("sequence requires at least 2 operations")

        for i in range(len(operations) - 1):
            self.depends_on(operations[i + 1], operations[i], label=label)

        return self

    def parallel(self, *operations: str) -> OperationGraphBuilder:
        """Mark operations as parallel (no-op for clarity). Returns self."""
        # Verify operations exist
        for name in operations:
            if name not in self._nodes:
                raise ValueError(f"Operation '{name}' not found")

        # No edges needed - operations are naturally parallel
        return self

    def get(self, name: str) -> Operation:
        """Get operation by name."""
        if name not in self._nodes:
            raise ValueError(f"Operation '{name}' not found")
        return self._nodes[name]

    def get_by_id(self, operation_id: UUID) -> Operation | None:
        """Get operation by UUID, or None if not found."""
        return self.graph.nodes.get(operation_id, None)

    def add_aggregation(
        self,
        name: str,
        operation: OperationType | str,
        parameters: dict[str, Any] | Any | None = None,
        source_names: list[str] | None = None,
        branch: str | UUID | Branch | None = None,
        inherit_context: bool = False,
        inherit_from_source: int = 0,
    ) -> OperationGraphBuilder:
        """Add aggregation operation that collects from multiple sources.

        Args:
            name: Operation name for reference
            operation: Operation type
            parameters: Operation parameters (typed params or dict)
            source_names: List of source operation names to aggregate from
            branch: Branch object, UUID, or name to execute this operation on (None = default branch)
            inherit_context: Whether to inherit context from sources
            inherit_from_source: Index of source to inherit context from
        """
        sources = source_names or self._current_heads
        if not sources:
            raise ValueError("No source operations for aggregation")

        # Validate all sources exist before proceeding
        for source_name in sources:
            if source_name not in self._nodes:
                raise ValueError(f"Source operation '{source_name}' not found")

        # Pass typed params directly - no conversion
        params = parameters

        # Create operation node
        op = Operation(
            operation_type=operation,
            parameters=params,
            metadata={},
        )
        op.metadata["name"] = name
        op.metadata["aggregation"] = True
        # Store aggregation sources in metadata, not parameters
        op.metadata["aggregation_sources"] = [str(self._nodes[s].id) for s in sources]
        op.metadata["aggregation_count"] = len(sources)

        # Store branch assignment for execution (UUID or name string)
        if branch is not None:
            op.metadata["branch"] = _resolve_branch_ref(branch)

        # Store context inheritance for aggregations
        if inherit_context and sources:
            op.metadata["inherit_context"] = True
            source_idx = min(inherit_from_source, len(sources) - 1)
            op.metadata["primary_dependency"] = self._nodes[sources[source_idx]].id
            op.metadata["inherit_from_source"] = source_idx

        # Add to graph
        self.graph.add_node(op)
        self._nodes[name] = op

        # Connect all sources (already validated above)
        for source_name in sources:
            source_node = self._nodes[source_name]
            edge = Edge(head=source_node.id, tail=op.id, label=["aggregate"])
            self.graph.add_edge(edge)

        # Update current heads
        self._current_heads = [name]

        return self

    def mark_executed(self, *names: str) -> OperationGraphBuilder:
        """Mark operations as executed for incremental building."""
        for name in names:
            if name in self._nodes:
                self._executed.add(self._nodes[name].id)
        return self

    def get_unexecuted_nodes(self) -> list[Operation]:
        """Get operations that haven't been executed yet."""
        return [op for op in self._nodes.values() if op.id not in self._executed]

    def build(self) -> Graph:
        """Build and validate operation graph (must be DAG)."""
        # Validate DAG
        if not self.graph.is_acyclic():
            raise ValueError("Operation graph has cycles - must be a DAG")

        return self.graph

    def clear(self) -> OperationGraphBuilder:
        """Clear all operations and start fresh."""
        self.graph = Graph()
        self._nodes = {}
        self._executed = set()
        self._current_heads = []
        return self

    def __repr__(self) -> str:
        return (
            f"OperationGraphBuilder("
            f"operations={len(self._nodes)}, "
            f"edges={len(self.graph.edges)}, "
            f"executed={len(self._executed)})"
        )


# Alias for convenience
Builder = OperationGraphBuilder
