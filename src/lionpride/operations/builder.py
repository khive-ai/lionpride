# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from lionpride.core import Edge, Graph

from .node import Operation, OperationType

__all__ = ("Builder", "OperationGraphBuilder")


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
        parameters: dict[str, Any] | BaseModel | None = None,
        depends_on: list[str] | None = None,
        inherit_context: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> OperationGraphBuilder:
        """Add operation to graph. Returns self for chaining.

        Args:
            name: Operation name for reference
            operation: Operation type (communicate, operate, react, generate)
            parameters: Operation parameters dict or model
            depends_on: List of operation names this depends on
            inherit_context: Whether to inherit context from dependencies
            metadata: Optional metadata dict for the Operation node
            **kwargs: Additional params merged into parameters (e.g., generate={...})
        """
        if name in self._nodes:
            raise ValueError(f"Operation with name '{name}' already exists")

        # Merge kwargs into parameters (allows builder.add(..., generate={...}))
        params = parameters if isinstance(parameters, dict) else {}
        if parameters and isinstance(parameters, BaseModel):
            params = parameters.model_dump()
        params.update(kwargs)

        # Create unified Operation node (Node + Event)
        op = Operation(
            operation_type=operation,
            parameters=params,
            metadata=metadata or {},
        )

        # Store name in metadata for reference
        op.metadata["name"] = name

        # Store context inheritance strategy
        if inherit_context and depends_on:
            op.metadata["inherit_context"] = True
            op.metadata["primary_dependency"] = self._nodes[depends_on[0]].id

        # Add to graph
        self.graph.add_node(op)

        # Track by name
        self._nodes[name] = op

        # Handle dependencies
        if depends_on:
            for dep_name in depends_on:
                if dep_name not in self._nodes:
                    raise ValueError(f"Dependency '{dep_name}' not found")
                dep_node = self._nodes[dep_name]
                edge = Edge(head=dep_node.id, tail=op.id, label=["depends_on"])
                self.graph.add_edge(edge)
        elif self._current_heads:
            # Auto-link from current heads if no explicit dependencies
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
        parameters: dict[str, Any] | BaseModel | None = None,
        source_names: list[str] | None = None,
        inherit_context: bool = False,
        inherit_from_source: int = 0,
        **kwargs,
    ) -> OperationGraphBuilder:
        """Add aggregation operation that collects from multiple sources."""
        sources = source_names or self._current_heads
        if not sources:
            raise ValueError("No source operations for aggregation")

        # Validate all sources exist before proceeding
        for source_name in sources:
            if source_name not in self._nodes:
                raise ValueError(f"Source operation '{source_name}' not found")

        # Handle parameters - keep as BaseModel if passed that way
        if parameters is None:
            params = {}
        elif isinstance(parameters, BaseModel):
            params = parameters  # Keep as BaseModel
        else:
            params = dict(parameters)  # Copy dict

        # Merge kwargs into params if it's a dict
        if isinstance(params, dict):
            params.update(kwargs)

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
