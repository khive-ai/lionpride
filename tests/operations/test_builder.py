# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage tests for OperationGraphBuilder.

Targets missing coverage lines:
- Lines 22-49: _resolve_branch_ref() function
- Lines 93-94: inherit_context with primary_dependency
- Lines 106-107: branch parameter in add()
- Line 149: Target not found in depends_on
- Line 155: Dependency not found in depends_on
- Lines 186-192: sequence() validation and execution
- Lines 210-215: parallel() validation
- Lines 232-234: get() ValueError path
- Line 248: get_by_id() None return path
- Lines 256-257: branch parameter in add_aggregation()
- Line 284: add_aggregation() ValueError for no sources
- Lines 302-305: add_aggregation() with inherit_context
- Line 314: add_aggregation() ValueError for source not found
- Lines 339-342: mark_executed() functionality
- Line 355: get_unexecuted_nodes() functionality
- Lines 384-388: clear() method
"""

from uuid import UUID, uuid4

import pytest

from lionpride.operations import Builder
from lionpride.operations.builder import _resolve_branch_ref
from lionpride.operations.node import create_operation


class TestResolveBranchRef:
    """Test _resolve_branch_ref() function (lines 22-49)."""

    def test_uuid_input_returns_uuid(self):
        """Test UUID input is returned as-is (line 32-33)."""
        test_uuid = uuid4()
        result = _resolve_branch_ref(test_uuid)
        assert result == test_uuid
        assert isinstance(result, UUID)

    def test_object_with_id_attribute(self):
        """Test object with id attribute returns its UUID (lines 36-37)."""

        class MockBranch:
            def __init__(self):
                self.id = uuid4()

        branch = MockBranch()
        result = _resolve_branch_ref(branch)
        assert result == branch.id
        assert isinstance(result, UUID)

    def test_uuid_string_converts_to_uuid(self):
        """Test valid UUID string is converted (lines 40-43)."""
        test_uuid = uuid4()
        uuid_str = str(test_uuid)
        result = _resolve_branch_ref(uuid_str)
        assert result == test_uuid
        assert isinstance(result, UUID)

    def test_branch_name_string_returned_stripped(self):
        """Test non-UUID string is returned as stripped name (lines 46-47)."""
        result = _resolve_branch_ref("  main  ")
        assert result == "main"
        assert isinstance(result, str)

    def test_branch_name_without_whitespace(self):
        """Test branch name without extra whitespace."""
        result = _resolve_branch_ref("feature/test-branch")
        assert result == "feature/test-branch"
        assert isinstance(result, str)

    def test_empty_string_raises_value_error(self):
        """Test empty string raises ValueError (line 49)."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref("")

    def test_whitespace_only_string_raises_value_error(self):
        """Test whitespace-only string raises ValueError (line 49)."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref("   ")

    def test_none_raises_value_error(self):
        """Test None raises ValueError (line 49)."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref(None)

    def test_integer_raises_value_error(self):
        """Test integer raises ValueError (line 49)."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref(123)

    def test_list_raises_value_error(self):
        """Test list raises ValueError (line 49)."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref(["main"])


class TestBuilderBranchParameter:
    """Test branch parameter in add() and add_aggregation() (lines 106-107, 256-257)."""

    def test_add_with_uuid_branch(self):
        """Test add() with UUID branch sets metadata (lines 106-107)."""
        builder = Builder()
        branch_uuid = uuid4()
        builder.add("task1", "generate", {"instruction": "First"}, branch=branch_uuid)

        task1 = builder.get("task1")
        assert task1.metadata.get("branch") == branch_uuid

    def test_add_with_string_branch_name(self):
        """Test add() with string branch name sets metadata (lines 106-107)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"}, branch="main")

        task1 = builder.get("task1")
        assert task1.metadata.get("branch") == "main"

    def test_add_with_branch_object(self):
        """Test add() with Branch-like object extracts UUID (lines 106-107)."""

        class MockBranch:
            def __init__(self):
                self.id = uuid4()

        builder = Builder()
        branch = MockBranch()
        builder.add("task1", "generate", {"instruction": "First"}, branch=branch)

        task1 = builder.get("task1")
        assert task1.metadata.get("branch") == branch.id

    def test_add_without_branch_no_metadata(self):
        """Test add() without branch doesn't set metadata."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        task1 = builder.get("task1")
        assert "branch" not in task1.metadata

    def test_add_aggregation_with_uuid_branch(self):
        """Test add_aggregation() with UUID branch sets metadata (lines 256-257)."""
        builder = Builder()
        builder.add("source1", "generate", {"instruction": "First"})
        builder.add("source2", "generate", {"instruction": "Second"})

        branch_uuid = uuid4()
        builder.add_aggregation(
            "agg",
            "operate",
            {"instruction": "Aggregate"},
            source_names=["source1", "source2"],
            branch=branch_uuid,
        )

        agg_op = builder.get("agg")
        assert agg_op.metadata.get("branch") == branch_uuid

    def test_add_aggregation_with_string_branch(self):
        """Test add_aggregation() with string branch name sets metadata (lines 256-257)."""
        builder = Builder()
        builder.add("source1", "generate", {"instruction": "First"})
        builder.add("source2", "generate", {"instruction": "Second"})

        builder.add_aggregation(
            "agg",
            "operate",
            {"instruction": "Aggregate"},
            source_names=["source1", "source2"],
            branch="feature-branch",
        )

        agg_op = builder.get("agg")
        assert agg_op.metadata.get("branch") == "feature-branch"

    def test_add_aggregation_without_branch_no_metadata(self):
        """Test add_aggregation() without branch doesn't set metadata."""
        builder = Builder()
        builder.add("source1", "generate", {"instruction": "First"})
        builder.add("source2", "generate", {"instruction": "Second"})

        builder.add_aggregation(
            "agg",
            "operate",
            {"instruction": "Aggregate"},
            source_names=["source1", "source2"],
        )

        agg_op = builder.get("agg")
        assert "branch" not in agg_op.metadata

    def test_multi_branch_workflow(self):
        """Test multi-branch workflow with operations on different branches."""
        builder = Builder()
        branch1_id = uuid4()
        branch2_id = uuid4()

        # Operations on branch 1
        builder.add("extract", "generate", {"instruction": "Extract"}, branch=branch1_id)
        builder.add("transform", "operate", {"instruction": "Transform"}, branch=branch1_id)

        # Operations on branch 2
        builder.add("analyze", "generate", {"instruction": "Analyze"}, branch=branch2_id)

        # Aggregation from both branches
        builder.add_aggregation(
            "combine",
            "operate",
            {"instruction": "Combine"},
            source_names=["transform", "analyze"],
            branch="main",
        )

        # Verify branch assignments
        assert builder.get("extract").metadata["branch"] == branch1_id
        assert builder.get("transform").metadata["branch"] == branch1_id
        assert builder.get("analyze").metadata["branch"] == branch2_id
        assert builder.get("combine").metadata["branch"] == "main"

        # Graph should still be valid
        graph = builder.build()
        assert len(graph.nodes) == 4


class TestBuilderParameterMutation:
    """Test that add() doesn't mutate caller's parameters."""

    def test_add_with_dict_params(self):
        """Test add() passes dict parameters."""
        builder = Builder()
        original_params = {"instruction": "First", "model": "default"}

        builder.add("task1", "generate", original_params)

        task1 = builder.get("task1")
        # Dict params are passed directly
        assert task1.parameters == original_params

    def test_add_with_typed_params(self):
        """Test add() passes typed params as-is."""
        from lionpride.operations.operate.types import GenerateParams

        builder = Builder()
        params = GenerateParams(instruction="Test", imodel="test_model")

        builder.add("task1", "generate", params)

        task1 = builder.get("task1")
        # Typed params are passed directly
        assert task1.parameters is params
        assert task1.parameters.instruction == "Test"


class TestBuilderContextInheritance:
    """Test context inheritance features (lines 92-94, 301-305)."""

    def test_inherit_context_from_dependency(self):
        """Test inherit_context sets metadata correctly (lines 92-94)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add(
            "task2",
            "generate",
            {"instruction": "Second"},
            depends_on=["task1"],
            inherit_context=True,
        )

        task2 = builder.get("task2")
        assert task2.metadata.get("inherit_context") is True
        assert "primary_dependency" in task2.metadata
        assert task2.metadata["primary_dependency"] == builder._nodes["task1"].id

    def test_inherit_context_without_dependency_no_effect(self):
        """Test inherit_context without depends_on has no effect."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"}, inherit_context=True)

        task1 = builder.get("task1")
        # Should not set inherit_context if no dependencies
        assert task1.metadata.get("inherit_context") is None

    def test_add_aggregation_with_inherit_context(self):
        """Test add_aggregation with inherit_context (lines 301-305)."""
        builder = Builder()
        builder.add("source1", "generate", {"instruction": "First"})
        builder.add("source2", "generate", {"instruction": "Second"})
        builder.add_aggregation(
            "agg",
            "operate",
            {"instruction": "Aggregate"},
            source_names=["source1", "source2"],
            inherit_context=True,
            inherit_from_source=1,
        )

        agg_op = builder.get("agg")
        assert agg_op.metadata.get("inherit_context") is True
        assert agg_op.metadata["primary_dependency"] == builder._nodes["source2"].id
        assert agg_op.metadata["inherit_from_source"] == 1

    def test_add_aggregation_inherit_from_source_clamped(self):
        """Test inherit_from_source is clamped to valid range (line 303)."""
        builder = Builder()
        builder.add("source1", "generate", {"instruction": "First"})
        builder.add("source2", "generate", {"instruction": "Second"})
        builder.add_aggregation(
            "agg",
            "operate",
            {"instruction": "Aggregate"},
            source_names=["source1", "source2"],
            inherit_context=True,
            inherit_from_source=10,  # Out of range - should clamp to len(sources) - 1 = 1
        )

        agg_op = builder.get("agg")
        # Should clamp to len(sources) - 1 = 1 (source2)
        assert agg_op.metadata["primary_dependency"] == builder._nodes["source2"].id
        # The actual clamped value is stored
        assert agg_op.metadata["inherit_from_source"] == 1


class TestBuilderDependencyValidation:
    """Test dependency validation error paths (lines 149, 155, 314)."""

    def test_depends_on_target_not_found(self):
        """Test depends_on raises ValueError when target not found (line 149)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="Target operation 'nonexistent' not found"):
            builder.depends_on("nonexistent", "task1")

    def test_depends_on_dependency_not_found(self):
        """Test depends_on raises ValueError when dependency not found (line 155)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="Dependency operation 'nonexistent' not found"):
            builder.depends_on("task1", "nonexistent")

    def test_depends_on_multiple_dependencies_one_missing(self):
        """Test depends_on with multiple dependencies where one is missing."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        with pytest.raises(ValueError, match="Dependency operation 'missing' not found"):
            builder.depends_on("task2", "task1", "missing")

    def test_add_aggregation_source_not_found(self):
        """Test add_aggregation raises ValueError when source not found (line 314)."""
        builder = Builder()
        builder.add("source1", "generate", {"instruction": "First"})

        # Using BaseModel params to avoid dict parameter path, reaching line 314 directly
        from pydantic import BaseModel, Field

        class AggParams(BaseModel):
            instruction: str = Field(default="Aggregate")

        with pytest.raises(ValueError, match="Source operation 'missing' not found"):
            builder.add_aggregation(
                "agg",
                "operate",
                AggParams(),  # Use BaseModel to skip dict path at line 289-293
                source_names=["source1", "missing"],
            )


class TestBuilderSequenceMethod:
    """Test sequence() method (lines 186-192)."""

    def test_sequence_requires_at_least_two_operations(self):
        """Test sequence() raises ValueError with < 2 operations (lines 186-187)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="sequence requires at least 2 operations"):
            builder.sequence("task1")

    def test_sequence_empty(self):
        """Test sequence() with empty arguments."""
        builder = Builder()

        with pytest.raises(ValueError, match="sequence requires at least 2 operations"):
            builder.sequence()

    def test_sequence_creates_dependency_chain(self):
        """Test sequence() creates proper dependency chain (lines 189-191)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        builder.sequence("task1", "task2", "task3")

        # Verify chain: task1 -> task2 -> task3
        task1 = builder.get("task1")
        task2 = builder.get("task2")
        task3 = builder.get("task3")

        successors_1 = builder.graph.get_successors(task1)
        successors_2 = builder.graph.get_successors(task2)

        assert task2 in successors_1
        assert task3 in successors_2

    def test_sequence_with_custom_label(self):
        """Test sequence() respects custom labels."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        # Sequence task2 and task3 with custom label
        builder.sequence("task2", "task3", label=["custom"])

        # Find edge and verify label using Pile filter
        task2 = builder.get("task2")
        task3 = builder.get("task3")
        matching_edges = builder.graph.edges[
            lambda e: e.head == task2.id and e.tail == task3.id and "custom" in e.label
        ]
        assert len(matching_edges) >= 1
        assert "custom" in matching_edges[0].label


class TestBuilderParallelMethod:
    """Test parallel() method (lines 209-215)."""

    def test_parallel_operation_not_found(self):
        """Test parallel() raises ValueError when operation not found (lines 210-212)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="Operation 'nonexistent' not found"):
            builder.parallel("task1", "nonexistent")

    def test_parallel_all_operations_not_found(self):
        """Test parallel() with no valid operations."""
        builder = Builder()

        with pytest.raises(ValueError, match="Operation 'missing1' not found"):
            builder.parallel("missing1", "missing2")

    def test_parallel_creates_no_edges(self):
        """Test parallel() doesn't create dependencies."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        # Clear any auto-created edges (from sequential addition)
        builder.graph.edges.clear()

        builder.parallel("task1", "task2", "task3")

        # Should have no edges
        assert len(builder.graph.edges) == 0


class TestBuilderGetMethods:
    """Test get() and get_by_id() methods (lines 232-234, 248)."""

    def test_get_operation_not_found(self):
        """Test get() raises ValueError when operation not found (lines 232-234)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="Operation 'nonexistent' not found"):
            builder.get("nonexistent")

    def test_get_by_id_not_found(self):
        """Test get_by_id() returns None when operation not found (line 248)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        random_uuid = uuid4()
        result = builder.get_by_id(random_uuid)
        assert result is None

    def test_get_by_id_found(self):
        """Test get_by_id() returns operation when found."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        task1 = builder.get("task1")
        retrieved = builder.get_by_id(task1.id)
        assert retrieved is task1


class TestBuilderAggregationValidation:
    """Test add_aggregation() validation (line 284)."""

    def test_add_aggregation_no_sources_no_heads(self):
        """Test add_aggregation raises ValueError when no sources available (line 284)."""
        builder = Builder()

        with pytest.raises(ValueError, match="No source operations for aggregation"):
            builder.add_aggregation("agg", "operate", {"instruction": "Aggregate"})

    def test_add_aggregation_uses_current_heads_by_default(self):
        """Test add_aggregation uses _current_heads when source_names not provided."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        # Current heads should be ["task2"] after sequential additions
        # But we need both as sources, so manually set current_heads
        builder._current_heads = ["task1", "task2"]

        builder.add_aggregation("agg", "operate", {"instruction": "Aggregate"})

        agg_op = builder.get("agg")
        assert agg_op.metadata.get("aggregation") is True

        # Verify both sources connected
        task1 = builder.get("task1")
        task2 = builder.get("task2")
        preds = builder.graph.get_predecessors(agg_op)
        assert task1 in preds
        assert task2 in preds


class TestBuilderExecutionTracking:
    """Test mark_executed() and get_unexecuted_nodes() (lines 339-342, 355)."""

    def test_mark_executed_adds_to_set(self):
        """Test mark_executed() adds operations to _executed set (lines 339-341)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        assert len(builder._executed) == 0

        builder.mark_executed("task1")
        assert len(builder._executed) == 1
        assert builder._nodes["task1"].id in builder._executed

    def test_mark_executed_multiple(self):
        """Test mark_executed() with multiple operations."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        builder.mark_executed("task1", "task2")
        assert len(builder._executed) == 2
        assert builder._nodes["task1"].id in builder._executed
        assert builder._nodes["task2"].id in builder._executed
        assert builder._nodes["task3"].id not in builder._executed

    def test_mark_executed_nonexistent_operation_ignored(self):
        """Test mark_executed() silently ignores nonexistent operations (line 340)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        # Should not raise error
        builder.mark_executed("task1", "nonexistent", "also_missing")
        assert len(builder._executed) == 1

    def test_get_unexecuted_nodes_empty(self):
        """Test get_unexecuted_nodes() when all executed (line 355)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        builder.mark_executed("task1", "task2")
        unexecuted = builder.get_unexecuted_nodes()
        assert len(unexecuted) == 0

    def test_get_unexecuted_nodes_partial(self):
        """Test get_unexecuted_nodes() returns only unexecuted operations."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        builder.mark_executed("task1")
        unexecuted = builder.get_unexecuted_nodes()

        assert len(unexecuted) == 2
        names = [op.metadata["name"] for op in unexecuted]
        assert "task1" not in names
        assert "task2" in names
        assert "task3" in names

    def test_get_unexecuted_nodes_all_unexecuted(self):
        """Test get_unexecuted_nodes() when nothing executed."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        unexecuted = builder.get_unexecuted_nodes()
        assert len(unexecuted) == 2


class TestBuilderClearMethod:
    """Test clear() method (lines 384-388)."""

    def test_clear_resets_all_state(self):
        """Test clear() resets all builder state (lines 384-388)."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.mark_executed("task1")

        # Verify state before clear
        assert len(builder._nodes) == 2
        assert len(builder.graph.nodes) == 2
        assert len(builder._executed) == 1
        assert len(builder._current_heads) == 1

        # Clear
        builder.clear()

        # Verify all state reset
        assert len(builder._nodes) == 0
        assert len(builder.graph.nodes) == 0
        assert len(builder._executed) == 0
        assert len(builder._current_heads) == 0

    def test_clear_allows_reuse(self):
        """Test builder can be reused after clear()."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        graph1 = builder.build()

        builder.clear()
        builder.add("task2", "generate", {"instruction": "Second"})
        graph2 = builder.build()

        # Graphs should be different
        assert graph1 is not graph2
        assert len(graph2.nodes) == 1
        assert "task2" in builder._nodes

    def test_clear_returns_self_for_chaining(self):
        """Test clear() returns self for method chaining."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        result = builder.clear()
        assert result is builder

        # Can chain immediately
        builder.clear().add("task2", "generate", {"instruction": "Second"})
        assert "task2" in builder._nodes


class TestBuilderEdgeCases:
    """Test additional edge cases and integration scenarios."""

    def test_depends_on_with_custom_label(self):
        """Test depends_on() with custom edge labels."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        # Create dependency with custom label between task3 and task1
        builder.depends_on("task3", "task1", label=["conditional", "optional"])

        # Find edge and verify labels using Pile filter
        task1 = builder.get("task1")
        task3 = builder.get("task3")
        matching_edges = builder.graph.edges[
            lambda e: e.head == task1.id and e.tail == task3.id and "conditional" in e.label
        ]
        assert len(matching_edges) >= 1
        assert "conditional" in matching_edges[0].label
        assert "optional" in matching_edges[0].label

    def test_add_aggregation_with_pydantic_parameters(self):
        """Test add_aggregation() with Pydantic BaseModel parameters."""
        from pydantic import BaseModel, Field

        class AggParams(BaseModel):
            instruction: str = Field(..., description="Aggregation instruction")
            strategy: str = Field(default="concat")

        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        params = AggParams(instruction="Combine results")
        builder.add_aggregation("agg", "operate", params, source_names=["task1", "task2"])

        agg_op = builder.get("agg")
        # Should store BaseModel as-is (not converted to dict)
        assert isinstance(agg_op.parameters, BaseModel)

    def test_repr_output(self):
        """Test __repr__ provides useful information."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=["task1"])
        builder.mark_executed("task1")

        repr_str = repr(builder)
        assert "operations=2" in repr_str
        assert "edges=1" in repr_str
        assert "executed=1" in repr_str

    def test_add_operation_with_metadata(self):
        """Test add() with custom metadata via kwargs."""
        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {"instruction": "First"},
            metadata={"priority": "high", "tags": ["important"]},
        )

        task1 = builder.get("task1")
        assert task1.metadata.get("priority") == "high"
        assert task1.metadata.get("tags") == ["important"]
        assert task1.metadata.get("name") == "task1"  # Should preserve name

    def test_incremental_building_workflow(self):
        """Test incremental build -> execute -> expand workflow."""
        builder = Builder()

        # Phase 1: Initial graph
        builder.add("extract", "generate", {"instruction": "Extract data"})
        graph1 = builder.build()
        assert len(graph1.nodes) == 1

        # Simulate execution
        builder.mark_executed("extract")

        # Phase 2: Expand graph
        builder.add("analyze", "operate", {"instruction": "Analyze"}, depends_on=["extract"])
        graph2 = builder.build()
        assert len(graph2.nodes) == 2

        # Verify only new operation is unexecuted
        unexecuted = builder.get_unexecuted_nodes()
        assert len(unexecuted) == 1
        assert unexecuted[0].metadata["name"] == "analyze"
