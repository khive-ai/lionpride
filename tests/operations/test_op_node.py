# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for operations/node.py module.

Targets lines 126-127, 151 for 100% coverage.
"""

import pytest

from lionpride.operations.node import Operation, create_operation


class TestNodeCoverage:
    """Test node.py uncovered lines."""

    def test_operation_repr_bound(self, session_with_model):
        """Test lines 126-127: __repr__ when operation is bound."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        op = Operation(operation_type="generate", parameters={"instruction": "Test"})
        op.bind(session, branch)

        repr_str = repr(op)

        # Verify repr shows 'bound' state
        assert "generate" in repr_str
        assert "bound" in repr_str
        assert "pending" in repr_str.lower()

    def test_operation_repr_unbound(self):
        """Test lines 126-127: __repr__ when operation is unbound."""
        op = Operation(operation_type="operate", parameters={"instruction": "Test"})

        repr_str = repr(op)

        # Verify repr shows 'unbound' state
        assert "operate" in repr_str
        assert "unbound" in repr_str

    def test_create_operation_no_type_raises_error(self):
        """Test line 151: create_operation with no type raises ValueError."""
        with pytest.raises(ValueError, match=r"operation_type.*required"):
            create_operation(operation_type=None, parameters={})

    def test_create_operation_legacy_kwarg(self):
        """Test create_operation with legacy 'operation=' kwarg."""
        op = create_operation(operation="generate", parameters={"instruction": "Test"})

        assert op.operation_type == "generate"
        assert op.parameters == {"instruction": "Test"}

    def test_create_operation_with_metadata(self):
        """Test create_operation with metadata kwargs."""
        op = create_operation(
            operation_type="communicate",
            parameters={"instruction": "Hello"},
            metadata={"name": "test_op"},
        )

        assert op.operation_type == "communicate"
        assert op.metadata.get("name") == "test_op"
