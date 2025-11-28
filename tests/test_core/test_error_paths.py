# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive error path tests for lionpride."""

from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError

from lionpride.core import Element, Flow, Graph, Node, Pile, Progression
from lionpride.core.graph import Edge
from lionpride.errors import (
    ConfigurationError,
    ConnectionError,
    ExecutionError,
    ExistsError,
    LionprideError,
    NotFoundError,
    QueueFullError,
    TimeoutError,
    ValidationError,
)


class TestLionprideErrorBase:
    """Tests for LionprideError base class."""

    def test_default_message(self):
        """Default message is used when none provided."""
        err = LionprideError()
        assert err.message == "lionpride error"

    def test_custom_message(self):
        """Custom message overrides default."""
        err = LionprideError("custom error message")
        assert err.message == "custom error message"

    def test_details_dict(self):
        """Details dict is stored and accessible."""
        details = {"key": "value", "count": 42}
        err = LionprideError("test", details=details)
        assert err.details == details

    def test_retryable_default(self):
        """Default retryable flag from class attribute."""
        err = LionprideError()
        assert err.retryable is True

    def test_cause_chaining(self):
        """Cause exception is preserved for traceback."""
        original = ValueError("original error")
        err = LionprideError("wrapped error", cause=original)
        assert err.__cause__ is original

    def test_to_dict_serialization(self):
        """Error serializes to dict with all fields."""
        err = LionprideError("test message", details={"key": "value"}, retryable=False)
        data = err.to_dict()
        assert data["error"] == "LionprideError"
        assert data["message"] == "test message"
        assert data["retryable"] is False
        assert data["details"] == {"key": "value"}


class TestSpecializedErrors:
    """Tests for specialized error subclasses."""

    def test_validation_error_not_retryable(self):
        """ValidationError is not retryable by default."""
        err = ValidationError("validation failed")
        assert err.retryable is False

    def test_not_found_error_not_retryable(self):
        """NotFoundError is not retryable by default."""
        err = NotFoundError("item not found")
        assert err.retryable is False

    def test_exists_error_not_retryable(self):
        """ExistsError is not retryable by default."""
        err = ExistsError("item exists")
        assert err.retryable is False

    def test_timeout_error_retryable(self):
        """TimeoutError is retryable by default."""
        err = TimeoutError("operation timed out")
        assert err.retryable is True

    def test_connection_error_retryable(self):
        """ConnectionError is retryable by default."""
        err = ConnectionError("connection lost")
        assert err.retryable is True

    def test_inheritance_hierarchy(self):
        """All specialized errors inherit from LionprideError."""
        errors = [
            ValidationError(),
            ConfigurationError(),
            ExecutionError(),
            ConnectionError(),
            TimeoutError(),
            NotFoundError(),
            ExistsError(),
            QueueFullError(),
        ]
        for err in errors:
            assert isinstance(err, LionprideError)


class TestPileNotFoundErrors:
    """Tests for NotFoundError in Pile operations."""

    def test_remove_not_found(self):
        """Pile.remove() raises NotFoundError for missing item."""
        pile = Pile[Element]()
        fake_id = uuid4()
        with pytest.raises(NotFoundError) as exc_info:
            pile.remove(fake_id)
        assert str(fake_id) in str(exc_info.value)

    def test_get_not_found_no_default(self):
        """Pile.get() raises NotFoundError when no default provided."""
        pile = Pile[Element]()
        fake_id = uuid4()
        with pytest.raises(NotFoundError):
            pile.get(fake_id)

    def test_get_not_found_with_default(self):
        """Pile.get() returns default when provided."""
        pile = Pile[Element]()
        sentinel = object()
        result = pile.get(uuid4(), default=sentinel)
        assert result is sentinel


class TestPileExistsErrors:
    """Tests for ExistsError in Pile operations."""

    def test_add_duplicate(self):
        """Pile.add() raises ExistsError for duplicate item."""
        elem = Element()
        pile = Pile[Element]()
        pile.add(elem)
        with pytest.raises(ExistsError):
            pile.add(elem)


class TestPileTypeErrors:
    """Tests for TypeError in Pile operations."""

    def test_add_wrong_type_strict(self):
        """Pile.add() raises TypeError when strict_type and wrong type."""
        pile = Pile[Node](item_type=Node, strict_type=True)
        elem = Element()
        with pytest.raises(TypeError):
            pile.add(elem)


class TestProgressionErrors:
    """Tests for errors in Progression operations."""

    def test_pop_index_not_found_no_default(self):
        """Progression.pop() raises NotFoundError for invalid index."""
        prog = Progression(order=[uuid4()])
        with pytest.raises(NotFoundError):
            prog.pop(99)

    def test_popleft_empty(self):
        """Progression.popleft() raises NotFoundError when empty."""
        prog = Progression()
        with pytest.raises(NotFoundError):
            prog.popleft()

    def test_setitem_slice_non_list(self):
        """Progression[slice] = non-list raises TypeError."""
        prog = Progression(order=[uuid4(), uuid4()])
        with pytest.raises(TypeError):
            prog[0:1] = uuid4()


class TestFlowNotFoundErrors:
    """Tests for NotFoundError in Flow operations."""

    def test_init_referential_integrity_violation(self):
        """Flow raises NotFoundError if progression UUIDs not in items."""
        elem = Element()
        fake_id = uuid4()
        prog = Progression(order=[elem.id, fake_id], name="test")
        with pytest.raises(NotFoundError):
            Flow(items=[elem], progressions=[prog])

    def test_add_progression_missing_uuids(self):
        """Flow.add_progression() raises NotFoundError for missing UUIDs."""
        elem = Element()
        flow = Flow(items=[elem])
        fake_id = uuid4()
        prog = Progression(order=[fake_id], name="bad_prog")
        with pytest.raises(NotFoundError):
            flow.add_progression(prog)


class TestFlowExistsErrors:
    """Tests for ExistsError in Flow operations."""

    def test_add_progression_duplicate_name(self):
        """Flow.add_progression() raises ExistsError for duplicate name."""
        elem = Element()
        flow = Flow(items=[elem])
        prog1 = Progression(order=[elem.id], name="shared_name")
        prog2 = Progression(order=[elem.id], name="shared_name")
        flow.add_progression(prog1)
        with pytest.raises(ExistsError):
            flow.add_progression(prog2)


class TestGraphNotFoundErrors:
    """Tests for NotFoundError in Graph operations."""

    def test_add_edge_head_not_found(self):
        """Graph.add_edge() raises NotFoundError for missing head node."""
        graph = Graph()
        node = Node(content={"test": "data"})
        graph.add_node(node)
        fake_id = uuid4()
        edge = Edge(head=fake_id, tail=node.id)
        with pytest.raises(NotFoundError):
            graph.add_edge(edge)

    def test_remove_node_not_found(self):
        """Graph.remove_node() raises NotFoundError for missing node."""
        graph = Graph()
        with pytest.raises(NotFoundError):
            graph.remove_node(uuid4())


class TestGraphExistsErrors:
    """Tests for ExistsError in Graph operations."""

    def test_add_node_duplicate(self):
        """Graph.add_node() raises ExistsError for duplicate node."""
        graph = Graph()
        node = Node(content={"test": "data"})
        graph.add_node(node)
        with pytest.raises(ExistsError):
            graph.add_node(node)


class TestGraphValueErrors:
    """Tests for ValueError in Graph operations."""

    def test_topological_sort_cyclic_graph(self):
        """Graph.topological_sort() raises ValueError for cyclic graph."""
        graph = Graph()
        node1 = Node(content={"id": 1})
        node2 = Node(content={"id": 2})
        node3 = Node(content={"id": 3})
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        graph.add_edge(Edge(head=node1.id, tail=node2.id))
        graph.add_edge(Edge(head=node2.id, tail=node3.id))
        graph.add_edge(Edge(head=node3.id, tail=node1.id))
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()


class TestElementErrors:
    """Tests for errors in Element operations."""

    def test_to_dict_invalid_mode(self):
        """Element.to_dict() raises ValueError for invalid mode."""
        elem = Element()
        with pytest.raises(ValueError):
            elem.to_dict(mode="invalid")


class TestRetryableConsistency:
    """Tests for retryable flag consistency."""

    def test_transient_errors_are_retryable(self):
        """Transient errors are retryable."""
        assert ConnectionError().retryable is True
        assert TimeoutError().retryable is True
        assert QueueFullError().retryable is True

    def test_permanent_errors_are_not_retryable(self):
        """Permanent errors are not retryable."""
        assert ValidationError().retryable is False
        assert ConfigurationError().retryable is False
        assert NotFoundError().retryable is False
        assert ExistsError().retryable is False
