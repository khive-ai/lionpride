# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Root conftest.py - shared test fixtures for all lionpride tests.

This module provides reusable test fixtures, factories, and strategies for
testing lionpride primitives. All fixtures are automatically available to
all tests in the tests/ directory.

Usage:
    # Fixtures are automatically injected by pytest
    def test_with_element(mock_element):
        assert mock_element.id is not None

    def test_with_pile(test_pile):
        assert len(test_pile) == 5

Note: Test classes (TestElement, SimpleTestEvent, etc.) are imported from
lionpride._testing to ensure proper lion_class serialization paths.
"""

# Add tests/ directory to sys.path to allow `from conftest import ...` in test files
import sys
from pathlib import Path

_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from typing import Any

import pytest
from pydantic import BaseModel

# Import test classes from _testing module for proper lion_class serialization paths
from lionpride._testing import (
    FailingTestEvent,
    SimpleTestEvent,
    SlowTestEvent,
    StreamingTestEvent,
    TestElement,
    TestProcessor,
)
from lionpride.core import Edge, Element, Event, Flow, Graph, Node, Pile, Progression

__all__ = (
    "FailingTestEvent",
    "SimpleTestEvent",
    "SlowTestEvent",
    "StreamingTestEvent",
    "TestElement",
    "TestProcessor",
    "create_cyclic_graph",
    "create_dag_graph",
    "create_empty_graph",
    "create_simple_graph",
    "create_test_elements",
    "create_test_flow",
    "create_test_nodes",
    "create_test_pile",
    "create_test_progression",
    "create_typed_pile",
    "mock_element",
    "mock_node",
)


# =============================================================================
# Factory Functions
# =============================================================================


def mock_element(
    *,
    value: int = 0,
    name: str = "test",
    metadata: dict[str, Any] | None = None,
) -> TestElement:
    """Create a single mock Element for testing (factory function, not fixture)."""
    elem = TestElement(value=value, name=name)
    if metadata:
        elem.metadata.update(metadata)
    return elem


def mock_node(
    *,
    content: dict[str, Any] | BaseModel | None = None,
    value: str = "test",
) -> Node:
    """Create a single mock Node for testing (factory function, not fixture)."""
    if content is None:
        content = {"value": value}
    return Node(content=content)


def create_test_elements(count: int = 5, start_value: int = 0) -> list[TestElement]:
    """Create a list of test Elements."""
    return [TestElement(value=start_value + i, name=f"element_{i}") for i in range(count)]


def create_test_nodes(count: int = 5) -> list[Node]:
    """Create a list of test Nodes."""
    return [Node(content={"value": f"node_{i}"}) for i in range(count)]


def create_test_pile(
    count: int = 5,
    *,
    item_type: type | None = None,
    strict_type: bool = False,
) -> Pile[TestElement]:
    """Create a test Pile with mock elements."""
    items = create_test_elements(count=count)
    return Pile(items=items, item_type=item_type, strict_type=strict_type)


def create_typed_pile(
    items: list[Element],
    *,
    item_type: type | set[type] | None = None,
    strict_type: bool = False,
) -> Pile:
    """Create a Pile with custom items and type constraints."""
    return Pile(items=items, item_type=item_type, strict_type=strict_type)


def create_test_progression(
    items: list[Element] | None = None,
    name: str = "test_progression",
) -> Progression:
    """Create a test Progression from items."""
    if items is None:
        items = create_test_elements(count=5)
    return Progression(order=[item.id for item in items], name=name)


def create_test_flow(
    item_count: int = 5,
    progression_count: int = 2,
) -> Flow[TestElement, Progression]:
    """Create a test Flow with elements and progressions."""
    items = create_test_elements(count=item_count)
    progressions = [Progression(name=f"prog_{i}", order=[]) for i in range(progression_count)]
    return Flow(items=items, progressions=progressions, item_type=TestElement)


def create_empty_graph() -> Graph:
    """Create an empty Graph for testing."""
    return Graph()


def create_simple_graph() -> tuple[Graph, list[Node]]:
    """Create a simple graph with 3 nodes in a chain: A -> B -> C."""
    graph = Graph()
    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_edge(Edge(head=n1.id, tail=n2.id))
    graph.add_edge(Edge(head=n2.id, tail=n3.id))
    return graph, [n1, n2, n3]


def create_cyclic_graph() -> tuple[Graph, list[Node]]:
    """Create a graph with a cycle: A -> B -> C -> A."""
    graph = Graph()
    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_edge(Edge(head=n1.id, tail=n2.id))
    graph.add_edge(Edge(head=n2.id, tail=n3.id))
    graph.add_edge(Edge(head=n3.id, tail=n1.id))
    return graph, [n1, n2, n3]


def create_dag_graph() -> tuple[Graph, list[Node]]:
    """Create a DAG: A -> B -> D, A -> C -> D."""
    graph = Graph()
    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})
    n4 = Node(content={"value": "D"})
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    graph.add_edge(Edge(head=n1.id, tail=n2.id))
    graph.add_edge(Edge(head=n1.id, tail=n3.id))
    graph.add_edge(Edge(head=n2.id, tail=n4.id))
    graph.add_edge(Edge(head=n3.id, tail=n4.id))
    return graph, [n1, n2, n3, n4]


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def element_fixture() -> TestElement:
    """Create a single mock Element for testing (pytest fixture)."""
    return TestElement(value=0, name="test")


@pytest.fixture
def node_fixture() -> Node:
    """Create a single mock Node for testing (pytest fixture)."""
    return Node(content={"value": "test"})


@pytest.fixture
def test_elements() -> list[TestElement]:
    """Create a list of 5 test Elements."""
    return create_test_elements(count=5)


@pytest.fixture
def test_nodes() -> list[Node]:
    """Create a list of 5 test Nodes."""
    return create_test_nodes(count=5)


@pytest.fixture
def test_pile() -> Pile[TestElement]:
    """Create a test Pile with 5 elements."""
    return create_test_pile(count=5)


@pytest.fixture
def test_progression(test_elements) -> Progression:
    """Create a test Progression from test_elements."""
    return create_test_progression(items=test_elements)


@pytest.fixture
def test_flow() -> Flow[TestElement, Progression]:
    """Create a test Flow with 5 items and 2 progressions."""
    return create_test_flow(item_count=5, progression_count=2)


@pytest.fixture
def empty_graph() -> Graph:
    """Create an empty Graph."""
    return create_empty_graph()


@pytest.fixture
def simple_graph() -> tuple[Graph, list[Node]]:
    """Create a simple chain graph: A -> B -> C."""
    return create_simple_graph()


@pytest.fixture
def cyclic_graph() -> tuple[Graph, list[Node]]:
    """Create a cyclic graph: A -> B -> C -> A."""
    return create_cyclic_graph()


@pytest.fixture
def dag_graph() -> tuple[Graph, list[Node]]:
    """Create a DAG graph."""
    return create_dag_graph()


@pytest.fixture
def simple_event() -> SimpleTestEvent:
    """Create a SimpleTestEvent with default value."""
    return SimpleTestEvent(return_value="test_result")


@pytest.fixture
def failing_event() -> FailingTestEvent:
    """Create a FailingTestEvent."""
    return FailingTestEvent(error_message="Test error")


@pytest.fixture
def slow_event() -> SlowTestEvent:
    """Create a SlowTestEvent with 0.1s delay."""
    return SlowTestEvent(delay=0.1)


@pytest.fixture
def streaming_event() -> StreamingTestEvent:
    """Create a StreamingTestEvent with 3 items."""
    return StreamingTestEvent(stream_count=3)


@pytest.fixture
def test_processor() -> TestProcessor:
    """Create a TestProcessor."""
    pile = Pile[Event]()
    return TestProcessor(pile=pile)


# =============================================================================
# Hypothesis Strategies (optional)
# =============================================================================

try:
    from hypothesis import strategies as st

    def element_strategy() -> st.SearchStrategy[TestElement]:
        """Hypothesis strategy for generating TestElement instances."""
        return st.builds(
            TestElement,
            value=st.integers(min_value=0, max_value=1000),
            name=st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"),
                    min_codepoint=ord("0"),
                    max_codepoint=ord("z"),
                ),
            ),
        )

    def node_strategy() -> st.SearchStrategy[Node]:
        """Hypothesis strategy for generating Node instances."""
        return st.builds(
            Node,
            content=st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.one_of(
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.text(max_size=50),
                    st.booleans(),
                ),
                min_size=1,
                max_size=5,
            ),
        )

    def progression_strategy(
        min_items: int = 0,
        max_items: int = 10,
    ) -> st.SearchStrategy[Progression]:
        """Hypothesis strategy for generating Progression instances."""
        return st.builds(
            Progression,
            order=st.lists(
                st.uuids().map(lambda u: u),
                min_size=min_items,
                max_size=max_items,
            ),
            name=st.text(min_size=1, max_size=20),
        )

except ImportError:

    def element_strategy():
        raise ImportError("hypothesis is required for property-based testing strategies")

    def node_strategy():
        raise ImportError("hypothesis is required for property-based testing strategies")

    def progression_strategy(min_items: int = 0, max_items: int = 10):
        raise ImportError("hypothesis is required for property-based testing strategies")
