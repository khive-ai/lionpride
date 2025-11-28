# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Flow that could expose potential bugs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from lionpride.core import Element, Flow, Pile, Progression
from lionpride.errors import ExistsError, NotFoundError


class SampleElement(Element):
    """Simple Element subclass for testing."""

    value: int = 0
    name: str = ""


def create_test_elements(n: int) -> list[SampleElement]:
    """Create n unique test elements."""
    return [SampleElement(value=i, name=f"elem_{i}") for i in range(n)]


# =============================================================================
# 1. Empty Flow Operations
# =============================================================================


class TestEmptyFlowOperations:
    """Tests for operations on completely empty flows."""

    def test_empty_flow_get_progression_raises_keyerror(self):
        """Getting progression from empty flow raises KeyError."""
        flow = Flow[SampleElement, Progression]()
        with pytest.raises(KeyError, match="not found"):
            flow.get_progression("nonexistent")

    def test_empty_flow_remove_item_raises_notfounderror(self):
        """Removing item from empty flow raises NotFoundError."""
        flow = Flow[SampleElement, Progression]()
        fake_id = uuid4()
        with pytest.raises(NotFoundError):
            flow.remove_item(fake_id)

    def test_empty_flow_iterate_progressions(self):
        """Iterating over empty flow progressions yields nothing."""
        flow = Flow[SampleElement, Progression]()
        items_list = list(flow.progressions)
        assert items_list == []

    def test_empty_flow_to_dict_roundtrip(self):
        """Empty flow serializes and deserializes correctly."""
        flow = Flow[SampleElement, Progression](name="empty_flow")
        data = flow.to_dict(mode="json")
        restored = Flow.from_dict(data)
        assert len(restored.items) == 0
        assert len(restored.progressions) == 0
        assert restored.name == "empty_flow"


# =============================================================================
# 2. Progression with Empty Order List
# =============================================================================


class TestEmptyProgression:
    """Tests for progressions with empty order lists."""

    def test_add_empty_progression_succeeds(self):
        """Adding progression with empty order list should succeed."""
        flow = Flow[SampleElement, Progression]()
        prog = Progression(name="empty_stage", order=[])
        flow.add_progression(prog)
        assert len(flow.progressions) == 1

    def test_add_item_to_empty_progression_via_name(self):
        """Adding item to empty progression by name succeeds."""
        flow = Flow[SampleElement, Progression]()
        prog = Progression(name="stage1", order=[])
        flow.add_progression(prog)
        item = SampleElement(value=1, name="item1")
        flow.add_item(item, progressions="stage1")
        assert item.id in prog


# =============================================================================
# 3. add_item to Non-Existent Progression
# =============================================================================


class TestAddItemToNonExistentProgression:
    """Tests for adding items to progressions that don't exist."""

    def test_add_item_nonexistent_name_raises_keyerror(self):
        """Adding item to non-existent progression name raises KeyError."""
        flow = Flow[SampleElement, Progression]()
        item = SampleElement(value=1, name="item1")
        with pytest.raises(KeyError, match="not found"):
            flow.add_item(item, progressions="ghost_stage")
        assert len(flow.items) == 0

    def test_add_item_partial_valid_progressions_fails_atomically(self):
        """Adding item to mix of valid/invalid progressions fails atomically."""
        flow = Flow[SampleElement, Progression]()
        valid_prog = Progression(name="valid", order=[])
        flow.add_progression(valid_prog)
        item = SampleElement(value=1, name="item1")
        with pytest.raises((KeyError, NotFoundError)):
            flow.add_item(item, progressions=["valid", "invalid_stage"])
        assert len(flow.items) == 0
        assert item.id not in valid_prog


# =============================================================================
# 4. remove_item from Item Not in Any Progression
# =============================================================================


class TestRemoveItemNotInProgressions:
    """Tests for removing items that exist in pile but not in any progression."""

    def test_remove_item_only_in_pile_succeeds(self):
        """Removing item that's in pile but no progressions succeeds."""
        flow = Flow[SampleElement, Progression]()
        prog = Progression(name="stage1", order=[])
        flow.add_progression(prog)
        item = SampleElement(value=1, name="item1")
        flow.add_item(item)
        removed = flow.remove_item(item.id)
        assert removed is item
        assert item.id not in flow.items


# =============================================================================
# 5. Progression Name Conflicts and Resolution
# =============================================================================


class TestProgressionNameConflicts:
    """Tests for progression name uniqueness and conflict handling."""

    def test_progression_name_none_allows_multiple(self):
        """Multiple progressions with name=None are allowed."""
        flow = Flow[SampleElement, Progression]()
        prog1 = Progression(name=None, order=[])
        prog2 = Progression(name=None, order=[])
        flow.add_progression(prog1)
        flow.add_progression(prog2)
        assert len(flow.progressions) == 2

    def test_remove_progression_by_name_frees_name(self):
        """After removing progression by name, the name can be reused."""
        flow = Flow[SampleElement, Progression]()
        prog1 = Progression(name="reusable", order=[])
        flow.add_progression(prog1)
        flow.remove_progression("reusable")
        prog2 = Progression(name="reusable", order=[])
        flow.add_progression(prog2)
        assert flow.get_progression("reusable") is prog2


# =============================================================================
# 6. Referential Integrity Under Concurrent Modifications
# =============================================================================


class TestConcurrentModifications:
    """Tests for thread safety under concurrent access."""

    def test_concurrent_add_item_thread_safety(self):
        """Adding items concurrently doesn't corrupt state."""
        flow = Flow[SampleElement, Progression]()
        items = [SampleElement(value=i, name=f"item{i}") for i in range(100)]

        def add_item(item):
            flow.items.add(item)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(add_item, items))

        assert len(flow.items) == 100

    def test_concurrent_name_conflict_only_one_wins(self):
        """Concurrent additions with same name: only one succeeds."""
        flow = Flow[SampleElement, Progression]()
        success_count = [0]
        failure_count = [0]
        lock = threading.Lock()

        def try_add_prog(i):
            prog = Progression(name="contested", order=[])
            try:
                flow.add_progression(prog)
                with lock:
                    success_count[0] += 1
            except ExistsError:
                with lock:
                    failure_count[0] += 1

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(try_add_prog, range(20)))

        assert success_count[0] == 1
        assert failure_count[0] == 19


# =============================================================================
# 7. add_progression with UUIDs Not in Items Pile
# =============================================================================


class TestProgressionReferentialIntegrity:
    """Tests for referential integrity validation on add_progression."""

    def test_add_progression_single_missing_uuid_fails(self):
        """Adding progression with single missing UUID fails."""
        flow = Flow[SampleElement, Progression]()
        item = SampleElement(value=1, name="item1")
        flow.items.add(item)
        missing_uuid = uuid4()
        prog = Progression(name="partial", order=[item.id, missing_uuid])
        with pytest.raises(NotFoundError):
            flow.add_progression(prog)
        assert len(flow.progressions) == 0


# =============================================================================
# 8. remove_progression by Name vs by UUID Consistency
# =============================================================================


class TestRemoveProgressionConsistency:
    """Tests for consistent behavior between name and UUID removal."""

    def test_remove_by_name_and_uuid_same_result(self):
        """Removing by name vs UUID produces identical results."""
        flow1 = Flow[SampleElement, Progression]()
        prog1 = Progression(name="test", order=[])
        flow1.add_progression(prog1)
        flow1.remove_progression("test")

        flow2 = Flow[SampleElement, Progression]()
        prog2 = Progression(name="test", order=[])
        flow2.add_progression(prog2)
        flow2.remove_progression(prog2.id)

        assert "test" not in flow1._progression_names
        assert "test" not in flow2._progression_names


# =============================================================================
# 9. get_progression with Invalid Types
# =============================================================================


class TestGetProgressionInvalidTypes:
    """Tests for get_progression with invalid key types."""

    def test_get_progression_with_malformed_uuid_string(self):
        """Getting progression with malformed UUID string raises KeyError."""
        flow = Flow[SampleElement, Progression]()
        with pytest.raises(KeyError, match="not found"):
            flow.get_progression("not-a-valid-uuid")


# =============================================================================
# 10. Serialization Roundtrip with Complex Nested Structures
# =============================================================================


class TestComplexSerialization:
    """Tests for serialization with complex nested data."""

    def test_roundtrip_with_items_in_multiple_progressions(self):
        """Serialize/deserialize with items in multiple progressions."""
        flow = Flow[SampleElement, Progression](name="complex")
        items = [SampleElement(value=i, name=f"item{i}") for i in range(5)]
        for item in items:
            flow.items.add(item)
        prog1 = Progression(name="stage1", order=[items[0].id, items[1].id, items[2].id])
        prog2 = Progression(name="stage2", order=[items[2].id, items[3].id, items[4].id])
        flow.add_progression(prog1)
        flow.add_progression(prog2)
        data = flow.to_dict(mode="json")
        restored = Flow.from_dict(data)
        assert len(restored.items) == 5
        assert len(restored.progressions) == 2

    def test_roundtrip_preserves_progression_order(self):
        """Serialization preserves exact order in progressions."""
        flow = Flow[SampleElement, Progression](name="order_test")
        items = [SampleElement(value=i, name=f"item{i}") for i in range(10)]
        for item in items:
            flow.items.add(item)
        order = [items[5].id, items[2].id, items[8].id, items[0].id]
        prog = Progression(name="ordered", order=order)
        flow.add_progression(prog)
        data = flow.to_dict(mode="json")
        restored = Flow.from_dict(data)
        restored_prog = restored.get_progression("ordered")
        assert list(restored_prog.order) == order
