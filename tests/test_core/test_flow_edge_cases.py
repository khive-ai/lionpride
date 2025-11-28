# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Flow: UNIQUE tests not covered by test_flow.py.

Note: Basic operations (add_item, remove_item, add_progression, get_progression)
are tested in tests/core/test_flow.py. This file covers edge cases only.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from lionpride.core import Element, Flow, Progression
from lionpride.errors import ExistsError


class SampleElement(Element):
    """Simple Element subclass for testing."""

    value: int = 0
    name: str = ""


# =============================================================================
# Progression Name Conflicts (unique - None names, name reuse after removal)
# =============================================================================


class TestProgressionNameConflicts:
    """Tests for progression name handling edge cases."""

    def test_multiple_progressions_with_none_name_allowed(self):
        """Multiple progressions with name=None are allowed (not indexed by name)."""
        flow = Flow[SampleElement, Progression]()
        prog1 = Progression(name=None, order=[])
        prog2 = Progression(name=None, order=[])

        flow.add_progression(prog1)
        flow.add_progression(prog2)

        assert len(flow.progressions) == 2
        # Both accessible by ID, not by name
        assert flow.progressions[prog1.id] is prog1
        assert flow.progressions[prog2.id] is prog2

    def test_remove_progression_frees_name_for_reuse(self):
        """After removing progression, its name can be reused."""
        flow = Flow[SampleElement, Progression]()
        prog1 = Progression(name="reusable", order=[])
        flow.add_progression(prog1)

        flow.remove_progression("reusable")

        # Name should now be available for reuse
        prog2 = Progression(name="reusable", order=[])
        flow.add_progression(prog2)  # Should not raise
        assert flow.get_progression("reusable") is prog2


# =============================================================================
# Concurrent Name Conflicts (unique - only one wins under concurrency)
# =============================================================================


class TestConcurrentNameConflicts:
    """Tests for concurrent name conflict resolution."""

    def test_concurrent_add_same_name_only_one_wins(self):
        """Concurrent additions with same name: exactly one succeeds."""
        flow = Flow[SampleElement, Progression]()
        success_count = [0]
        failure_count = [0]
        lock = threading.Lock()

        def try_add_prog(i: int):
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

        assert success_count[0] == 1, f"Expected 1 success, got {success_count[0]}"
        assert failure_count[0] == 19, f"Expected 19 failures, got {failure_count[0]}"


# =============================================================================
# Remove Consistency (unique - name vs UUID removal produces same result)
# =============================================================================


class TestRemoveProgressionConsistency:
    """Tests for consistent behavior between removal methods."""

    def test_remove_by_name_and_uuid_same_internal_state(self):
        """Removing by name vs UUID should produce identical internal state."""
        flow1 = Flow[SampleElement, Progression]()
        prog1 = Progression(name="test", order=[])
        flow1.add_progression(prog1)

        flow2 = Flow[SampleElement, Progression]()
        prog2 = Progression(name="test", order=[])
        flow2.add_progression(prog2)

        # Remove by different methods
        flow1.remove_progression("test")
        flow2.remove_progression(prog2.id)

        # Both should have identical state
        assert "test" not in flow1._progression_names
        assert "test" not in flow2._progression_names
        assert len(flow1.progressions) == len(flow2.progressions) == 0


# =============================================================================
# Serialization Order Preservation (unique - non-sequential order)
# =============================================================================


class TestSerializationOrderPreservation:
    """Tests for serialization fidelity with specific ordering."""

    def test_roundtrip_preserves_non_sequential_order(self):
        """Serialization preserves exact order, not just membership."""
        flow = Flow[SampleElement, Progression](name="order_test")
        items = [SampleElement(value=i, name=f"item{i}") for i in range(10)]
        for item in items:
            flow.items.add(item)

        # Create progression with specific non-sequential order
        order = [items[5].id, items[2].id, items[8].id, items[0].id]
        prog = Progression(name="ordered", order=order)
        flow.add_progression(prog)

        data = flow.to_dict(mode="json")
        restored = Flow.from_dict(data)

        restored_prog = restored.get_progression("ordered")
        assert list(restored_prog.order) == order, "Order not preserved through roundtrip"

    def test_roundtrip_items_in_multiple_progressions(self):
        """Same item can be in multiple progressions after roundtrip."""
        flow = Flow[SampleElement, Progression](name="multi_prog")
        items = [SampleElement(value=i, name=f"item{i}") for i in range(5)]
        for item in items:
            flow.items.add(item)

        # items[2] in both progressions
        prog1 = Progression(name="stage1", order=[items[0].id, items[1].id, items[2].id])
        prog2 = Progression(name="stage2", order=[items[2].id, items[3].id, items[4].id])
        flow.add_progression(prog1)
        flow.add_progression(prog2)

        data = flow.to_dict(mode="json")
        restored = Flow.from_dict(data)

        assert len(restored.items) == 5
        assert len(restored.progressions) == 2
        # Verify shared item is in both
        assert items[2].id in restored.get_progression("stage1")
        assert items[2].id in restored.get_progression("stage2")
