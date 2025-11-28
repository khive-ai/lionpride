# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Pile: UNIQUE tests not covered by test_pile.py.

Note: Basic operations (get, remove, pop, contains, include, exclude) are
tested in tests/core/test_pile.py. This file covers edge cases only.
"""

import concurrent.futures
import threading

from lionpride.core import Element, Pile


class SampleElement(Element):
    """Simple Element subclass for testing."""

    value: int = 0
    name: str = ""


def create_test_elements(n: int) -> list[SampleElement]:
    """Create n unique test elements."""
    return [SampleElement(value=i, name=f"elem_{i}") for i in range(n)]


# =============================================================================
# Single-Item Edge Cases (unique - tests boundary of 1 item)
# =============================================================================


class TestSingleItemPile:
    """Tests for piles with exactly one item - boundary condition."""

    def test_remove_only_item_leaves_valid_empty_pile(self):
        """Removing the only item should leave valid empty pile state."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        pile.remove(item.id)

        assert len(pile) == 0
        assert pile.is_empty()
        assert item.id not in pile
        # Verify internal state consistency
        assert len(list(pile.keys())) == len(pile._items) == 0


# =============================================================================
# Slice Edge Cases (unique - negative indices, reverse, edge slices)
# =============================================================================


class TestSliceEdgeCases:
    """Tests for slice operations - negative indices, reverse, boundaries."""

    def test_empty_slice_start_equals_end(self):
        """pile[2:2] should return empty pile (edge case)."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[2:2]
        assert len(result) == 0
        assert isinstance(result, Pile)

    def test_negative_start_slice(self):
        """pile[-2:] should return last 2 items."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[-2:]
        assert len(result) == 2
        # Verify it's the LAST 2 items
        assert list(result) == items[-2:]

    def test_reverse_slice(self):
        """pile[::-1] should return items in reverse order."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[::-1]
        assert len(result) == 5
        assert list(result) == items[::-1]

    def test_step_slice(self):
        """pile[::2] should return every other item."""
        items = create_test_elements(6)
        pile = Pile(items=items)
        result = pile[::2]
        assert len(result) == 3
        assert list(result) == items[::2]


# =============================================================================
# Include Behavior (unique - tests that modified item doesn't replace)
# =============================================================================


class TestIncludePreservesOriginal:
    """Test that include() doesn't replace existing item with same ID."""

    def test_include_existing_item_preserves_original_data(self):
        """include() of existing item should NOT replace original's data.

        This is a critical semantic test: include() is idempotent and should
        preserve the FIRST item added, not silently update it.
        """
        original = SampleElement(value=42, name="original")
        pile = Pile(items=[original])

        # Create item with SAME ID but different data
        modified = SampleElement(value=999, name="modified", id=original.id)
        pile.include(modified)

        # Original should be preserved
        retrieved = pile.get(original.id)
        assert retrieved.value == 42
        assert retrieved.name == "original"
        assert len(pile) == 1


# =============================================================================
# Thread Safety Stress (unique - higher stress than test_pile.py)
# =============================================================================


class TestThreadSafetyStress:
    """High-stress concurrency tests beyond basic thread safety."""

    def test_concurrent_add_remove_same_items_many_iterations(self):
        """Concurrent add/remove of same items across many iterations."""
        pile: Pile[SampleElement] = Pile()
        items = create_test_elements(10)
        errors: list[tuple[str, Exception]] = []
        lock = threading.Lock()

        def add_items():
            for item in items:
                try:
                    pile.include(item)
                except Exception as e:
                    with lock:
                        errors.append(("add", e))

        def remove_items():
            for item in items:
                try:
                    pile.exclude(item.id)
                except Exception as e:
                    with lock:
                        errors.append(("remove", e))

        # Run many iterations to expose race conditions
        for _ in range(20):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(add_items),
                    executor.submit(add_items),
                    executor.submit(remove_items),
                    executor.submit(remove_items),
                ]
                concurrent.futures.wait(futures)

        assert not errors, f"Errors occurred: {errors}"
        # Verify internal consistency
        assert len(list(pile.keys())) == len(pile._items)


# =============================================================================
# Serialization Edge Cases (unique - metadata preservation)
# =============================================================================


class TestSerializationEdgeCases:
    """Tests for serialization edge cases not in test_pile.py."""

    def test_empty_pile_with_metadata_roundtrip(self):
        """Empty pile with metadata should roundtrip correctly."""
        pile: Pile[SampleElement] = Pile()
        pile.metadata["custom_key"] = "custom_value"
        pile.metadata["nested"] = {"deep": "data"}

        data = pile.to_dict()
        restored = Pile.from_dict(data)

        assert len(restored) == 0
        assert restored.metadata.get("custom_key") == "custom_value"
        assert restored.metadata.get("nested") == {"deep": "data"}

    def test_pile_with_progression_name_roundtrip(self):
        """Pile with custom progression name should preserve it."""
        items = create_test_elements(3)
        pile: Pile[SampleElement] = Pile(items=items)
        pile._progression.name = "custom_order"

        data = pile.to_dict(mode="json")
        restored = Pile.from_dict(data)

        assert restored._progression.name == "custom_order"
        assert len(restored) == 3
