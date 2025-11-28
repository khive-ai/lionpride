# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Pile: Focus on behaviors that could reveal bugs."""

import concurrent.futures
import threading
import time
from uuid import UUID, uuid4

import pytest

from lionpride.core import Element, Node, Pile, Progression
from lionpride.errors import ExistsError, NotFoundError


class SampleElement(Element):
    """Simple Element subclass for testing."""

    value: int = 0
    name: str = ""


def create_test_elements(n: int) -> list[SampleElement]:
    """Create n unique test elements."""
    return [SampleElement(value=i, name=f"elem_{i}") for i in range(n)]


# =============================================================================
# 1. Empty Pile Operations
# =============================================================================


class TestEmptyPileOperations:
    """Tests for operations on empty piles that could reveal bugs."""

    def test_get_on_empty_pile_raises_not_found(self):
        """Empty pile should raise NotFoundError for any get(), not KeyError."""
        pile = Pile()
        with pytest.raises(NotFoundError, match="not found"):
            pile.get(uuid4())

    def test_remove_on_empty_pile_raises_not_found(self):
        """Empty pile should raise NotFoundError for remove(), not KeyError."""
        pile = Pile()
        with pytest.raises(NotFoundError, match="not found"):
            pile.remove(uuid4())

    def test_update_on_empty_pile_raises_not_found(self):
        """Empty pile should raise NotFoundError for update()."""
        pile = Pile()
        item = SampleElement(value=42)
        with pytest.raises(NotFoundError, match="not found"):
            pile.update(item)

    def test_pop_on_empty_pile_without_default_raises(self):
        """pop() on empty pile without default should raise NotFoundError."""
        pile = Pile()
        with pytest.raises(NotFoundError, match="not found"):
            pile.pop(uuid4())

    def test_pop_on_empty_pile_with_default_returns_default(self):
        """pop() on empty pile with default should return default."""
        pile = Pile()
        sentinel = object()
        result = pile.pop(uuid4(), default=sentinel)
        assert result is sentinel

    def test_getitem_index_on_empty_pile_raises_index_error(self):
        """pile[0] on empty pile should raise IndexError, not KeyError."""
        pile = Pile()
        with pytest.raises(IndexError):
            _ = pile[0]

    def test_slice_on_empty_pile_returns_empty_pile(self):
        """Slicing empty pile should return empty pile, not raise."""
        pile = Pile()
        result = pile[:]
        assert isinstance(result, Pile)
        assert len(result) == 0

    def test_filter_on_empty_pile_returns_empty_pile(self):
        """Filter on empty pile should return empty pile, not raise."""
        pile = Pile()
        result = pile[lambda x: True]
        assert isinstance(result, Pile)
        assert len(result) == 0


# =============================================================================
# 2. Single-Item Edge Cases
# =============================================================================


class TestSingleItemPile:
    """Tests for piles with exactly one item."""

    def test_remove_only_item_leaves_empty_pile(self):
        """Removing the only item should leave valid empty pile."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        pile.remove(item.id)

        assert len(pile) == 0
        assert pile.is_empty()
        assert item.id not in pile

    def test_slice_single_item_all(self):
        """pile[:] on single item should return pile with that item."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        result = pile[:]
        assert len(result) == 1
        assert item.id in result


# =============================================================================
# 3. Type Coercion Edge Cases
# =============================================================================


class TestTypeCoercion:
    """Tests for UUID/string/Element coercion in operations."""

    def test_get_by_uuid_string(self):
        """get() should accept UUID as string."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        result = pile.get(str(item.id))
        assert result == item

    def test_get_by_uuid_object(self):
        """get() should accept UUID object."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        result = pile.get(item.id)
        assert result == item

    def test_get_by_element(self):
        """get() should accept Element (extracts id)."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        result = pile.get(item)
        assert result == item

    def test_contains_by_string_uuid(self):
        """__contains__ should work with string UUID."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        assert str(item.id) in pile


# =============================================================================
# 4. Predicate Filter Edge Cases
# =============================================================================


class TestPredicateFilter:
    """Tests for callable predicate filtering edge cases."""

    def test_filter_returns_empty_pile(self):
        """Filter that matches nothing returns empty pile, not None."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[lambda x: False]
        assert isinstance(result, Pile)
        assert len(result) == 0

    def test_filter_returns_all_items(self):
        """Filter that matches everything returns pile with all items."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[lambda x: True]
        assert len(result) == 5

    def test_filter_creates_independent_pile(self):
        """Filtered pile should be independent - mutations don't affect original."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        filtered = pile[lambda x: x.value < 3]
        filtered.clear()
        assert len(pile) == 5


# =============================================================================
# 5. Slice Edge Cases
# =============================================================================


class TestSliceEdgeCases:
    """Tests for slice operations edge cases."""

    def test_full_slice(self):
        """pile[::] should return all items."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[::]
        assert len(result) == 5

    def test_empty_slice_start_equals_end(self):
        """pile[2:2] should return empty pile."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[2:2]
        assert len(result) == 0

    def test_negative_start_slice(self):
        """pile[-2:] should return last 2 items."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[-2:]
        assert len(result) == 2

    def test_reverse_slice(self):
        """pile[::-1] should return items in reverse order."""
        items = create_test_elements(5)
        pile = Pile(items=items)
        result = pile[::-1]
        assert len(result) == 5
        assert list(result) == items[::-1]


# =============================================================================
# 6. include() Idempotency Edge Cases
# =============================================================================


class TestIncludeIdempotency:
    """Tests for include() behavior with existing items."""

    def test_include_existing_item_returns_true(self):
        """include() of existing item should return True."""
        item = SampleElement(value=42)
        pile = Pile(items=[item])
        result = pile.include(item)
        assert result is True
        assert len(pile) == 1

    def test_include_existing_item_preserves_original(self):
        """include() of existing item should NOT replace original."""
        original = SampleElement(value=42, name="original")
        pile = Pile(items=[original])
        modified = SampleElement(value=999, name="modified", id=original.id)
        pile.include(modified)
        retrieved = pile.get(original.id)
        assert retrieved.value == 42
        assert retrieved.name == "original"


# =============================================================================
# 7. exclude() Edge Cases
# =============================================================================


class TestExcludeEdgeCases:
    """Tests for exclude() behavior."""

    def test_exclude_nonexistent_uuid_returns_true(self):
        """exclude() of non-existent UUID should return True."""
        pile = Pile(items=[SampleElement(value=42)])
        result = pile.exclude(uuid4())
        assert result is True

    def test_exclude_from_empty_pile_returns_true(self):
        """exclude() on empty pile should return True."""
        pile = Pile()
        result = pile.exclude(uuid4())
        assert result is True


# =============================================================================
# 8. Thread Safety Stress Tests
# =============================================================================


class TestThreadSafetyStress:
    """Stress tests for thread safety."""

    def test_concurrent_add_remove_same_items(self):
        """Concurrent add/remove of same items should not corrupt state."""
        pile = Pile()
        items = create_test_elements(10)
        errors = []
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

        for _ in range(10):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(add_items),
                    executor.submit(add_items),
                    executor.submit(remove_items),
                    executor.submit(remove_items),
                ]
                concurrent.futures.wait(futures)

        assert not errors, f"Errors occurred: {errors}"
        assert len(list(pile.keys())) == len(pile._items)


# =============================================================================
# 9. Serialization Edge Cases
# =============================================================================


class TestSerializationEdgeCases:
    """Tests for serialization edge cases."""

    def test_empty_pile_roundtrip_with_metadata(self):
        """Empty pile with metadata should roundtrip correctly."""
        pile = Pile()
        pile.metadata["key"] = "value"
        data = pile.to_dict()
        restored = Pile.from_dict(data)
        assert len(restored) == 0
        assert restored.metadata.get("key") == "value"

    def test_pile_id_preserved_through_serialization(self):
        """Pile's own ID should be preserved through serialization."""
        pile = Pile(items=[SampleElement(value=42)])
        original_id = pile.id
        data = pile.to_dict()
        restored = Pile.from_dict(data)
        assert restored.id == original_id
