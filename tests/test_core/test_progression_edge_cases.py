# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Progression that could expose bugs."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import UUID, uuid4

import pytest

from lionpride.core import Element, Progression
from lionpride.errors import NotFoundError


class TestIncludeIdempotency:
    """Tests for include() idempotency guarantees."""

    def test_include_same_uuid_twice_returns_false_second_time(self):
        """include() with same UUID twice should return True then False."""
        prog = Progression()
        uid = uuid4()
        first_result = prog.include(uid)
        second_result = prog.include(uid)
        assert first_result is True
        assert second_result is False
        assert len(prog) == 1

    def test_include_element_then_uuid_of_same_element(self):
        """include() with Element then its UUID should detect duplicate."""
        prog = Progression()
        elem = Element()
        first_result = prog.include(elem)
        second_result = prog.include(elem.id)
        assert first_result is True
        assert second_result is False
        assert len(prog) == 1


class TestExcludeIdempotency:
    """Tests for exclude() behavior on non-existent items."""

    def test_exclude_nonexistent_uuid_returns_false(self):
        """exclude() on non-existent UUID should return False."""
        prog = Progression(order=[uuid4()])
        nonexistent = uuid4()
        result = prog.exclude(nonexistent)
        assert result is False
        assert len(prog) == 1

    def test_exclude_from_empty_progression_returns_false(self):
        """exclude() from empty progression should return False."""
        prog = Progression()
        result = prog.exclude(uuid4())
        assert result is False


class TestMoveOutOfBounds:
    """Tests for move() with out-of-bounds indices."""

    def test_move_from_index_out_of_bounds_raises(self):
        """move() with from_index out of bounds should raise."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        with pytest.raises(NotFoundError, match="out of range"):
            prog.move(10, 0)

    def test_move_on_empty_progression_raises(self):
        """move() on empty progression should raise."""
        prog = Progression()
        with pytest.raises(NotFoundError, match="empty"):
            prog.move(0, 0)


class TestInsertNegativeIndices:
    """Tests for insert() with negative indices."""

    def test_insert_at_negative_one(self):
        """insert() at -1 should insert before last item."""
        uid1, uid2 = uuid4(), uuid4()
        uid_new = uuid4()
        prog = Progression(order=[uid1, uid2])
        prog.insert(-1, uid_new)
        assert prog.order == [uid1, uid_new, uid2]


class TestExtendEmptyIterable:
    """Tests for extend() with empty iterables."""

    def test_extend_empty_list(self):
        """extend() with empty list should be no-op."""
        uid1 = uuid4()
        prog = Progression(order=[uid1])
        original_len = len(prog)
        prog.extend([])
        assert len(prog) == original_len

    def test_extend_on_empty_progression(self):
        """extend() on empty progression should add items."""
        prog = Progression()
        uids = [uuid4(), uuid4()]
        prog.extend(uids)
        assert len(prog) == 2


class TestClearOnEmptyProgression:
    """Tests for clear() on empty progressions."""

    def test_clear_empty_progression(self):
        """clear() on empty progression should not raise."""
        prog = Progression()
        prog.clear()
        assert len(prog) == 0

    def test_clear_twice(self):
        """clear() called twice should work."""
        prog = Progression(order=[uuid4(), uuid4()])
        prog.clear()
        prog.clear()
        assert len(prog) == 0


class TestSetitemSliceEdgeCases:
    """Tests for __setitem__ with slice edge cases."""

    def test_setitem_slice_empty_list(self):
        """__setitem__ with slice and empty list should delete items."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        prog[0:2] = []
        assert prog.order == [uid3]

    def test_setitem_single_index_with_element(self):
        """__setitem__ single index should coerce Element to UUID."""
        uid_original = uuid4()
        elem_new = Element()
        prog = Progression(order=[uid_original])
        prog[0] = elem_new
        assert prog.order[0] == elem_new.id


class TestContainsVariousTypes:
    """Tests for __contains__ with different input types."""

    def test_contains_uuid_object(self):
        """__contains__ should accept UUID objects."""
        uid = uuid4()
        prog = Progression(order=[uid])
        assert uid in prog

    def test_contains_uuid_string(self):
        """__contains__ should accept UUID strings."""
        uid = uuid4()
        prog = Progression(order=[uid])
        assert str(uid) in prog

    def test_contains_element(self):
        """__contains__ should accept Element and extract ID."""
        elem = Element()
        prog = Progression(order=[elem.id])
        assert elem in prog

    def test_contains_invalid_string_returns_false(self):
        """__contains__ with invalid UUID string should return False."""
        prog = Progression(order=[uuid4()])
        assert "not-a-uuid" not in prog


class TestToDictFromDictRoundtrip:
    """Tests for serialization roundtrip fidelity."""

    def test_roundtrip_preserves_order_exactly(self):
        """Roundtrip should preserve exact order of UUIDs."""
        uids = [uuid4() for _ in range(5)]
        original = Progression(order=uids, name="test")
        data = original.to_dict(mode="json")
        restored = Progression.from_dict(data)
        assert restored.order == original.order

    def test_roundtrip_preserves_id(self):
        """Roundtrip should preserve the progression's own ID."""
        original = Progression(order=[uuid4()], name="test")
        original_id = original.id
        data = original.to_dict(mode="json")
        restored = Progression.from_dict(data)
        assert restored.id == original_id

    def test_roundtrip_with_duplicates(self):
        """Roundtrip should preserve duplicate UUIDs."""
        uid = uuid4()
        original = Progression(order=[uid, uid, uid])
        data = original.to_dict(mode="json")
        restored = Progression.from_dict(data)
        assert len(restored) == 3


class TestThreadSafety:
    """Tests for thread safety (Progression is NOT thread-safe by design)."""

    def test_concurrent_appends_eventually_consistent(self):
        """Concurrent appends should not lose items (GIL protection)."""
        prog = Progression()
        uids = [uuid4() for _ in range(100)]

        def append_uid(uid):
            prog.append(uid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(append_uid, uid) for uid in uids]
            for f in as_completed(futures):
                f.result()

        assert len(prog) == 100
        for uid in uids:
            assert uid in prog.order
