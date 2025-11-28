# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Progression: UNIQUE tests not covered by test_progression.py.

Note: Basic operations (include/exclude idempotency, move validation, contains,
serialization roundtrip) are tested in tests/core/test_progression.py.
This file covers edge cases only.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from lionpride.core import Element, Progression

# =============================================================================
# Insert Negative Indices (unique - behavior at -1)
# =============================================================================


class TestInsertNegativeIndices:
    """Tests for insert() with negative indices - boundary behavior."""

    def test_insert_at_negative_one(self):
        """insert() at -1 should insert before last item."""
        uid1, uid2 = uuid4(), uuid4()
        uid_new = uuid4()
        prog = Progression(order=[uid1, uid2])
        prog.insert(-1, uid_new)
        assert prog.order == [uid1, uid_new, uid2]


# =============================================================================
# Extend Edge Cases (unique - empty iterable handling)
# =============================================================================


class TestExtendEdgeCases:
    """Tests for extend() with empty/boundary iterables."""

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


# =============================================================================
# Clear on Empty (unique - idempotent clear behavior)
# =============================================================================


class TestClearEdgeCases:
    """Tests for clear() on empty progressions."""

    def test_clear_empty_progression(self):
        """clear() on empty progression should not raise."""
        prog = Progression()
        prog.clear()
        assert len(prog) == 0

    def test_clear_twice(self):
        """clear() called twice should work (idempotent)."""
        prog = Progression(order=[uuid4(), uuid4()])
        prog.clear()
        prog.clear()
        assert len(prog) == 0


# =============================================================================
# Setitem Slice Edge Cases (unique - empty list assignment)
# =============================================================================


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


# =============================================================================
# Thread Safety Stress (unique - high concurrency stress test)
# =============================================================================


class TestThreadSafetyStress:
    """Stress tests for thread safety (Progression is NOT thread-safe by design)."""

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
