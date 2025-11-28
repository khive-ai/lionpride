# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Thread safety stress tests for lionpride collections."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import pytest

from lionpride.core import Element, Flow, Pile, Progression


class SampleElement(Element):
    """Simple Element subclass for testing."""

    value: int = 0


def create_elements(n: int) -> list[SampleElement]:
    """Create n unique test elements."""
    return [SampleElement(value=i) for i in range(n)]


class TestPileConcurrentAdd:
    """Test concurrent Pile.add() operations."""

    @pytest.mark.parametrize("num_threads", [10, 20])
    def test_concurrent_add_no_lost_items(self, num_threads: int):
        """Verify no items are lost during concurrent add operations."""
        iterations = 20
        items_per_thread = 10

        for _ in range(iterations):
            pile: Pile[SampleElement] = Pile()
            elements = create_elements(num_threads * items_per_thread)

            # Use default args to capture loop variables (avoid B023)
            def add_batch(start_idx: int, p: Pile = pile, elems: list = elements):
                for i in range(items_per_thread):
                    p.include(elems[start_idx + i])

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(add_batch, i * items_per_thread) for i in range(num_threads)
                ]
                for future in as_completed(futures):
                    future.result()

            assert len(pile) == num_threads * items_per_thread


class TestPileConcurrentIncludeExclude:
    """Test concurrent include/exclude operations."""

    def test_concurrent_include_idempotency(self):
        """Verify include is idempotent under concurrency."""
        iterations = 50
        num_threads = 20

        for _ in range(iterations):
            elem = SampleElement(value=42)
            pile: Pile[SampleElement] = Pile()

            results: list[bool] = []
            lock = threading.Lock()

            # Use default args to capture loop variables (avoid B023)
            def try_include(
                p: Pile = pile,
                e: SampleElement = elem,
                lk: threading.Lock = lock,
                res: list = results,
            ):
                result = p.include(e)
                with lk:
                    res.append(result)

            threads = [threading.Thread(target=try_include) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(pile) == 1
            assert any(results)

    def test_concurrent_exclude_idempotency(self):
        """Verify exclude is idempotent under concurrency."""
        iterations = 50
        num_threads = 20

        for _ in range(iterations):
            elem = SampleElement(value=42)
            pile: Pile[SampleElement] = Pile(items=[elem])

            results: list[bool] = []
            lock = threading.Lock()

            # Use default args to capture loop variables (avoid B023)
            def try_exclude(
                p: Pile = pile,
                e: SampleElement = elem,
                lk: threading.Lock = lock,
                res: list = results,
            ):
                result = p.exclude(e.id)
                with lk:
                    res.append(result)

            threads = [threading.Thread(target=try_exclude) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(pile) == 0
            assert all(results)


class TestPileMixedOperations:
    """Test mixed concurrent operations."""

    def test_add_remove_get_simultaneously(self):
        """Verify mixed operations maintain consistency."""
        iterations = 20
        num_operations = 20

        for _ in range(iterations):
            pile: Pile[SampleElement] = Pile()
            errors: list[str] = []
            lock = threading.Lock()

            # Use default args to capture loop variables (avoid B023)
            def adder(p: Pile = pile, n: int = num_operations):
                for i in range(n):
                    elem = SampleElement(value=i)
                    p.include(elem)

            def remover(p: Pile = pile, n: int = num_operations):
                for _ in range(n):
                    try:
                        if len(p) > 0:
                            p.exclude(p[0].id)
                    except Exception:
                        pass

            def reader(
                p: Pile = pile,
                n: int = num_operations,
                lk: threading.Lock = lock,
                errs: list = errors,
            ):
                for _ in range(n):
                    try:
                        _ = len(p)
                        if len(p) > 0:
                            _ = p[0]
                    except (IndexError, KeyError):
                        pass
                    except Exception as e:
                        with lk:
                            errs.append(str(e))

            threads = [
                threading.Thread(target=adder),
                threading.Thread(target=adder),
                threading.Thread(target=remover),
                threading.Thread(target=reader),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(list(pile.keys())) == len(pile._items)
            assert not errors


class TestFlowConcurrentAddItem:
    """Test concurrent Flow.add_item() operations."""

    @pytest.mark.xfail(
        reason="BUG: Flow._progression_names not synchronized - concurrent add_item fails",
        strict=False,
    )
    def test_concurrent_add_item_to_same_progression(self):
        """Verify concurrent adds to same progression.

        BUG DISCOVERED: Flow.get_progression() reads _progression_names without
        synchronization, causing KeyError under concurrent access even when
        the progression exists. The @synchronized decorator only protects
        add_progression/remove_progression, not get_progression.
        """
        iterations = 20
        num_threads = 10
        items_per_thread = 5

        for _ in range(iterations):
            flow: Flow = Flow()
            progression = Progression(name="main")
            flow.progressions.add(progression)

            elements = create_elements(num_threads * items_per_thread)

            # Use default args to capture loop variables (avoid B023)
            def add_batch(start_idx: int, f: Flow = flow, elems: list = elements):
                for i in range(items_per_thread):
                    f.add_item(elems[start_idx + i], progressions=["main"])

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(add_batch, i * items_per_thread) for i in range(num_threads)
                ]
                for future in as_completed(futures):
                    future.result()

            assert len(flow.items) == num_threads * items_per_thread
            prog = flow.get_progression("main")
            assert len(prog) == num_threads * items_per_thread
