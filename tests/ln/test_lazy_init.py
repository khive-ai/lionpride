# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LazyInit thread-safe lazy initialization utility."""

import threading
from concurrent.futures import ThreadPoolExecutor

from lionpride.ln._lazy_init import LazyInit


class TestLazyInit:
    """Test suite for LazyInit class."""

    def test_initial_state(self):
        """LazyInit starts uninitialized."""
        lazy = LazyInit()
        assert lazy.initialized is False

    def test_ensure_calls_init_func_once(self):
        """ensure() calls init_func exactly once."""
        lazy = LazyInit()
        call_count = 0

        def init_func():
            nonlocal call_count
            call_count += 1

        # Call ensure multiple times
        lazy.ensure(init_func)
        lazy.ensure(init_func)
        lazy.ensure(init_func)

        assert call_count == 1
        assert lazy.initialized is True

    def test_thread_safety(self):
        """ensure() is thread-safe - init_func called once with concurrent calls."""
        lazy = LazyInit()
        call_count = 0
        call_count_lock = threading.Lock()
        barrier = threading.Barrier(20)

        def init_func():
            nonlocal call_count
            with call_count_lock:
                call_count += 1

        def worker():
            # Synchronize all threads to start at the same time
            barrier.wait()
            lazy.ensure(init_func)

        # Use 20 concurrent threads (> 10 as required)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            # Wait for all to complete
            for f in futures:
                f.result()

        assert call_count == 1, f"init_func called {call_count} times, expected 1"
        assert lazy.initialized is True

    def test_multiple_instances_independent(self):
        """Multiple LazyInit instances are independent."""
        lazy1 = LazyInit()
        lazy2 = LazyInit()
        lazy3 = LazyInit()

        results = {"lazy1": 0, "lazy2": 0, "lazy3": 0}

        def make_init_func(key):
            def init_func():
                results[key] += 1

            return init_func

        # Initialize lazy1 only
        lazy1.ensure(make_init_func("lazy1"))

        assert lazy1.initialized is True
        assert lazy2.initialized is False
        assert lazy3.initialized is False
        assert results == {"lazy1": 1, "lazy2": 0, "lazy3": 0}

        # Initialize lazy2 only
        lazy2.ensure(make_init_func("lazy2"))

        assert lazy1.initialized is True
        assert lazy2.initialized is True
        assert lazy3.initialized is False
        assert results == {"lazy1": 1, "lazy2": 1, "lazy3": 0}

        # Initialize lazy3 only
        lazy3.ensure(make_init_func("lazy3"))

        assert lazy1.initialized is True
        assert lazy2.initialized is True
        assert lazy3.initialized is True
        assert results == {"lazy1": 1, "lazy2": 1, "lazy3": 1}

    def test_ensure_executes_init_func_before_setting_initialized(self):
        """ensure() executes init_func completely before setting initialized flag."""
        lazy = LazyInit()
        execution_order = []

        def init_func():
            execution_order.append("init_start")
            # Simulate some work
            execution_order.append("init_end")

        lazy.ensure(init_func)
        execution_order.append("after_ensure")

        assert execution_order == ["init_start", "init_end", "after_ensure"]
        assert lazy.initialized is True

    def test_thread_safety_with_slow_init(self):
        """Thread safety holds even when init_func is slow."""
        import time

        lazy = LazyInit()
        call_count = 0
        call_count_lock = threading.Lock()

        def slow_init_func():
            nonlocal call_count
            # Simulate slow initialization
            time.sleep(0.05)
            with call_count_lock:
                call_count += 1

        threads = []
        # Create 15 threads that all try to initialize
        for _ in range(15):
            t = threading.Thread(target=lambda: lazy.ensure(slow_init_func))
            threads.append(t)

        # Start all threads as close together as possible
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        assert call_count == 1, f"init_func called {call_count} times, expected 1"
        assert lazy.initialized is True
