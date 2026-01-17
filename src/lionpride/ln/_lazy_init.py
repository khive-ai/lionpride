# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe lazy initialization utility.

Provides a reusable pattern for lazy module-level initialization
that only runs once even in multi-threaded environments.
"""

import threading
from collections.abc import Callable

__all__ = ("LazyInit",)


class LazyInit:
    """Thread-safe lazy initialization helper.

    Usage:
        _lazy = LazyInit()
        _MODEL_LIKE = None

        def _do_init():
            global _MODEL_LIKE
            from pydantic import BaseModel
            _MODEL_LIKE = (BaseModel,)

        def my_function(x):
            _lazy.ensure(_do_init)
            # now _MODEL_LIKE is initialized
    """

    __slots__ = ("_initialized", "_lock")

    def __init__(self) -> None:
        self._initialized = False
        self._lock = threading.RLock()

    @property
    def initialized(self) -> bool:
        """Check if initialization has completed."""
        return self._initialized

    def ensure(self, init_func: Callable[[], None]) -> None:
        """Execute init_func only once, thread-safely.

        Args:
            init_func: Function to call for initialization.
                       Must be idempotent as a safety measure.
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            init_func()
            self._initialized = True
