# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import threading
from collections.abc import Awaitable
from typing import Any, TypeVar

import anyio

T = TypeVar("T")

__all__ = ("run_async",)


def run_async(coro: Awaitable[T]) -> T:
    """Run async coroutine from sync context (creates isolated thread with event loop)."""
    result_container: list[Any] = []
    exception_container: list[Exception] = []

    def run_in_thread() -> None:
        """Execute coroutine using anyio.run() in isolated thread."""
        try:

            async def _runner() -> T:
                return await coro

            result = anyio.run(_runner)
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=run_in_thread, daemon=False)
    thread.start()
    thread.join()

    if exception_container:
        raise exception_container[0]
    if not result_container:  # pragma: no cover
        raise RuntimeError("Coroutine did not produce a result")
    return result_container[0]
