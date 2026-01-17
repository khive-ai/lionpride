# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Async batch processing utilities with retry and concurrency control.

Refactored for reduced cyclomatic complexity (33 -> <15) per #116.
"""

from collections.abc import AsyncGenerator, Callable
from typing import Any, ParamSpec, TypeVar

from lionpride.libs.concurrency import (
    Semaphore,
    create_task_group,
    get_cancelled_exc_class,
    is_coro_func,
    move_on_after,
    non_cancel_subgroup,
    run_sync,
    sleep,
)
from lionpride.types import Unset, not_sentinel

from ._lazy_init import LazyInit
from ._to_list import to_list

T = TypeVar("T")
P = ParamSpec("P")

_lazy = LazyInit()
_MODEL_LIKE = None


__all__ = (
    "alcall",
    "bcall",
)


def _do_init() -> None:
    """Initialize Pydantic model type detection."""
    global _MODEL_LIKE
    from pydantic import BaseModel

    _MODEL_LIKE = (BaseModel,)


def _ensure_initialized() -> None:
    """Lazy initialization of Pydantic model type detection."""
    _lazy.ensure(_do_init)


def _validate_func(func: Any) -> Callable:
    """Validate and extract a single callable from func."""
    if callable(func):
        return func

    # Try to extract from iterable
    try:
        func_list = list(func)
    except TypeError:
        raise ValueError("func must be callable or an iterable containing one callable.")

    if len(func_list) != 1 or not callable(func_list[0]):
        raise ValueError("Only one callable function is allowed.")

    return func_list[0]


def _normalize_input(
    input_: Any,
    *,
    flatten: bool,
    dropna: bool,
    unique: bool,
    flatten_tuple_set: bool,
) -> list:
    """Normalize input to a list."""
    if flatten or dropna:
        return to_list(
            input_,
            flatten=flatten,
            dropna=dropna,
            unique=unique,
            flatten_tuple_set=flatten_tuple_set,
        )

    if isinstance(input_, list):
        return input_

    # Handle Pydantic models specially
    if _MODEL_LIKE and isinstance(input_, _MODEL_LIKE):
        return [input_]

    # Try to iterate
    try:
        iter(input_)
        return list(input_)
    except TypeError:
        return [input_]


async def _call_with_timeout(
    func: Callable,
    item: Any,
    is_coro: bool,
    timeout: float | None,
    **kwargs,
) -> Any:
    """Call function with optional timeout."""
    if is_coro:
        if timeout is not None:
            with move_on_after(timeout) as cancel_scope:
                result = await func(item, **kwargs)
            if cancel_scope.cancelled_caught:
                raise TimeoutError(f"Function call timed out after {timeout}s")
            return result
        return await func(item, **kwargs)
    else:
        if timeout is not None:
            with move_on_after(timeout) as cancel_scope:
                result = await run_sync(func, item, **kwargs)
            if cancel_scope.cancelled_caught:
                raise TimeoutError(f"Function call timed out after {timeout}s")
            return result
        return await run_sync(func, item, **kwargs)


async def _execute_with_retry(
    func: Callable,
    item: Any,
    index: int,
    *,
    is_coro: bool,
    timeout: float | None,
    initial_delay: float,
    backoff: float,
    max_attempts: int,
    default: Any,
    **kwargs,
) -> tuple[int, Any]:
    """Execute function with retry logic."""
    attempts = 0
    current_delay = initial_delay

    while True:
        try:
            result = await _call_with_timeout(func, item, is_coro, timeout, **kwargs)
            return index, result

        except get_cancelled_exc_class():
            raise

        except Exception:
            attempts += 1
            if attempts <= max_attempts:
                if current_delay:
                    await sleep(current_delay)
                    current_delay *= backoff
            else:
                if not_sentinel(default):
                    return index, default
                raise


async def alcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    *,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    delay_before_start: float = 0,
    retry_initial_delay: float = 0,
    retry_backoff: float = 1,
    retry_default: Any = Unset,
    retry_timeout: float | None = None,
    retry_attempts: int = 0,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[T | BaseException]:
    """Apply function to each list element asynchronously with retry and concurrency control.

    Args:
        input_: List of items to process (or iterable that will be converted)
        func: Callable to apply (sync or async)
        input_flatten: Flatten nested input structures
        input_dropna: Remove None/undefined from input
        input_unique: Remove duplicate inputs (requires flatten)
        input_flatten_tuple_set: Include tuples/sets in flattening
        output_flatten: Flatten nested output structures
        output_dropna: Remove None/undefined from output
        output_unique: Remove duplicate outputs (requires flatten)
        output_flatten_tuple_set: Include tuples/sets in output flattening
        delay_before_start: Initial delay before processing (seconds)
        retry_initial_delay: Initial retry delay (seconds)
        retry_backoff: Backoff multiplier for retry delays
        retry_default: Default value on retry exhaustion (Unset = raise)
        retry_timeout: Timeout per function call (seconds)
        retry_attempts: Maximum retry attempts (0 = no retry)
        max_concurrent: Max concurrent executions (None = unlimited)
        throttle_period: Delay between starting tasks (seconds)
        return_exceptions: Return exceptions instead of raising
        **kwargs: Additional arguments passed to func

    Returns:
        List of results (preserves input order, may include exceptions if return_exceptions=True)

    Raises:
        ValueError: If func is not callable
        TimeoutError: If retry_timeout exceeded
        ExceptionGroup: If return_exceptions=False and tasks raise
    """
    _ensure_initialized()

    func = _validate_func(func)
    input_ = _normalize_input(
        input_,
        flatten=input_flatten,
        dropna=input_dropna,
        unique=input_unique,
        flatten_tuple_set=input_flatten_tuple_set,
    )

    if delay_before_start:
        await sleep(delay_before_start)

    semaphore = Semaphore(max_concurrent) if max_concurrent else None
    throttle_delay = throttle_period or 0
    is_coro = is_coro_func(func)
    n_items = len(input_)
    out: list[Any] = [None] * n_items

    async def task_wrapper(item: Any, idx: int) -> None:
        try:
            if semaphore:
                async with semaphore:
                    _, result = await _execute_with_retry(
                        func,
                        item,
                        idx,
                        is_coro=is_coro,
                        timeout=retry_timeout,
                        initial_delay=retry_initial_delay,
                        backoff=retry_backoff,
                        max_attempts=retry_attempts,
                        default=retry_default,
                        **kwargs,
                    )
            else:
                _, result = await _execute_with_retry(
                    func,
                    item,
                    idx,
                    is_coro=is_coro,
                    timeout=retry_timeout,
                    initial_delay=retry_initial_delay,
                    backoff=retry_backoff,
                    max_attempts=retry_attempts,
                    default=retry_default,
                    **kwargs,
                )
            out[idx] = result
        except BaseException as exc:
            out[idx] = exc
            if not return_exceptions:
                raise

    try:
        async with create_task_group() as tg:
            for idx, item in enumerate(input_):
                tg.start_soon(task_wrapper, item, idx)
                if throttle_delay and idx < n_items - 1:
                    await sleep(throttle_delay)
    except ExceptionGroup as eg:
        if not return_exceptions:
            rest = non_cancel_subgroup(eg)
            if rest is not None:
                raise rest
            raise

    return to_list(
        out,
        flatten=output_flatten,
        dropna=output_dropna,
        unique=output_unique,
        flatten_tuple_set=output_flatten_tuple_set,
    )


async def bcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    batch_size: int,
    **kwargs: Any,
) -> AsyncGenerator[list[T | BaseException], None]:
    """Process input in batches using alcall. Yields results batch by batch.

    Args:
        input_: Items to process
        func: Callable to apply
        batch_size: Number of items per batch
        **kwargs: Arguments passed to alcall (see alcall for details)

    Yields:
        List of results for each batch
    """
    input_ = to_list(input_, flatten=True, dropna=True)

    for i in range(0, len(input_), batch_size):
        batch = input_[i : i + batch_size]
        yield await alcall(batch, func, **kwargs)
