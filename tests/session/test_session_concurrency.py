# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Session concurrency and error handling.

Campaign 1: Foundation & Error Infrastructure
- 01 Session Concurrency: Full Internal Locking (anyio.Lock)
- 03 Error Handling: Error Hooks + Compensation
"""

from __future__ import annotations

import anyio
import pytest

from lionpride.errors import (
    LionConnectionError,
    LionprideError,
    LionTimeoutError,
    ValidationError,
)
from lionpride.session import Branch, ErrorAction, Session

# =============================================================================
# Session Lock Tests (Decision 01)
# =============================================================================


class TestSessionLock:
    """Test Session async lock for concurrent access protection."""

    def test_session_has_lock_property(self):
        """Session exposes lock property."""
        session = Session()
        assert hasattr(session, "lock")
        assert isinstance(session.lock, anyio.Lock)

    def test_lock_is_lazily_initialized(self):
        """Lock is created on first access."""
        session = Session()
        # Access internal _lock before property
        assert session._lock is None
        # Access property triggers creation
        lock = session.lock
        assert lock is not None
        assert session._lock is lock

    def test_same_lock_returned_on_multiple_access(self):
        """Same lock instance returned on multiple accesses."""
        session = Session()
        lock1 = session.lock
        lock2 = session.lock
        assert lock1 is lock2


class TestConcurrentBranchCreation:
    """Test concurrent branch creation doesn't race."""

    @pytest.mark.anyio
    async def test_concurrent_create_branch(self):
        """Multiple concurrent create_branch calls don't corrupt state."""
        session = Session()
        num_branches = 10

        # create_branch is sync, so this tests state consistency
        for i in range(num_branches):
            session.create_branch(name=f"branch_{i}")

        # All branches should be created
        assert len(session.branches) == num_branches

        # All should have unique names
        names = [b.name for b in session.branches]
        assert len(set(names)) == num_branches


class TestConcurrentFork:
    """Test concurrent fork operations don't race."""

    @pytest.mark.anyio
    async def test_concurrent_fork_same_branch(self):
        """Multiple concurrent forks from same source don't corrupt state."""
        session = Session()
        source = session.create_branch(name="source")
        num_forks = 5
        results: list[Branch] = []

        async def fork_worker(idx: int) -> None:
            forked = await session.fork(source, name=f"fork_{idx}")
            results.append(forked)

        async with anyio.create_task_group() as tg:
            for i in range(num_forks):
                tg.start_soon(fork_worker, i)

        # All forks should complete
        assert len(results) == num_forks

        # Total branches: source + num_forks
        assert len(session.branches) == 1 + num_forks

        # All forks should have unique names
        fork_names = [r.name for r in results]
        assert len(set(fork_names)) == num_forks

        # All forks should have forked_from metadata
        for fork in results:
            assert "forked_from" in fork.metadata
            assert fork.metadata["forked_from"]["branch_id"] == str(source.id)


# =============================================================================
# Error Hook Tests (Decision 03)
# =============================================================================


class TestErrorAction:
    """Test ErrorAction enum."""

    def test_error_actions_exist(self):
        """All expected error actions exist."""
        assert hasattr(ErrorAction, "RETRY")
        assert hasattr(ErrorAction, "CONTINUE")
        assert hasattr(ErrorAction, "ABORT")

    def test_error_actions_are_distinct(self):
        """Error actions have distinct values."""
        assert ErrorAction.RETRY != ErrorAction.CONTINUE
        assert ErrorAction.CONTINUE != ErrorAction.ABORT
        assert ErrorAction.RETRY != ErrorAction.ABORT


class TestErrorHookRegistration:
    """Test error hook registration and management."""

    def test_register_error_hook(self):
        """Error hook can be registered."""
        session = Session()

        @session.on_operation_error
        def my_hook(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.ABORT

        assert len(session._error_hooks) == 1
        assert session._error_hooks[0] is my_hook

    def test_register_multiple_hooks(self):
        """Multiple error hooks can be registered."""
        session = Session()

        @session.on_operation_error
        def hook1(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.ABORT

        @session.on_operation_error
        def hook2(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.CONTINUE

        assert len(session._error_hooks) == 2

    def test_remove_error_hook(self):
        """Error hook can be removed."""
        session = Session()

        def my_hook(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.ABORT

        session.on_operation_error(my_hook)
        assert len(session._error_hooks) == 1

        result = session.remove_error_hook(my_hook)
        assert result is True
        assert len(session._error_hooks) == 0

    def test_remove_nonexistent_hook(self):
        """Removing nonexistent hook returns False."""
        session = Session()

        def my_hook(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.ABORT

        result = session.remove_error_hook(my_hook)
        assert result is False

    def test_clear_error_hooks(self):
        """All error hooks can be cleared."""
        session = Session()

        @session.on_operation_error
        def hook1(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.ABORT

        @session.on_operation_error
        def hook2(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.CONTINUE

        count = session.clear_error_hooks()
        assert count == 2
        assert len(session._error_hooks) == 0


class TestErrorHookDispatch:
    """Test error hook dispatch behavior."""

    def test_dispatch_with_no_hooks_returns_abort(self):
        """No hooks registered returns ABORT."""
        session = Session()
        error = LionprideError("test error")

        action = session._dispatch_error_hooks(error, {})
        assert action == ErrorAction.ABORT

    def test_dispatch_returns_first_non_abort(self):
        """First non-ABORT action wins."""
        session = Session()
        call_order = []

        @session.on_operation_error
        def hook1(error: LionprideError, context: dict) -> ErrorAction:
            call_order.append("hook1")
            return ErrorAction.ABORT

        @session.on_operation_error
        def hook2(error: LionprideError, context: dict) -> ErrorAction:
            call_order.append("hook2")
            return ErrorAction.RETRY

        @session.on_operation_error
        def hook3(error: LionprideError, context: dict) -> ErrorAction:
            call_order.append("hook3")
            return ErrorAction.CONTINUE

        error = LionprideError("test")
        action = session._dispatch_error_hooks(error, {})

        # hook2 returns RETRY (first non-ABORT)
        assert action == ErrorAction.RETRY
        # All hooks up to winner should be called
        assert call_order == ["hook1", "hook2"]

    def test_dispatch_passes_context(self):
        """Context is passed to hooks."""
        session = Session()
        received_context = {}

        @session.on_operation_error
        def hook(error: LionprideError, context: dict) -> ErrorAction:
            received_context.update(context)
            return ErrorAction.CONTINUE

        error = LionprideError("test")
        session._dispatch_error_hooks(error, {"operation": "test", "retry": 0})

        assert received_context == {"operation": "test", "retry": 0}

    def test_dispatch_passes_error(self):
        """Error is passed to hooks."""
        session = Session()
        received_error = None

        @session.on_operation_error
        def hook(error: LionprideError, context: dict) -> ErrorAction:
            nonlocal received_error
            received_error = error
            return ErrorAction.CONTINUE

        error = LionConnectionError("connection failed", retryable=True)
        session._dispatch_error_hooks(error, {})

        assert received_error is error
        assert received_error.retryable is True

    def test_dispatch_handles_hook_exception(self):
        """Hook that raises exception is skipped."""
        session = Session()

        @session.on_operation_error
        def failing_hook(error: LionprideError, context: dict) -> ErrorAction:
            raise RuntimeError("hook failed")

        @session.on_operation_error
        def working_hook(error: LionprideError, context: dict) -> ErrorAction:
            return ErrorAction.CONTINUE

        error = LionprideError("test")
        action = session._dispatch_error_hooks(error, {})

        # Failing hook skipped, working hook returns CONTINUE
        assert action == ErrorAction.CONTINUE


class TestErrorHookRetryBehavior:
    """Test error hook integration with retry logic."""

    def test_retry_hook_for_timeout_errors(self):
        """Hook can return RETRY for timeout errors."""
        session = Session()

        @session.on_operation_error
        def timeout_handler(error: LionprideError, context: dict) -> ErrorAction:
            if isinstance(error, LionTimeoutError) and error.retryable:
                return ErrorAction.RETRY
            return ErrorAction.ABORT

        timeout_err = LionTimeoutError("timed out", retryable=True)
        action = session._dispatch_error_hooks(timeout_err, {})
        assert action == ErrorAction.RETRY

        # Non-retryable timeout
        timeout_err2 = LionTimeoutError("timed out", retryable=False)
        action2 = session._dispatch_error_hooks(timeout_err2, {})
        assert action2 == ErrorAction.ABORT

    def test_continue_hook_for_validation_errors(self):
        """Hook can return CONTINUE for validation errors."""
        session = Session()

        @session.on_operation_error
        def validation_handler(error: LionprideError, context: dict) -> ErrorAction:
            if isinstance(error, ValidationError):
                return ErrorAction.CONTINUE
            return ErrorAction.ABORT

        val_err = ValidationError("invalid input")
        action = session._dispatch_error_hooks(val_err, {})
        assert action == ErrorAction.CONTINUE


# =============================================================================
# Integration Tests
# =============================================================================


class TestSessionDocstring:
    """Verify Session docstring reflects new features."""

    def test_session_docstring_mentions_thread_safety(self):
        """Session docstring mentions thread safety."""
        assert "Thread Safety" in Session.__doc__
        assert "anyio.Lock" in Session.__doc__

    def test_session_docstring_mentions_error_handling(self):
        """Session docstring mentions error handling."""
        assert "Error Handling" in Session.__doc__
        assert "ErrorAction" in Session.__doc__
