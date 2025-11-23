# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for OperationDispatcher edge cases and error paths.

Coverage targets:
- Line 66: ValueError on duplicate registration without override
- Lines 101-104: unregister() branches (exists/not exists)
- Line 131: is_registered() return
- Line 141: clear() method

Complements test_operations_system.py TestOperationDispatcher class.
"""

import pytest

from lionpride.operations.dispatcher import (
    OperationDispatcher,
    get_dispatcher,
    register_operation,
)


class TestDispatcherErrorPaths:
    """Test error handling and edge cases in OperationDispatcher."""

    def test_register_duplicate_without_override_raises_error(self):
        """Line 66: Register duplicate operation without override raises ValueError."""
        dispatcher = OperationDispatcher()

        async def factory1(session, branch, params):
            return "result1"

        async def factory2(session, branch, params):
            return "result2"

        # Register first time - should succeed
        dispatcher.register("duplicate_op", factory1)

        # Register again without override - should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            dispatcher.register("duplicate_op", factory2)

        assert "already registered" in str(exc_info.value)
        assert "override=True" in str(exc_info.value)

        # Verify original factory is still registered
        assert dispatcher.get_factory("duplicate_op") is factory1

    def test_register_duplicate_with_override_succeeds(self):
        """Test override=True replaces existing registration."""
        dispatcher = OperationDispatcher()

        async def factory1(session, branch, params):
            return "result1"

        async def factory2(session, branch, params):
            return "result2"

        dispatcher.register("override_op", factory1)
        assert dispatcher.get_factory("override_op") is factory1

        # Override should succeed
        dispatcher.register("override_op", factory2, override=True)
        assert dispatcher.get_factory("override_op") is factory2

    def test_unregister_existing_operation_returns_true(self):
        """Lines 101-103: unregister() returns True when operation exists."""
        dispatcher = OperationDispatcher()

        async def factory(session, branch, params):
            return "result"

        dispatcher.register("to_remove", factory)
        assert dispatcher.is_registered("to_remove")

        # Unregister should return True
        result = dispatcher.unregister("to_remove")
        assert result is True
        assert not dispatcher.is_registered("to_remove")
        assert dispatcher.get_factory("to_remove") is None

    def test_unregister_nonexistent_operation_returns_false(self):
        """Line 104: unregister() returns False when operation doesn't exist."""
        dispatcher = OperationDispatcher()

        # Unregister non-existent operation should return False
        result = dispatcher.unregister("never_existed")
        assert result is False

    def test_is_registered_returns_true_for_existing(self):
        """Line 131: is_registered() returns True for existing operation."""
        dispatcher = OperationDispatcher()

        async def factory(session, branch, params):
            return "result"

        dispatcher.register("exists", factory)

        # Should return True
        assert dispatcher.is_registered("exists") is True

    def test_is_registered_returns_false_for_nonexistent(self):
        """Line 131: is_registered() returns False for non-existent operation."""
        dispatcher = OperationDispatcher()

        # Should return False
        assert dispatcher.is_registered("does_not_exist") is False

    def test_clear_removes_all_registrations(self):
        """Line 141: clear() removes all registered operations."""
        dispatcher = OperationDispatcher()

        async def factory1(session, branch, params):
            return "result1"

        async def factory2(session, branch, params):
            return "result2"

        async def factory3(session, branch, params):
            return "result3"

        dispatcher.register("op1", factory1)
        dispatcher.register("op2", factory2)
        dispatcher.register("op3", factory3)

        assert len(dispatcher.list_types()) == 3
        assert dispatcher.is_registered("op1")
        assert dispatcher.is_registered("op2")
        assert dispatcher.is_registered("op3")

        # Clear should remove all
        dispatcher.clear()

        assert len(dispatcher.list_types()) == 0
        assert not dispatcher.is_registered("op1")
        assert not dispatcher.is_registered("op2")
        assert not dispatcher.is_registered("op3")
        assert dispatcher.get_factory("op1") is None

    def test_clear_on_empty_dispatcher(self):
        """Test clear() on empty dispatcher is safe."""
        dispatcher = OperationDispatcher()
        assert len(dispatcher.list_types()) == 0

        # Should not raise
        dispatcher.clear()
        assert len(dispatcher.list_types()) == 0


class TestDispatcherDecorator:
    """Test register_operation decorator."""

    def test_decorator_registers_factory(self):
        """Test decorator successfully registers operation."""

        # Note: dispatcher instance not needed - decorator uses global dispatcher
        @register_operation("decorated_op")
        async def decorated_factory(session, branch, params):
            return "decorated"

        # Should be registered in global dispatcher
        global_dispatcher = get_dispatcher()
        factory = global_dispatcher.get_factory("decorated_op")
        assert factory is decorated_factory

    def test_decorator_with_override(self):
        """Test decorator with override=True."""
        dispatcher = get_dispatcher()

        # Register initial factory
        @register_operation("decorator_override")
        async def factory1(session, branch, params):
            return "v1"

        # Override with decorator
        @register_operation("decorator_override", override=True)
        async def factory2(session, branch, params):
            return "v2"

        # Should have the second factory
        factory = dispatcher.get_factory("decorator_override")
        assert factory is factory2


class TestDispatcherRepr:
    """Test __repr__ for debugging."""

    def test_repr_shows_registration_count(self):
        """Test __repr__ includes number of registered operations."""
        dispatcher = OperationDispatcher()

        repr_empty = repr(dispatcher)
        assert "OperationDispatcher" in repr_empty
        assert "registered=0" in repr_empty

        async def factory1(session, branch, params):
            pass

        async def factory2(session, branch, params):
            pass

        dispatcher.register("op1", factory1)
        dispatcher.register("op2", factory2)

        repr_with_ops = repr(dispatcher)
        assert "OperationDispatcher" in repr_with_ops
        assert "registered=2" in repr_with_ops


class TestDispatcherThreadSafety:
    """Test dispatcher behavior with singleton pattern."""

    def test_global_dispatcher_persistence(self):
        """Test global dispatcher persists registrations across get_dispatcher() calls."""
        dispatcher1 = get_dispatcher()

        async def persistent_factory(session, branch, params):
            return "persistent"

        dispatcher1.register("persistent_op", persistent_factory)

        # Get dispatcher again - should be same instance
        dispatcher2 = get_dispatcher()
        assert dispatcher2 is dispatcher1
        assert dispatcher2.is_registered("persistent_op")
        assert dispatcher2.get_factory("persistent_op") is persistent_factory

    def test_clear_affects_global_dispatcher(self):
        """Test clear() on global dispatcher removes all registrations."""
        dispatcher = get_dispatcher()

        async def temp_factory(session, branch, params):
            return "temp"

        dispatcher.register("temp_op", temp_factory)
        assert dispatcher.is_registered("temp_op")

        # Clear global dispatcher
        dispatcher.clear()

        # Should be empty
        assert not dispatcher.is_registered("temp_op")
        assert len(dispatcher.list_types()) == 0


class TestDispatcherEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_register_empty_string_operation_type(self):
        """Test registering with empty string operation type."""
        dispatcher = OperationDispatcher()

        async def factory(session, branch, params):
            return "result"

        # Empty string should work (no validation in dispatcher)
        dispatcher.register("", factory)
        assert dispatcher.is_registered("")
        assert dispatcher.get_factory("") is factory

    def test_unregister_then_reregister(self):
        """Test unregister followed by re-registration."""
        dispatcher = OperationDispatcher()

        async def factory1(session, branch, params):
            return "v1"

        async def factory2(session, branch, params):
            return "v2"

        # Register, unregister, then register different factory
        dispatcher.register("reuse_op", factory1)
        assert dispatcher.unregister("reuse_op") is True

        # Should be able to register without override after unregister
        dispatcher.register("reuse_op", factory2)
        assert dispatcher.get_factory("reuse_op") is factory2

    def test_multiple_unregister_calls(self):
        """Test unregister returns False on subsequent calls."""
        dispatcher = OperationDispatcher()

        async def factory(session, branch, params):
            return "result"

        dispatcher.register("multi_unregister", factory)

        # First unregister returns True
        assert dispatcher.unregister("multi_unregister") is True

        # Second unregister returns False
        assert dispatcher.unregister("multi_unregister") is False

        # Third unregister returns False
        assert dispatcher.unregister("multi_unregister") is False

    def test_list_types_returns_copy(self):
        """Test list_types() returns independent list."""
        dispatcher = OperationDispatcher()

        async def factory(session, branch, params):
            pass

        dispatcher.register("op1", factory)

        types_list = dispatcher.list_types()
        assert "op1" in types_list

        # Modifying returned list shouldn't affect dispatcher
        types_list.append("fake_op")
        assert "fake_op" not in dispatcher.list_types()
        assert not dispatcher.is_registered("fake_op")
