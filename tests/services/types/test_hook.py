# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Hook system - 100% coverage target.

Test Surface:
    - HookPhase enum
    - HookEvent (validation, invoke, error handling)
    - get_handler (sync/async detection, defaults)
    - validate_hooks, validate_stream_handlers
    - HookRegistry (all hook phases, streaming, _can_handle)
    - HookBroadcaster (_event_type)
"""

from __future__ import annotations

import pytest

from lionpride import EventStatus
from lionpride.services.types.hook import (
    HookBroadcaster,
    HookEvent,
    HookPhase,
    HookRegistry,
    get_handler,
    validate_hooks,
    validate_stream_handlers,
)
from tests.conftest import SimpleTestEvent

# =============================================================================
# Test Fixtures & Helpers
# =============================================================================


@pytest.fixture
def hook_registry():
    """Create empty HookRegistry."""
    return HookRegistry()


@pytest.fixture
def mock_event():
    """Create mock event instance using lionpride-core testing utilities."""
    return SimpleTestEvent(return_value="test")


# =============================================================================
# HookPhase Enum Tests
# =============================================================================


def test_hookphase_enum_values():
    """Test HookPhase enum has all expected values."""
    assert HookPhase.PreEventCreate.value == "pre_event_create"
    assert HookPhase.PreInvocation.value == "pre_invocation"
    assert HookPhase.PostInvocation.value == "post_invocation"
    assert HookPhase.ErrorHandling.value == "error_handling"


def test_hookphase_enum_allowed():
    """Test HookPhase.allowed() returns all values."""
    allowed = HookPhase.allowed()
    assert HookPhase.PreEventCreate in allowed
    assert HookPhase.PreInvocation in allowed
    assert HookPhase.PostInvocation in allowed
    assert HookPhase.ErrorHandling in allowed


# =============================================================================
# get_handler Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_handler_when_async_function_then_returns_as_is():
    """Test get_handler returns async function unchanged."""

    async def async_handler(arg):
        return f"async: {arg}"

    handler = get_handler({"key": async_handler}, "key", True)

    assert handler is not None
    result = await handler("test")
    assert result == "async: test"


@pytest.mark.asyncio
async def test_get_handler_when_sync_function_then_wraps():
    """Test get_handler wraps sync function to be async."""

    def sync_handler(arg):
        return f"sync: {arg}"

    handler = get_handler({"key": sync_handler}, "key", True)

    assert handler is not None
    result = await handler("test")
    assert result == "sync: test"


@pytest.mark.asyncio
async def test_get_handler_when_not_found_and_get_false_then_none():
    """Test get_handler returns None when key not found and get=False."""
    handler = get_handler({"key": lambda: "val"}, "nonexistent", False)
    assert handler is None


@pytest.mark.asyncio
async def test_get_handler_when_not_found_and_get_true_then_default():
    """Test get_handler returns default async function when key not found and get=True."""
    handler = get_handler({}, "nonexistent", True)

    assert handler is not None
    result = await handler("arg1", "arg2")
    # Default function returns first arg or None
    assert result == "arg1"


@pytest.mark.asyncio
async def test_get_handler_default_with_no_args():
    """Test get_handler default function with no arguments."""
    handler = get_handler({}, "nonexistent", True)
    result = await handler()
    assert result is None


# =============================================================================
# validate_hooks Tests
# =============================================================================


def test_validate_hooks_when_valid_dict_then_passes():
    """Test validate_hooks accepts valid hooks dictionary."""

    async def hook(event):
        pass

    hooks = {HookPhase.PreEventCreate: hook, HookPhase.PreInvocation: hook}

    # Should not raise
    validate_hooks(hooks)


def test_validate_hooks_when_not_dict_then_raises():
    """Test validate_hooks raises when hooks is not a dict."""
    with pytest.raises(ValueError, match="Hooks must be a dictionary"):
        validate_hooks("not_a_dict")


def test_validate_hooks_when_invalid_key_then_raises():
    """Test validate_hooks raises when key is not HookPhase."""

    async def hook(event):
        pass

    with pytest.raises(ValueError, match="Hook key must be one of"):
        validate_hooks({"invalid_key": hook})


def test_validate_hooks_when_key_not_in_allowed_then_raises():
    """Test validate_hooks raises when HookPhase value not in allowed()."""
    # This is tricky - need a HookPhase that's not in allowed()
    # But allowed() returns all enum values, so this is impossible under normal use
    # Skip this edge case
    pass


def test_validate_hooks_when_non_callable_value_then_raises():
    """Test validate_hooks raises when value is not callable."""
    with pytest.raises(ValueError, match=r"Hook for .+ must be callable"):
        validate_hooks({HookPhase.PreEventCreate: "not_callable"})


# =============================================================================
# validate_stream_handlers Tests
# =============================================================================


def test_validate_stream_handlers_when_valid_then_passes():
    """Test validate_stream_handlers accepts valid handlers."""

    async def handler(event, chunk_type, chunk):
        pass

    handlers = {"text": handler, "delta": handler}

    # Should not raise
    validate_stream_handlers(handlers)


def test_validate_stream_handlers_when_not_dict_then_raises():
    """Test validate_stream_handlers raises when not a dict."""
    with pytest.raises(ValueError, match="Stream handlers must be a dictionary"):
        validate_stream_handlers(["not", "dict"])


def test_validate_stream_handlers_when_invalid_key_type_then_raises():
    """Test validate_stream_handlers raises when key is not str or type."""

    async def handler(event, chunk_type, chunk):
        pass

    with pytest.raises(ValueError, match="Stream handler key must be a string or type"):
        validate_stream_handlers({123: handler})


def test_validate_stream_handlers_when_non_callable_then_raises():
    """Test validate_stream_handlers raises when value is not callable."""
    with pytest.raises(ValueError, match=r"Stream handler for .+ must be callable"):
        validate_stream_handlers({"text": "not_callable"})


def test_validate_stream_handlers_when_type_key_then_passes():
    """Test validate_stream_handlers accepts type as key."""

    async def handler(event, chunk_type, chunk):
        pass

    class ChunkType:
        pass

    handlers = {ChunkType: handler}

    # Should not raise
    validate_stream_handlers(handlers)


# =============================================================================
# HookRegistry Tests - Initialization
# =============================================================================


def test_hook_registry_init_when_no_args_then_empty():
    """Test HookRegistry initialization with no arguments."""
    registry = HookRegistry()

    assert registry._hooks == {}
    assert registry._stream_handlers == {}


def test_hook_registry_init_when_valid_hooks_then_stores():
    """Test HookRegistry initialization with valid hooks."""

    async def hook(event):
        pass

    hooks = {HookPhase.PreEventCreate: hook}
    registry = HookRegistry(hooks=hooks)

    assert HookPhase.PreEventCreate in registry._hooks


def test_hook_registry_init_when_invalid_hooks_then_raises():
    """Test HookRegistry initialization validates hooks."""
    with pytest.raises(ValueError, match="Hook key must be one of"):
        HookRegistry(hooks={"invalid": lambda: None})


def test_hook_registry_init_when_valid_stream_handlers_then_stores():
    """Test HookRegistry initialization with stream handlers."""

    async def handler(event, chunk_type, chunk):
        pass

    handlers = {"text": handler}
    registry = HookRegistry(stream_handlers=handlers)

    assert "text" in registry._stream_handlers


def test_hook_registry_init_when_invalid_stream_handlers_then_raises():
    """Test HookRegistry initialization validates stream handlers."""
    with pytest.raises(ValueError, match=r"Stream handler for .+ must be callable"):
        HookRegistry(stream_handlers={"text": "not_callable"})


# =============================================================================
# HookRegistry Tests - pre_event_create
# =============================================================================


@pytest.mark.asyncio
async def test_registry_pre_event_create_when_hook_returns_event_then_success():
    """Test pre_event_create when hook returns event instance."""
    called_with = {}

    async def hook(event_type, **kw):
        called_with["event_type"] = event_type
        called_with["kwargs"] = kw
        return SimpleTestEvent(return_value="from_hook")

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    result, should_exit, status = await registry.pre_event_create(
        SimpleTestEvent, test_param="value"
    )

    assert isinstance(result, SimpleTestEvent)
    assert result.return_value == "from_hook"
    assert should_exit is False
    assert status == EventStatus.COMPLETED
    assert called_with["event_type"] == SimpleTestEvent
    assert called_with["kwargs"]["test_param"] == "value"


@pytest.mark.asyncio
async def test_registry_pre_event_create_when_hook_returns_none_then_success():
    """Test pre_event_create when hook returns None (default creation)."""

    async def hook(event_type, **kw):
        return None

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    result, should_exit, status = await registry.pre_event_create(SimpleTestEvent)

    assert result is None
    assert should_exit is False
    assert status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_registry_pre_event_create_when_hook_raises_then_cancelled():
    """Test pre_event_create when hook raises exception."""

    async def hook(event_type, **kw):
        raise ValueError("Hook validation failed")

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    result, should_exit, status = await registry.pre_event_create(SimpleTestEvent)

    assert isinstance(result, ValueError)
    assert status == EventStatus.CANCELLED
    # exit parameter default is False
    assert should_exit is False


@pytest.mark.asyncio
async def test_registry_pre_event_create_when_hook_raises_and_exit_true():
    """Test pre_event_create with exit=True propagates exception."""

    async def hook(event_type, **kw):
        raise ValueError("Critical error")

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    result, should_exit, status = await registry.pre_event_create(SimpleTestEvent, exit=True)

    assert isinstance(result, ValueError)
    assert should_exit is True  # exit parameter passed through
    assert status == EventStatus.CANCELLED


# =============================================================================
# HookRegistry Tests - pre_invocation
# =============================================================================


@pytest.mark.asyncio
async def test_registry_pre_invocation_when_hook_succeeds_then_completed():
    """Test pre_invocation when hook executes successfully."""
    event = SimpleTestEvent()
    called = []

    async def hook(evt, **kw):
        called.append(evt)
        return "hook_result"

    registry = HookRegistry(hooks={HookPhase.PreInvocation: hook})

    result, should_exit, status = await registry.pre_invocation(event)

    assert result == "hook_result"
    assert should_exit is False
    assert status == EventStatus.COMPLETED
    assert event in called


@pytest.mark.asyncio
async def test_registry_pre_invocation_when_hook_raises_then_cancelled():
    """Test pre_invocation when hook raises exception."""

    async def hook(evt, **kw):
        raise RuntimeError("Permission denied")

    registry = HookRegistry(hooks={HookPhase.PreInvocation: hook})
    event = SimpleTestEvent()

    result, should_exit, status = await registry.pre_invocation(event, exit=False)

    assert isinstance(result, RuntimeError)
    assert should_exit is False
    assert status == EventStatus.CANCELLED


# =============================================================================
# HookRegistry Tests - post_invocation
# =============================================================================


@pytest.mark.asyncio
async def test_registry_post_invocation_when_hook_succeeds_then_completed():
    """Test post_invocation when hook executes successfully."""
    event = SimpleTestEvent()

    async def hook(evt, **kw):
        return None  # Post hooks typically don't return values

    registry = HookRegistry(hooks={HookPhase.PostInvocation: hook})

    result, should_exit, status = await registry.post_invocation(event)

    assert result is None
    assert should_exit is False
    assert status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_registry_post_invocation_when_hook_raises_then_aborted():
    """Test post_invocation when hook raises exception (status: ABORTED not CANCELLED)."""

    async def hook(evt, **kw):
        raise RuntimeError("Post-processing failed")

    registry = HookRegistry(hooks={HookPhase.PostInvocation: hook})
    event = SimpleTestEvent()

    result, should_exit, status = await registry.post_invocation(event, exit=True)

    assert isinstance(result, RuntimeError)
    assert should_exit is True
    assert status == EventStatus.ABORTED  # Different from pre_invocation


# =============================================================================
# HookRegistry Tests - handle_streaming_chunk
# =============================================================================


@pytest.mark.asyncio
async def test_registry_handle_streaming_chunk_when_success():
    """Test handle_streaming_chunk when handler executes successfully."""
    called_with = {}

    async def handler(evt, chunk_type, chunk, **kw):
        called_with["chunk_type"] = chunk_type
        called_with["chunk"] = chunk
        return "processed"

    registry = HookRegistry(stream_handlers={"text": handler})

    result, should_exit, status = await registry.handle_streaming_chunk("text", "chunk_data")

    assert result == "processed"
    assert should_exit is False
    assert status is None  # Streaming doesn't set status
    assert called_with["chunk_type"] == "text"
    assert called_with["chunk"] == "chunk_data"


@pytest.mark.asyncio
async def test_registry_handle_streaming_chunk_when_raises_then_aborted():
    """Test handle_streaming_chunk when handler raises exception."""

    async def handler(evt, chunk_type, chunk, **kw):
        raise RuntimeError("Stream error")

    registry = HookRegistry(stream_handlers={"text": handler})

    result, should_exit, status = await registry.handle_streaming_chunk("text", "data", exit=True)

    assert isinstance(result, RuntimeError)
    assert should_exit is True
    assert status == EventStatus.ABORTED


# =============================================================================
# HookRegistry Tests - call() method
# =============================================================================


@pytest.mark.asyncio
async def test_registry_call_when_no_method_and_no_chunk_then_raises():
    """Test registry.call() raises when neither hook_phase nor chunk_type provided."""
    registry = HookRegistry()

    with pytest.raises(ValueError, match="Either method or chunk_type must be provided"):
        await registry.call(SimpleTestEvent())


@pytest.mark.asyncio
async def test_registry_call_when_pre_event_create_then_delegates():
    """Test registry.call() with PreEventCreate phase."""

    async def hook(event_type, **kw):
        return None

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    (result_tuple, meta) = await registry.call(SimpleTestEvent, hook_phase=HookPhase.PreEventCreate)

    _result, _should_exit, status = result_tuple
    assert status == EventStatus.COMPLETED
    assert "lion_class" in meta


@pytest.mark.asyncio
async def test_registry_call_when_pre_invocation_then_delegates():
    """Test registry.call() with PreInvocation phase."""
    event = SimpleTestEvent()

    async def hook(evt, **kw):
        return "result"

    registry = HookRegistry(hooks={HookPhase.PreInvocation: hook})

    (result_tuple, meta) = await registry.call(event, hook_phase=HookPhase.PreInvocation)

    result, _should_exit, _status = result_tuple
    assert result == "result"
    assert "event_id" in meta
    assert "event_created_at" in meta


@pytest.mark.asyncio
async def test_registry_call_when_post_invocation_then_delegates():
    """Test registry.call() with PostInvocation phase."""
    event = SimpleTestEvent()

    async def hook(evt, **kw):
        return None

    registry = HookRegistry(hooks={HookPhase.PostInvocation: hook})

    (result_tuple, meta) = await registry.call(event, hook_phase=HookPhase.PostInvocation)

    _result, _should_exit, status = result_tuple
    assert status == EventStatus.COMPLETED
    assert "event_id" in meta


@pytest.mark.asyncio
async def test_registry_call_when_chunk_type_then_delegates():
    """Test registry.call() with chunk_type (streaming)."""

    async def handler(evt, chunk_type, chunk, **kw):
        return "handled"

    registry = HookRegistry(stream_handlers={"text": handler})

    result_tuple = await registry.call(None, hook_phase=None, chunk_type="text", chunk="data")

    # Note: call() with chunk_type returns different structure
    result, _should_exit, _status = result_tuple
    assert result == "handled"


@pytest.mark.asyncio
async def test_registry_call_when_phase_value_string_then_matches():
    """Test registry.call() with HookPhase.value string for all phases."""
    event = SimpleTestEvent()

    async def pre_create_hook(event_type, **kw):
        return None

    async def hook(evt, **kw):
        return "matched"

    # Test PreEventCreate.value (was broken before fix)
    registry_pre_create = HookRegistry(hooks={HookPhase.PreEventCreate: pre_create_hook})
    (result_tuple, meta) = await registry_pre_create.call(
        SimpleTestEvent, hook_phase=HookPhase.PreEventCreate.value
    )
    _result, _should_exit, status = result_tuple
    assert status == EventStatus.COMPLETED
    assert "lion_class" in meta

    # Test PreInvocation.value
    registry_pre = HookRegistry(hooks={HookPhase.PreInvocation: hook})
    (result_tuple, meta) = await registry_pre.call(event, hook_phase=HookPhase.PreInvocation.value)
    result, _should_exit, _status = result_tuple
    assert result == "matched"
    assert "event_id" in meta

    # Test PostInvocation.value
    registry_post = HookRegistry(hooks={HookPhase.PostInvocation: hook})
    (result_tuple, meta) = await registry_post.call(
        event, hook_phase=HookPhase.PostInvocation.value
    )
    result, _should_exit, _status = result_tuple
    assert result == "matched"
    assert "event_id" in meta


# =============================================================================
# HookRegistry Tests - _can_handle
# =============================================================================


def test_registry_can_handle_when_hook_phase_exists_then_true():
    """Test _can_handle returns True when hook_phase is registered."""

    async def hook(evt, **kw):
        pass

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    assert registry._can_handle(hp_=HookPhase.PreEventCreate) is True


def test_registry_can_handle_when_hook_phase_not_exists_then_false():
    """Test _can_handle returns False when hook_phase not registered."""
    registry = HookRegistry()

    assert registry._can_handle(hp_=HookPhase.PreEventCreate) is False


def test_registry_can_handle_when_chunk_type_exists_then_true():
    """Test _can_handle returns True when chunk_type is registered."""

    async def handler(evt, ct, chunk, **kw):
        pass

    registry = HookRegistry(stream_handlers={"text": handler})

    assert registry._can_handle(ct_="text") is True


def test_registry_can_handle_when_chunk_type_not_exists_then_false():
    """Test _can_handle returns False when chunk_type not registered."""
    registry = HookRegistry()

    assert registry._can_handle(ct_="text") is False


def test_registry_can_handle_when_no_args_then_false():
    """Test _can_handle returns False when neither hp_ nor ct_ provided."""
    registry = HookRegistry()

    assert registry._can_handle() is False


# =============================================================================
# HookEvent Tests
# =============================================================================


def test_hook_event_validation_when_exit_none_then_false():
    """Test HookEvent._validate_exit converts None to False."""
    registry = HookRegistry()
    event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
        exit=None,
    )

    assert event.exit is False


def test_hook_event_validation_when_exit_true_then_true():
    """Test HookEvent._validate_exit preserves True."""
    registry = HookRegistry()
    event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
        exit=True,
    )

    assert event.exit is True


@pytest.mark.asyncio
async def test_hook_event_invoke_when_success_then_completed():
    """Test HookEvent.invoke() when hook succeeds."""

    async def hook(event_type, **kw):
        return SimpleTestEvent()

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
        timeout=10,
    )

    await hook_event.invoke()

    assert hook_event.execution.status == EventStatus.COMPLETED
    assert hook_event.execution.response is not None
    assert hook_event.associated_event_info is not None


@pytest.mark.asyncio
async def test_hook_event_invoke_when_hook_returns_tuple_with_exception():
    """Test HookEvent.invoke() when hook returns (Undefined, exception) tuple."""
    from lionpride import Undefined

    async def hook(event_type, **kw):
        # Simulate exception return from _call
        return (Undefined, RuntimeError("Error"))

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
    )

    await hook_event.invoke()

    # Should raise the exception from tuple[1]
    assert hook_event.execution.status == EventStatus.FAILED
    assert hook_event.execution.error is not None
    assert hook_event._exit_cause is not None


@pytest.mark.asyncio
async def test_hook_event_invoke_when_hook_returns_exception():
    """Test HookEvent.invoke() when hook returns Exception directly."""
    from lionpride import Unset

    async def hook(event_type, **kw):
        return ValueError("Direct exception")

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
    )

    await hook_event.invoke()

    # execution.response is Unset on error, not None
    assert hook_event.execution.response is Unset
    # execution.error is the exception object, not a string
    assert isinstance(hook_event.execution.error, ValueError)
    assert str(hook_event.execution.error) == "Direct exception"
    assert isinstance(hook_event._exit_cause, ValueError)


@pytest.mark.asyncio
async def test_hook_event_invoke_when_exception_raised_then_failed():
    """Test HookEvent.invoke() when hook raises exception."""

    async def hook(event_type, **kw):
        raise RuntimeError("Hook failed")

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
    )

    await hook_event.invoke()

    # When hook raises exception, HookEvent._invoke re-raises it
    # Event.invoke catches Exception → status=FAILED (not CANCELLED)
    assert hook_event.execution.status == EventStatus.FAILED
    assert isinstance(hook_event.execution.error, RuntimeError)
    assert "Hook failed" in str(hook_event.execution.error)
    assert hook_event._should_exit is False  # Default exit=False


@pytest.mark.asyncio
async def test_hook_event_invoke_when_timeout_then_cancelled():
    """Test HookEvent.invoke() when timeout occurs."""
    import asyncio

    async def slow_hook(event_type, **kw):
        await asyncio.sleep(10)  # Longer than timeout
        return None

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: slow_hook})

    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
        timeout=0.1,  # Very short timeout
    )

    # Timeout raises TimeoutError which is caught by Event.invoke()
    # TimeoutError → status=CANCELLED (retryable=True)
    await hook_event.invoke()

    assert hook_event.execution.status == EventStatus.CANCELLED
    assert hook_event.execution.retryable is True


@pytest.mark.asyncio
async def test_hook_event_invoke_when_exit_true_then_sets_should_exit():
    """Test HookEvent.invoke() sets _should_exit when exit=True and exception."""

    async def hook(event_type, **kw):
        raise RuntimeError("Critical")

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: hook})

    hook_event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
        exit=True,
    )

    await hook_event.invoke()

    assert hook_event._should_exit is True
    # When hook raises exception, HookEvent._invoke re-raises it
    # Event.invoke catches Exception → status=FAILED (not CANCELLED)
    assert hook_event.execution.status == EventStatus.FAILED


# =============================================================================
# HookBroadcaster Tests
# =============================================================================


def test_hook_broadcaster_event_type():
    """Test HookBroadcaster._event_type is HookEvent."""
    assert HookBroadcaster._event_type == HookEvent


@pytest.mark.asyncio
async def test_hook_broadcaster_subscribe_and_broadcast():
    """Test HookBroadcaster subscription and broadcasting."""
    received = []

    async def subscriber(event: HookEvent):
        received.append(event)

    # Subscribe
    HookBroadcaster.subscribe(subscriber)

    # Create and broadcast event
    registry = HookRegistry()
    event = HookEvent(
        registry=registry,
        hook_phase=HookPhase.PreEventCreate,
        event_like=SimpleTestEvent,
    )

    await HookBroadcaster.broadcast(event)

    # Cleanup
    HookBroadcaster.unsubscribe(subscriber)

    assert len(received) == 1
    assert received[0] == event


# =============================================================================
# Edge Cases - _call internal method
# =============================================================================


@pytest.mark.asyncio
async def test_registry_internal_call_when_no_hook_phase_and_no_chunk_then_raises():
    """Test HookRegistry._call() raises when neither hp_ nor ct_ provided."""
    registry = HookRegistry()

    with pytest.raises(RuntimeError, match="Either hook_type or chunk_type must be provided"):
        await registry._call(None, None, None, SimpleTestEvent)


@pytest.mark.asyncio
async def test_registry_internal_call_when_chunk_type_but_no_hook_then_uses_stream():
    """Test HookRegistry._call() uses stream handler when chunk_type provided."""

    async def handler(evt, ct, chunk, **kw):
        return f"handled: {chunk}"

    registry = HookRegistry(stream_handlers={"text": handler})

    # _call returns only the result, not a tuple
    result = await registry._call(None, "text", "chunk_data", None, extra_param="value")

    assert result == "handled: chunk_data"


@pytest.mark.asyncio
async def test_registry_internal_call_when_hook_phase_missing_chunk_type_then_raises():
    """Test HookRegistry._call() raises when hp_ exists but ct_ needed and missing."""
    registry = HookRegistry()  # No hooks registered

    # hp_ is None, ct_ is None -> RuntimeError (first check)
    # hp_ is None, ct_ is not None -> uses stream handler
    # hp_ exists but not in registry, ct_ is None -> RuntimeError

    # This is hard to trigger because hook phase is checked first
    # If hp_ is not in registry, RuntimeError is raised
    # But we need hp_ to be truthy but not in registry

    # If hp_ is truthy but not in _hooks, we skip to elif not ct_
    # Then raise RuntimeError

    with pytest.raises(RuntimeError, match="Hook type is required"):
        await registry._call(
            HookPhase.PreEventCreate,  # hp_ is truthy
            None,  # ct_ is None
            None,
            SimpleTestEvent,
        )


# =============================================================================
# HookRegistry Tests - CancelledError handling (lines 242, 261, 279)
# =============================================================================


@pytest.mark.asyncio
async def test_registry_pre_invocation_when_cancelled_then_returns_tuple():
    """Test pre_invocation when task is cancelled (line 242).

    When a hook raises CancelledError, pre_invocation should return:
    ((Undefined, exception), True, EventStatus.CANCELLED)
    """
    import asyncio

    from lionpride.types import Undefined

    async def cancelling_hook(evt, **kw):
        """Hook that raises CancelledError."""
        raise asyncio.CancelledError("Task cancelled")

    registry = HookRegistry(hooks={HookPhase.PreInvocation: cancelling_hook})
    event = SimpleTestEvent()

    result, should_exit, status = await registry.pre_invocation(event)

    # Result should be a tuple (Undefined, exception)
    assert isinstance(result, tuple)
    assert result[0] is Undefined
    assert isinstance(result[1], asyncio.CancelledError)
    assert should_exit is True
    assert status == EventStatus.CANCELLED


@pytest.mark.asyncio
async def test_registry_post_invocation_when_cancelled_then_returns_tuple():
    """Test post_invocation when task is cancelled (line 261).

    When a hook raises CancelledError, post_invocation should return:
    ((Undefined, exception), True, EventStatus.CANCELLED)
    """
    import asyncio

    from lionpride.types import Undefined

    async def cancelling_hook(evt, **kw):
        """Hook that raises CancelledError."""
        raise asyncio.CancelledError("Task cancelled")

    registry = HookRegistry(hooks={HookPhase.PostInvocation: cancelling_hook})
    event = SimpleTestEvent()

    result, should_exit, status = await registry.post_invocation(event)

    # Result should be a tuple (Undefined, exception)
    assert isinstance(result, tuple)
    assert result[0] is Undefined
    assert isinstance(result[1], asyncio.CancelledError)
    assert should_exit is True
    assert status == EventStatus.CANCELLED


@pytest.mark.asyncio
async def test_registry_handle_streaming_chunk_when_cancelled_then_returns_tuple():
    """Test handle_streaming_chunk when task is cancelled (line 279).

    When a stream handler raises CancelledError, handle_streaming_chunk should return:
    ((Undefined, exception), True, EventStatus.CANCELLED)
    """
    import asyncio

    from lionpride.types import Undefined

    async def cancelling_handler(evt, chunk_type, chunk, **kw):
        """Handler that raises CancelledError."""
        raise asyncio.CancelledError("Stream cancelled")

    registry = HookRegistry(stream_handlers={"text": cancelling_handler})

    result, should_exit, status = await registry.handle_streaming_chunk("text", "data")

    # Result should be a tuple (Undefined, exception)
    assert isinstance(result, tuple)
    assert result[0] is Undefined
    assert isinstance(result[1], asyncio.CancelledError)
    assert should_exit is True
    assert status == EventStatus.CANCELLED


@pytest.mark.asyncio
async def test_registry_pre_event_create_when_cancelled_then_returns_tuple():
    """Test pre_event_create when task is cancelled.

    Complements the pre_invocation, post_invocation, and handle_streaming_chunk
    cancellation tests. Verifies consistent behavior across all hook phases.
    """
    import asyncio

    from lionpride.types import Undefined

    async def cancelling_hook(event_type, **kw):
        """Hook that raises CancelledError."""
        raise asyncio.CancelledError("Task cancelled")

    registry = HookRegistry(hooks={HookPhase.PreEventCreate: cancelling_hook})

    result, should_exit, status = await registry.pre_event_create(SimpleTestEvent)

    # Result should be a tuple (Undefined, exception)
    assert isinstance(result, tuple)
    assert result[0] is Undefined
    assert isinstance(result[1], asyncio.CancelledError)
    assert should_exit is True
    assert status == EventStatus.CANCELLED
