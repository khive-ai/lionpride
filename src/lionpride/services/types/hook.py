# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, TypeVar

from pydantic import Field, PrivateAttr, field_validator
from typing_extensions import TypedDict

from lionpride.core import Broadcaster, Event, EventStatus
from lionpride.libs import concurrency
from lionpride.types import Enum, Undefined

SC = TypeVar("SC")
"""Stream Chunk type variable."""

StreamHandlers = dict[str, Callable[[SC], Awaitable[None]]]
"""Mapping of chunk type names to their respective asynchronous handler functions."""


E = TypeVar("E", bound=Event)


class HookPhase(Enum):
    PreEventCreate = "pre_event_create"
    PreInvocation = "pre_invocation"
    PostInvocation = "post_invocation"
    ErrorHandling = "error_handling"


class AssociatedEventInfo(TypedDict, total=False):
    """Information about the event associated with the hook."""

    lion_class: str
    """Full qualified name of the event class."""

    event_id: str
    """ID of the event."""

    event_created_at: float
    """Creation timestamp of the event."""


class HookEvent(Event):
    """Hook execution event that delegates to HookRegistry.

    Extends lionpride.Event with hook-specific execution logic.
    Parent Event.invoke() handles lifecycle, this implements _invoke().
    """

    registry: HookRegistry = Field(..., exclude=True)
    hook_phase: HookPhase
    exit: bool = Field(False, exclude=True)
    params: dict[str, Any] = Field(default_factory=dict, exclude=True)
    event_like: Event | type[Event] = Field(..., exclude=True)
    _should_exit: bool = PrivateAttr(False)
    _exit_cause: BaseException | None = PrivateAttr(None)

    associated_event_info: AssociatedEventInfo | None = None

    @field_validator("exit", mode="before")
    def _validate_exit(cls, v: Any) -> bool:  # noqa: N805
        if v is None:
            return False
        return v

    async def _invoke(self) -> Any:
        """Execute hook via registry (called by parent Event.invoke()).

        Parent Event.invoke() handles status/timing/errors automatically.
        Just execute hook logic and let exceptions propagate naturally.
        """
        (res, se, _st), meta = await self.registry.call(
            self.event_like,
            hook_phase=self.hook_phase,
            exit=self.exit,
            **self.params,
        )

        self.associated_event_info = AssociatedEventInfo(**meta)
        self._should_exit = se

        # Handle error results - raise them so parent Event catches and sets FAILED status
        if isinstance(res, tuple) and len(res) == 2:
            # Tuple (Undefined, exception) from cancelled hook
            self._exit_cause = res[1]
            raise res[1]

        if isinstance(res, Exception):
            # Exception result from failed hook
            self._exit_cause = res
            raise res

        # Success - return result (parent sets COMPLETED status)
        return res


def get_handler(d_: dict, k: str | type, get: bool = False, /):
    handler = d_.get(k)
    if handler is None and not get:
        return None

    if handler is not None:
        if not concurrency.is_coro_func(handler):

            async def _func(*args, **kwargs):
                await concurrency.sleep(0)
                return handler(*args, **kwargs)

            return _func
        return handler

    async def _func(*args, **_kwargs):
        await concurrency.sleep(0)
        return args[0] if args else None

    return _func


def validate_hooks(kw):
    """Validate that all hooks are callable."""
    if not isinstance(kw, dict):
        raise ValueError("Hooks must be a dictionary of callable functions")

    for k, v in kw.items():
        if not isinstance(k, HookPhase) or k not in HookPhase.allowed():
            raise ValueError(f"Hook key must be one of {HookPhase.allowed()}, got {k}")
        if not callable(v):
            raise ValueError(f"Hook for {k} must be callable, got {type(v)}")


def validate_stream_handlers(kw):
    """Validate that all stream handlers are callable."""
    if not isinstance(kw, dict):
        raise ValueError("Stream handlers must be a dictionary of callable functions")

    for k, v in kw.items():
        if not isinstance(k, str | type):
            raise ValueError(f"Stream handler key must be a string or type, got {type(k)}")

        if not callable(v):
            raise ValueError(f"Stream handler for {k} must be callable, got {type(v)}")


class HookRegistry:
    """Registry for hook callbacks at event lifecycle phases.

    Hook phases: PreEventCreate, PreInvocation, PostInvocation, ErrorHandling.
    Handlers can return values, raise exceptions to cancel/abort, or pass through.
    Stream handlers process chunks during streaming execution.
    """

    def __init__(
        self,
        hooks: dict[HookPhase, Callable] | None = None,
        stream_handlers: StreamHandlers | None = None,
    ):
        _hooks = {}
        _stream_handlers = {}

        if hooks is not None:
            validate_hooks(hooks)
            _hooks.update(hooks)

        if stream_handlers is not None:
            validate_stream_handlers(stream_handlers)
            _stream_handlers.update(stream_handlers)

        self._hooks = _hooks
        self._stream_handlers = _stream_handlers

    async def _call(
        self,
        hp_: HookPhase,
        ct_: str | type,
        ch_: Any,
        ev_: E | type[E],
        /,
        **kw,
    ) -> tuple[Any | Exception, bool]:
        if hp_ is None and ct_ is None:
            raise RuntimeError("Either hook_type or chunk_type must be provided")
        if hp_ and (self._hooks.get(hp_)):
            validate_hooks({hp_: self._hooks[hp_]})
            h = get_handler(self._hooks, hp_, True)
            return await h(ev_, **kw)
        elif not ct_:
            raise RuntimeError("Hook type is required when chunk_type is not provided")
        else:
            validate_stream_handlers({ct_: self._stream_handlers.get(ct_)})
            h = get_handler(self._stream_handlers, ct_, True)
            return await h(ev_, ct_, ch_, **kw)

    async def _call_stream_handler(
        self,
        ct_: str | type,
        ch_: Any,
        ev_,
        /,
        **kw,
    ):
        validate_stream_handlers({ct_: self._stream_handlers.get(ct_)})
        handler = get_handler(self._stream_handlers, ct_, True)
        return await handler(ev_, ct_, ch_, **kw)

    async def pre_event_create(
        self, event_type: type[E], /, exit: bool = False, **kw
    ) -> tuple[E | Exception | None, bool, EventStatus]:
        """Call hook before event creation. Returns event, None, or raises to cancel."""
        try:
            res = await self._call(
                HookPhase.PreEventCreate,
                None,
                None,
                event_type,
                exit=exit,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.CANCELLED)

    async def pre_invocation(
        self, event: E, /, exit: bool = False, **kw
    ) -> tuple[Any, bool, EventStatus]:
        """Call hook before event invocation. Raise to cancel."""
        try:
            res = await self._call(
                HookPhase.PreInvocation,
                None,
                None,
                event,
                exit=exit,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.CANCELLED)

    async def post_invocation(
        self, event: E, /, exit: bool = False, **kw
    ) -> tuple[None | Exception, bool, EventStatus]:
        """Call hook after event execution. Raise to abort."""
        try:
            res = await self._call(
                HookPhase.PostInvocation,
                None,
                None,
                event,
                exit=exit,
                **kw,
            )
            return (res, False, EventStatus.COMPLETED)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.ABORTED)

    async def handle_streaming_chunk(
        self, chunk_type: str | type, chunk: Any, /, exit: bool = False, **kw
    ) -> tuple[Any, bool, EventStatus | None]:
        """Process streaming chunk. Raise to abort stream."""
        try:
            res = await self._call_stream_handler(
                chunk_type,
                chunk,
                None,
                exit=exit,
                **kw,
            )
            return (res, False, None)
        except concurrency.get_cancelled_exc_class() as e:
            return ((Undefined, e), True, EventStatus.CANCELLED)
        except Exception as e:
            return (e, exit, EventStatus.ABORTED)

    async def call(
        self,
        event_like: Event | type[Event],
        /,
        *,
        hook_phase: HookPhase | None = None,
        chunk_type=None,
        chunk=None,
        exit=False,
        **kw,
    ):
        """Call a hook or stream handler.

        If method is provided, it will call the corresponding hook.
        If chunk_type is provided, it will call the corresponding stream handler.
        If both are provided, method will be used.
        """
        if hook_phase is None and chunk_type is None:
            raise ValueError("Either method or chunk_type must be provided")

        if hook_phase:
            meta = {"lion_class": event_like.class_name(full=True)}
            match hook_phase:
                case HookPhase.PreEventCreate | HookPhase.PreEventCreate.value:
                    return (
                        await self.pre_event_create(event_like, exit=exit, **kw),
                        meta,
                    )
                case HookPhase.PreInvocation | HookPhase.PreInvocation.value:
                    meta["event_id"] = str(event_like.id)
                    meta["event_created_at"] = event_like.created_at
                    return (
                        await self.pre_invocation(event_like, exit=exit, **kw),
                        meta,
                    )
                case HookPhase.PostInvocation | HookPhase.PostInvocation.value:
                    meta["event_id"] = str(event_like.id)
                    meta["event_created_at"] = event_like.created_at
                    return (
                        await self.post_invocation(event_like, exit=exit, **kw),
                        meta,
                    )
        return await self.handle_streaming_chunk(chunk_type, chunk, exit=exit, **kw)

    def _can_handle(
        self,
        /,
        *,
        hp_: HookPhase | None = None,
        ct_=None,
    ) -> bool:
        """Check if the registry can handle the given event or chunk type."""
        if hp_:
            return hp_ in self._hooks
        if ct_:
            return ct_ in self._stream_handlers
        return False


class HookBroadcaster(Broadcaster):
    _event_type: ClassVar[type[HookEvent]] = HookEvent
