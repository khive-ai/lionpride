# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from lionpride.core import Element, Event, EventStatus
from lionpride.types import HashableModel, MaybeUnset, Unset, is_sentinel

from .hook import HookBroadcaster, HookEvent, HookPhase

logger = logging.getLogger(__name__)


class ServiceConfig(HashableModel):
    provider: str = Field(..., min_length=4, max_length=50)
    name: str = Field(..., min_length=4, max_length=100)
    request_options: type[BaseModel] | None = None
    timeout: int = Field(default=300, ge=1, le=3600)
    max_retries: int = Field(default=3, ge=0, le=10)
    version: str | None = None
    tags: list[str] = Field(default_factory=list)
    kwargs: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    def _validate_kwargs(cls, data: dict):  # noqa: N805
        kwargs = data.pop("kwargs", {})
        field_keys = list(cls.model_json_schema().get("properties", {}).keys())
        for k in list(data.keys()):
            if k not in field_keys:
                kwargs[k] = data.pop(k)
        data["kwargs"] = kwargs
        return data

    @field_serializer("request_options")
    def _serialize_request_options(self, v: type[BaseModel] | None):
        if v is None:
            return None
        return v.model_json_schema()

    @field_validator("request_options", mode="before")
    def _validate_request_options(cls, v):  # noqa: N805
        if v is None:
            return None
        try:
            if isinstance(v, type) and issubclass(v, BaseModel):
                return v
            if isinstance(v, BaseModel):
                return v.__class__
            if isinstance(v, dict | str):
                try:
                    from lionpride.libs import schema_handlers

                    return schema_handlers.load_pydantic_model_from_schema(v)
                except ImportError:
                    logger.warning(
                        "datamodel-code-generator not installed. "
                        "Install with: pip install 'lionpride[schema-gen]' "
                        "request_options will not be validated"
                    )
                    return None
        except Exception as e:
            raise ValueError("Invalid request options") from e
        raise ValueError("Invalid request options: must be a Pydantic model or a schema dict")

    def validate_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.request_options:
            return data
        try:
            self.request_options.model_validate(data)
            return data
        except Exception as e:
            raise ValueError("Invalid payload") from e


class NormalizedResponse(HashableModel):
    """Generic normalized response for all service backends.

    Works for any backend type: HTTP endpoints, tools, LLM APIs, etc.
    Provides consistent interface regardless of underlying service.
    """

    status: str = Field(..., description="Response status: 'success' or 'error'")
    data: Any = None
    error: str | None = Field(None, description="Error message if status='error'")
    raw_response: dict = Field(..., description="Original unmodified response")
    metadata: dict | None = Field(None, description="Provider-specific metadata")

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return self.model_dump(exclude_none=True)


class Calling(Event):
    """Base calling event with hook support.

    Extends lionpride.Event with pre/post invocation hooks.
    Always delegates to backend.call() for actual service invocation.

    Attributes:
        backend: ServiceBackend instance (Tool, Endpoint, etc.)
        payload: Request payload/arguments for backend call
    """

    backend: ServiceBackend = Field(..., exclude=True, description="Service backend instance")
    payload: dict[str, Any] = Field(..., description="Request payload/arguments")
    _pre_invoke_hook_event: HookEvent | None = PrivateAttr(None)
    _post_invoke_hook_event: HookEvent | None = PrivateAttr(None)

    @property
    def response(self) -> MaybeUnset[NormalizedResponse]:
        """Get normalized response from execution."""
        if is_sentinel(self.execution.response):
            return Unset
        return self.execution.response

    @property
    @abstractmethod
    def call_args(self) -> dict:
        """Get arguments for backend.call(**self.call_args).

        Subclasses must implement this to return their specific call arguments:
        - APICalling: {"request": ..., "extra_headers": ..., "skip_payload_creation": True}
        - ToolCalling: {"arguments": ...}

        Returns:
            Dict of keyword arguments for backend.call()
        """
        ...

    async def _invoke(self) -> NormalizedResponse:
        """Execute with hook lifecycle (called by parent Event.invoke()).

        Hook execution order:
        1. Pre-invocation hook (if configured) - can abort
        2. backend.call(**self.call_args) - actual service invocation
        3. Post-invocation hook (if configured) - runs even on failure (finally)

        Returns:
            NormalizedResponse from backend
        """
        # Pre-invocation hook
        if h_ev := self._pre_invoke_hook_event:
            await h_ev.invoke()

            # Check hook status and propagate failures
            if h_ev.execution.status in (EventStatus.FAILED, EventStatus.CANCELLED):
                raise RuntimeError(
                    f"Pre-invoke hook {h_ev.execution.status.value}: {h_ev.execution.error}"
                )

            if h_ev._should_exit:
                raise h_ev._exit_cause or RuntimeError(
                    "Pre-invocation hook requested exit without a cause"
                )
            await HookBroadcaster.broadcast(h_ev)

        # Actual service call via backend (post-hook runs in finally)
        try:
            response = await self.backend.call(**self.call_args)
            return response
        finally:
            # Post-invocation hook runs even on failure (for cleanup, metrics, logging)
            if h_ev := self._post_invoke_hook_event:
                await h_ev.invoke()

                # Check hook status (post-hook failures don't block, just log)
                if h_ev.execution.status in (EventStatus.FAILED, EventStatus.CANCELLED):
                    logger.warning(
                        f"Post-invoke hook {h_ev.execution.status.value}: {h_ev.execution.error}"
                    )

                if h_ev._should_exit:
                    raise h_ev._exit_cause or RuntimeError(
                        "Post-invocation hook requested exit without a cause"
                    )
                await HookBroadcaster.broadcast(h_ev)

    def create_pre_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool | None = None,
        hook_timeout: float = 30.0,
        hook_params: dict | None = None,
    ):
        """Create pre-invocation hook event."""
        h_ev = HookEvent(
            hook_phase=HookPhase.PreInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._pre_invoke_hook_event = h_ev

    def create_post_invoke_hook(
        self,
        hook_registry,
        exit_hook: bool | None = None,
        hook_timeout: float = 30.0,
        hook_params: dict | None = None,
    ):
        """Create post-invocation hook event."""
        h_ev = HookEvent(
            hook_phase=HookPhase.PostInvocation,
            event_like=self,
            registry=hook_registry,
            exit=exit_hook,
            timeout=hook_timeout,
            params=hook_params or {},
        )
        self._post_invoke_hook_event = h_ev


class ServiceBackend(Element):
    """Base class for all service backends (Tool, Endpoint, etc.).

    Inherits from lionpride.Element for UUID-based identity.
    Subclasses must implement event_type and call() methods.
    """

    config: ServiceConfig = Field(..., description="Service configuration")

    @property
    def provider(self) -> str:
        """Provider name from config."""
        return self.config.provider

    @property
    def name(self) -> str:
        """Service name from config."""
        return self.config.name

    @property
    def version(self) -> str | None:
        """Service version from config."""
        return self.config.version

    @property
    def tags(self) -> set[str]:
        """Service tags from config."""
        return set(self.config.tags) if self.config.tags else set()

    @property
    def request_options(self) -> type[BaseModel] | None:
        """Request options schema (Pydantic model type) from config."""
        return self.config.request_options

    @property
    @abstractmethod
    def event_type(self) -> type[Calling]:
        """Return Calling type for this backend (e.g., ToolCalling, APICalling)."""
        ...

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        """Normalize raw response into NormalizedResponse.

        Default implementation wraps response as-is. Subclasses can override
        to extract specific fields or add metadata.

        Args:
            raw_response: Raw response from service call

        Returns:
            NormalizedResponse with status, data, raw_response
        """
        return NormalizedResponse(
            status="success",
            data=raw_response,
            raw_response=raw_response,
        )

    @abstractmethod
    async def call(self, *args, **kw) -> NormalizedResponse:
        """Execute service call and return normalized response."""
        ...

    async def stream(self, *args, **kw):
        """Stream responses (not supported by default)."""
        raise NotImplementedError("This backend does not support streaming calls.")
