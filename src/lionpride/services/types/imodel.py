# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, field_serializer, field_validator

from lionpride.core import Element, Executor
from lionpride.libs.concurrency import sleep
from lionpride.protocols import Invocable, implements
from lionpride.services.types.hook import HookRegistry

from ..utilities.rate_limiter import TokenBucket
from .backend import ServiceBackend
from .endpoint import APICalling, Endpoint
from .tool import Tool, ToolCalling

if TYPE_CHECKING:
    from .backend import Calling


__all__ = ("iModel",)


@implements(Invocable)
class iModel(Element):  # noqa: N801
    """Unified service interface wrapping ServiceBackend with rate limiting and hooks.

    Initialization requires EITHER:
    - `backend`: A ServiceBackend instance (Endpoint, Tool, or Action)
    - `provider`: A provider string for auto-matching via `match_endpoint()`

    The `backend` field is typed as Optional to support the provider auto-match path,
    but at runtime exactly one of backend/provider must be provided.

    Note:
        Tool-backed iModel instances cannot be serialized (contain callables).
        Serialization returns backend=None, and deserialization will fail.
        Use Endpoint backends for persistence scenarios.
    """

    # Allow extra fields to pass through to match_endpoint via __init__ kwargs
    model_config = {"extra": "allow"}

    # Executor polling configuration (lionagi v0 pattern)
    _EXECUTOR_POLL_TIMEOUT_ITERATIONS = 100
    _EXECUTOR_POLL_SLEEP_INTERVAL = 0.1  # seconds

    backend: ServiceBackend | None = Field(
        None,
        description="ServiceBackend instance (Tool, Endpoint, Action)",
    )

    rate_limiter: TokenBucket | None = Field(
        None,
        description="Optional TokenBucket rate limiter (simple blocking)",
    )

    executor: Executor | None = Field(
        None,
        description="Optional Executor for event-driven processing with rate limiting",
    )

    hook_registry: HookRegistry | None = Field(
        None,
        description="Optional HookRegistry for invocation lifecycle hooks",
    )

    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata (e.g., Claude Code session_id for context continuation)",
    )

    def __init__(
        self,
        backend: ServiceBackend | None = None,
        provider: str | None = None,
        endpoint: str = "chat/completions",
        rate_limiter: TokenBucket | None = None,
        executor: Executor | None = None,
        hook_registry: HookRegistry | None = None,
        # Executor auto-construction params (lionagi v0 pattern):
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        limit_requests: int | None = None,
        limit_tokens: int | None = None,
        **kwargs: Any,
    ):
        """Initialize with ServiceBackend or auto-match from provider.

        Provide backend directly OR provider string for auto-matching.
        If limit_requests/limit_tokens set without executor, auto-constructs
        RateLimitedExecutor.

        Example:
            >>> model = iModel(backend=AnthropicMessagesEndpoint())
            >>> model = iModel(provider="anthropic", limit_requests=50)
        """
        # Auto-match endpoint if backend not provided (lionagi v0 pattern)
        if backend is None:
            if provider is None:
                raise ValueError("Either backend or provider must be provided")
            from ..providers import match_endpoint

            backend = match_endpoint(provider, endpoint, **kwargs)

        # Auto-construct executor from rate limiting params (lionagi v0 pattern)
        if executor is None and (limit_requests or limit_tokens):
            from ..execution import RateLimitedExecutor
            from ..utilities.rate_limiter import RateLimitConfig

            request_bucket = None
            if limit_requests:
                request_bucket = TokenBucket(
                    RateLimitConfig(
                        capacity=limit_requests,
                        refill_rate=limit_requests / capacity_refresh_time,
                    )
                )

            token_bucket = None
            if limit_tokens:
                token_bucket = TokenBucket(
                    RateLimitConfig(
                        capacity=limit_tokens,
                        refill_rate=limit_tokens / capacity_refresh_time,
                    )
                )

            executor = RateLimitedExecutor(
                processor_config={
                    "queue_capacity": queue_capacity,
                    "capacity_refresh_time": capacity_refresh_time,
                    "request_bucket": request_bucket,
                    "token_bucket": token_bucket,
                }
            )

        super().__init__(
            backend=backend,
            rate_limiter=rate_limiter,
            executor=executor,
            hook_registry=hook_registry,
            **kwargs,
        )

    @property
    def name(self) -> str:
        """Service name from backend."""
        return self.backend.name

    @property
    def version(self) -> str:
        """Service version from backend."""
        return self.backend.version

    @property
    def tags(self) -> set[str]:
        """Service tags from backend."""
        return self.backend.tags

    async def create_calling(
        self,
        timeout: float | None = None,
        streaming: bool = False,
        create_event_exit_hook: bool | None = None,
        create_event_hook_timeout: float = 10.0,
        create_event_hook_params: dict | None = None,
        pre_invoke_exit_hook: bool | None = None,
        pre_invoke_hook_timeout: float = 30.0,
        pre_invoke_hook_params: dict | None = None,
        post_invoke_exit_hook: bool | None = None,
        post_invoke_hook_timeout: float = 30.0,
        post_invoke_hook_params: dict | None = None,
        **arguments: Any,
    ) -> Calling:
        """Create Calling instance via backend.

        For API backends (Endpoint): calls create_payload to get (payload, headers)
        For Tool backends: passes request arguments directly
        Attaches hook_registry to Calling if configured.

        Args:
            timeout: Event timeout in seconds (enforced in Event.invoke via fail_after)
            streaming: Whether this is a streaming request (Event.streaming attr)
            create_event_exit_hook: Whether pre-event-create hook should trigger exit on failure (None = use default)
            create_event_hook_timeout: Timeout for pre-event-create hook execution in seconds
            create_event_hook_params: Optional parameters to pass to pre-event-create hook
            pre_invoke_exit_hook: Whether pre-invoke hook should trigger exit on failure (None = use default)
            pre_invoke_hook_timeout: Timeout for pre-invoke hook execution in seconds
            pre_invoke_hook_params: Optional parameters to pass to pre-invoke hook
            post_invoke_exit_hook: Whether post-invoke hook should trigger exit on failure (None = use default)
            post_invoke_hook_timeout: Timeout for post-invoke hook execution in seconds
            post_invoke_hook_params: Optional parameters to pass to post-invoke hook
            **arguments: Request arguments to pass to backend
        """
        from .hook import HookEvent, HookPhase

        # Claude Code: auto-inject session_id for resume if available (lionagi v0 pattern)
        if (
            isinstance(self.backend, Endpoint)
            and self.backend.config.provider == "claude_code"
            and "resume" not in arguments
            and self.provider_metadata.get("session_id")
        ):
            arguments["resume"] = self.provider_metadata["session_id"]

        # Get Calling type from backend
        calling_type = self.backend.event_type

        # Pre-event create hook
        if self.hook_registry is not None and self.hook_registry._can_handle(
            hp_=HookPhase.PreEventCreate
        ):
            h_ev = HookEvent(
                hook_phase=HookPhase.PreEventCreate,
                event_like=calling_type,
                registry=self.hook_registry,
                exit=create_event_exit_hook if create_event_exit_hook is not None else False,
                timeout=create_event_hook_timeout,
                params=create_event_hook_params or {},
            )
            await h_ev.invoke()

            if h_ev._should_exit:
                raise h_ev._exit_cause or RuntimeError(
                    "PreEventCreate hook requested exit without a cause"
                )

        # Type-based dispatch for Calling creation
        if isinstance(self.backend, Endpoint):
            payload, headers = self.backend.create_payload(request=arguments)
            calling = APICalling(
                backend=self.backend,
                payload=payload,
                headers=headers,
            )
        elif isinstance(self.backend, Tool):
            calling = ToolCalling(
                backend=self.backend,
                payload=arguments,  # Direct arguments dict, not wrapped
            )
        else:
            raise RuntimeError(f"Unsupported backend type: {type(self.backend)}")

        # Attach pre-invoke hook if registry can handle it
        if self.hook_registry is not None and self.hook_registry._can_handle(
            hp_=HookPhase.PreInvocation
        ):
            calling.create_pre_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=pre_invoke_exit_hook if pre_invoke_exit_hook is not None else False,
                hook_timeout=pre_invoke_hook_timeout,
                hook_params=pre_invoke_hook_params or {},
            )

        # Attach post-invoke hook if registry can handle it
        if self.hook_registry is not None and self.hook_registry._can_handle(
            hp_=HookPhase.PostInvocation
        ):
            calling.create_post_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=post_invoke_exit_hook if post_invoke_exit_hook is not None else False,
                hook_timeout=post_invoke_hook_timeout,
                hook_params=post_invoke_hook_params or {},
            )

        # Set Event attributes
        if timeout is not None:
            calling.timeout = timeout
        if streaming:
            calling.streaming = streaming

        return calling

    async def invoke(
        self,
        calling: Calling | None = None,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **arguments: Any,
    ) -> Calling:
        """Invoke calling with optional event-driven processing.

        Routes invocation based on executor presence:
        - If executor configured: event-driven processing with rate limiting (lionagi v0 pattern)
        - Otherwise: direct invocation with optional simple rate limiting

        Hooks are handled by Calling itself during invocation.

        Args:
            calling: Pre-created Calling instance. If provided, **arguments are IGNORED
                and the calling is invoked directly. Use this when you need to configure
                the Calling beforehand (e.g., set timeout on the Event).
            poll_timeout: Max seconds to wait for executor completion (default: 10s).
                For long-running LLM calls, increase this (e.g., 120s for large models).
            poll_interval: Seconds between status checks (default: 0.1s).
            **arguments: Request arguments passed to create_calling. IGNORED if calling provided.

        Returns:
            Calling instance with execution results populated

        Raises:
            TimeoutError: If rate limit acquisition or polling times out
            RuntimeError: If event aborted after 3 permission denials (executor path)

        Example:
            # Standard usage - create and invoke in one call
            calling = await imodel.invoke(model="gpt-4", messages=[...])

            # Pre-created calling with custom timeout
            calling = await imodel.create_calling(model="gpt-4", messages=[...])
            calling.timeout = 120.0  # 2 minute timeout
            calling = await imodel.invoke(calling=calling)
        """
        if calling is None:
            calling = await self.create_calling(**arguments)

        # Route based on executor presence
        if self.executor:
            # Event-driven path with executor (lionagi v0 pattern)
            # Start processor if not already running
            if self.executor.processor is None or self.executor.processor.is_stopped():
                await self.executor.start()

            # Enqueue event for processing
            await self.executor.append(calling)
            await self.executor.forward()

            # Poll for completion with active forwarding (lionagi v0 pattern)
            # Performance note (BLIND-4): Polling adds latency overhead.
            # Fast backends (<100ms): 100-200% overhead
            # Slow backends (>1s): <10% overhead (acceptable)
            interval = poll_interval or self._EXECUTOR_POLL_SLEEP_INTERVAL
            timeout_seconds = poll_timeout or (
                self._EXECUTOR_POLL_TIMEOUT_ITERATIONS * self._EXECUTOR_POLL_SLEEP_INTERVAL
            )
            max_iterations = int(timeout_seconds / interval)
            ctr = 0

            while calling.execution.status.value in ["pending", "processing"]:
                if ctr > max_iterations:
                    raise TimeoutError(
                        f"Event processing timeout after {timeout_seconds:.1f}s: {calling.id}"
                    )
                await self.executor.forward()
                ctr += 1
                await sleep(interval)

            # Check final status and raise if aborted or failed
            if calling.execution.status.value == "aborted":
                raise RuntimeError(
                    f"Event aborted after 3 permission denials (rate limited): {calling.id}"
                )
            elif calling.execution.status.value == "failed":
                raise calling.execution.error or RuntimeError(f"Event failed: {calling.id}")

            # Claude Code: store session_id for context continuation (lionagi v0 pattern)
            self._store_claude_code_session_id(calling)

            return calling

        else:
            # Direct invocation (current behavior)
            if self.rate_limiter:
                acquired = await self.rate_limiter.acquire(timeout=30.0)
                if not acquired:
                    raise TimeoutError("Rate limit acquisition timeout (30s)")

            await calling.invoke()

            # Claude Code: store session_id for context continuation (lionagi v0 pattern)
            self._store_claude_code_session_id(calling)

            return calling

    def _store_claude_code_session_id(self, calling: Calling) -> None:
        """Store Claude Code session_id from response for context continuation.

        When using the same iModel instance for multiple Claude Code calls,
        subsequent calls automatically resume the previous session context.
        """
        if (
            isinstance(self.backend, Endpoint)
            and self.backend.config.provider == "claude_code"
            and calling.execution.response is not None
        ):
            # session_id is in response metadata
            session_id = calling.execution.response.metadata.get("session_id")
            if session_id:
                self.provider_metadata["session_id"] = session_id

    @field_serializer("backend")
    def _serialize_backend(self, backend: ServiceBackend) -> dict[str, Any] | None:
        """Serialize backend to dict (Endpoint only, Tool has callables).

        Warning:
            Tool-backed iModel instances serialize backend as None.
            Deserialization will fail with "backend is required" error.
            Use Endpoint backends for persistence scenarios.
        """
        if isinstance(backend, Endpoint):
            # Use model_dump() to get Pydantic structure (not Element.to_dict())
            # Excluded fields (circuit_breaker, retry_config) are automatically omitted
            backend_dict = backend.model_dump()
            # Add lion_class for polymorphic deserialization
            if "metadata" not in backend_dict:
                backend_dict["metadata"] = {}
            backend_dict["metadata"]["lion_class"] = backend.__class__.class_name(full=True)
            return backend_dict
        else:
            # Tool has callables, cannot serialize - deserialization will fail
            return None

    @field_serializer("rate_limiter")
    def _serialize_rate_limiter(self, v: TokenBucket | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return v.to_dict()

    @field_serializer("executor")
    def _serialize_executor(self, executor: Executor | None) -> dict[str, Any] | None:
        """Serialize executor as config dict (ephemeral state lost).

        TokenBucket instances in processor_config are serialized to their config dicts,
        ensuring fresh capacity on deserialization (state is not preserved).
        """
        if executor is None:
            return None

        # Only serialize RateLimitedExecutor config for now, because the config is well-defined
        from ..execution import RateLimitedExecutor

        if isinstance(executor, RateLimitedExecutor):
            config = {**executor.processor_config}
            # Serialize TokenBucket instances to config dicts (fresh capacity on deserialize)
            if "request_bucket" in config and config["request_bucket"] is not None:
                bucket = config["request_bucket"]
                if isinstance(bucket, TokenBucket):
                    config["request_bucket"] = bucket.to_dict()
            if "token_bucket" in config and config["token_bucket"] is not None:
                bucket = config["token_bucket"]
                if isinstance(bucket, TokenBucket):
                    config["token_bucket"] = bucket.to_dict()
            return config

        return None

    @field_validator("rate_limiter", mode="before")
    @classmethod
    def _deserialize_rate_limiter(cls, v: Any) -> TokenBucket | None:
        """Reconstruct TokenBucket from RateLimitConfig dict."""
        if v is None:
            return None

        from ..utilities.rate_limiter import RateLimitConfig, TokenBucket

        if isinstance(v, TokenBucket):
            return v

        if not isinstance(v, dict):
            raise ValueError("rate_limiter must be a dict or TokenBucket instance")

        config = RateLimitConfig(**v)
        return TokenBucket(config)

    @field_validator("backend", mode="before")
    @classmethod
    def _deserialize_backend(cls, v: Any) -> ServiceBackend:
        """Reconstruct backend from dict via Element polymorphic deserialization.

        Security: Element.from_dict() only instantiates registered Element subclasses.
        Post-construction isinstance check ensures only ServiceBackend types are accepted.
        """
        if v is None:
            raise ValueError("backend is required")

        if isinstance(v, ServiceBackend):
            return v

        if not isinstance(v, dict):
            raise ValueError("backend must be a dict or ServiceBackend instance")

        from lionpride import Element

        backend = Element.from_dict(v)  # Polymorphic type restoration

        if not isinstance(backend, ServiceBackend):
            raise ValueError(
                f"Deserialized backend must be ServiceBackend subclass, got: {type(backend).__name__}"
            )
        return backend

    @field_validator("executor", mode="before")
    @classmethod
    def _deserialize_executor(cls, v: Any) -> Executor | None:
        """Reconstruct executor from config dict.

        TokenBucket configs in the dict are converted back to TokenBucket instances,
        ensuring fresh capacity (serialization does not preserve depleted state).
        """
        if v is None:
            return None

        if isinstance(v, Executor):
            return v  # Already an Executor instance

        if not isinstance(v, dict):
            raise ValueError("executor must be a dict or Executor instance")

        from ..execution import RateLimitedExecutor
        from ..utilities.rate_limiter import RateLimitConfig

        # Reconstruct TokenBuckets from config dicts (fresh capacity)
        config = {**v}
        if "request_bucket" in config and isinstance(config["request_bucket"], dict):
            config["request_bucket"] = TokenBucket(RateLimitConfig(**config["request_bucket"]))
        if "token_bucket" in config and isinstance(config["token_bucket"], dict):
            config["token_bucket"] = TokenBucket(RateLimitConfig(**config["token_bucket"]))

        return RateLimitedExecutor(processor_config=config)

    def _to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict, excluding id/created_at for fresh identity on reconstruction.

        Field serializers handle backend/executor/rate_limiter/hook_registry serialization.

        Note:
            Tool backends serialize as None and will fail on deserialization.
            Use Endpoint backends for persistence scenarios.
        """
        # Exclude iModel identity (id/created_at) - fresh identity on reconstruction
        kwargs.setdefault("exclude", set()).update({"id", "created_at"})
        return super()._to_dict(**kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return f"iModel(backend={self.backend.name}, version={self.backend.version})"
