# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import re
from typing import Any, TypeVar

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)

from ..utilities.header_factory import AUTH_TYPES, HeaderFactory
from ..utilities.resilience import CircuitBreaker, RetryConfig, retry_with_backoff
from .backend import Calling, ServiceBackend, ServiceConfig

logger = logging.getLogger(__name__)

# System environment variables that cannot be used as API keys
# V1: Prevent collision with common system env vars
SYSTEM_ENV_VARS = frozenset(
    {
        "HOME",
        "PATH",
        "USER",
        "SHELL",
        "PWD",
        "LANG",
        "TERM",
        "TMPDIR",
        "LOGNAME",
        "HOSTNAME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "PS1",
        "OLDPWD",
        "EDITOR",
        "PAGER",
        "DISPLAY",
        "SSH_AUTH_SOCK",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)


B = TypeVar("B", bound=type[BaseModel])


class EndpointConfig(ServiceConfig):
    """Configuration for Endpoint backend."""

    base_url: str | None = None
    endpoint: str
    endpoint_params: list[str] | None = None
    method: str = "POST"
    params: dict[str, str] = Field(default_factory=dict)
    content_type: str | None = "application/json"
    auth_type: AUTH_TYPES = "bearer"
    default_headers: dict = Field(default_factory=dict)
    api_key: str | None = Field(
        None, description="API key environment variable name, not the key itself.", frozen=True
    )
    api_key_is_env: bool = Field(
        False,
        description="True if api_key is an env var name, False if was raw credential (cleared).",
        frozen=True,
    )
    openai_compatible: bool = False
    requires_tokens: bool = False
    client_kwargs: dict = Field(default_factory=dict)
    _api_key: SecretStr | None = PrivateAttr(None)  # Resolved credential (never serialized)

    @property
    def api_key_env(self) -> str | None:
        """Alias for api_key for explicit env var semantics."""
        return self.api_key

    @model_validator(mode="after")
    def _validate_api_key_n_params(self):
        """Resolve API key from api_key field.

        Security model:
        - F6: Validate credentials are non-empty/non-whitespace
        - V1: Prevent system env var collision with heuristic pattern matching
        - V2: Track whether api_key is env var name (api_key_is_env)
        - If str input + os.getenv() succeeds → env var name, KEEP api_key for serialization
        - If str input + os.getenv() fails → raw credential, CLEAR api_key (prevent leak)
        - Resolved credential always stored in _api_key (SecretStr, never serialized)
        """
        if self.api_key is not None:
            # V2: Check if this is deserialization with api_key_is_env=True
            # If so, verify the env var still exists
            if self.api_key_is_env:
                # This is deserialization - verify env var still exists
                if not os.getenv(self.api_key):
                    raise ValueError(
                        f"Environment variable '{self.api_key}' not found during deserialization. "
                        f"Model was serialized with env var reference that no longer exists."
                    )
                # Env var exists, resolve it
                resolved = os.getenv(self.api_key, None)
                if resolved and resolved.strip():
                    object.__setattr__(self, "_api_key", SecretStr(resolved.strip()))
                # api_key_is_env already set, keep api_key as is
                return self

            # F6: Validate input is non-empty and non-whitespace
            if not self.api_key.strip():
                raise ValueError("api_key cannot be empty or whitespace")

            # V1: Check if matches env var pattern (UPPERCASE with underscores)
            is_env_var_pattern = bool(re.match(r"^[A-Z][A-Z0-9_]*$", self.api_key))

            if is_env_var_pattern:
                # V1: Block system env vars to prevent collision
                if self.api_key in SYSTEM_ENV_VARS:
                    raise ValueError(
                        f"'{self.api_key}' is a system environment variable and cannot be used as api_key. "
                        f"If this is a raw credential, pass it as SecretStr('{self.api_key}')."
                    )

                # Try to resolve as env var
                resolved = os.getenv(self.api_key, None)
                if resolved is not None:
                    # Successfully resolved as env var (exists in environment)
                    # F6: Validate resolved value is non-empty/non-whitespace
                    if not resolved.strip():
                        raise ValueError(
                            f"Environment variable '{self.api_key}' is empty or whitespace"
                        )
                    # V2: Mark as env var and store stripped credential
                    object.__setattr__(self, "api_key_is_env", True)
                    object.__setattr__(self, "_api_key", SecretStr(resolved.strip()))
                    # KEEP api_key (env var name) for serialization
                else:
                    # Pattern matches but env var doesn't exist → treat as raw credential
                    # V2: Mark as NOT env var
                    object.__setattr__(self, "api_key_is_env", False)
                    object.__setattr__(self, "_api_key", SecretStr(self.api_key.strip()))
                    object.__setattr__(self, "api_key", None)  # CLEAR to prevent leak
            else:
                # Doesn't match env var pattern → must be raw credential
                # V2: Mark as NOT env var
                object.__setattr__(self, "api_key_is_env", False)
                object.__setattr__(self, "_api_key", SecretStr(self.api_key.strip()))
                object.__setattr__(self, "api_key", None)  # CLEAR to prevent leak

        if self.endpoint_params and self.params:
            # Check that all params are in endpoint_params
            invalid_params = set(self.params.keys()) - set(self.endpoint_params)
            if invalid_params:
                raise ValueError(
                    f"Invalid params {invalid_params}. Must be subset of endpoint_params: {self.endpoint_params}"
                )

            # Warn if endpoint_params expects params but none provided
            missing_params = set(self.endpoint_params) - set(self.params.keys())
            if missing_params:
                logger.warning(
                    f"Endpoint expects params {missing_params} but they were not provided. "
                    f"URL formatting may fail."
                )
        return self

    @property
    def full_url(self):
        """Construct full URL from base_url and endpoint."""
        if not self.endpoint_params:
            return f"{self.base_url}/{self.endpoint}"
        return f"{self.base_url}/{self.endpoint.format(**self.params)}"


class Endpoint(ServiceBackend):
    circuit_breaker: CircuitBreaker | None = None
    retry_config: RetryConfig | None = None
    config: EndpointConfig

    def __init__(
        self,
        config: dict | EndpointConfig,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
        **kwargs,
    ):
        # Extract SecretStr api_key if present (raw credential)
        secret_api_key = None
        if isinstance(config, dict):
            config_dict = {**config, **kwargs}
            # Intercept SecretStr api_key
            if "api_key" in config_dict and isinstance(config_dict["api_key"], SecretStr):
                secret_api_key = config_dict.pop("api_key")
            _config = EndpointConfig(**config_dict)
        elif isinstance(config, EndpointConfig):
            _config = (
                config.model_copy(deep=True, update=kwargs)
                if kwargs
                else config.model_copy(deep=True)
            )
        else:
            raise ValueError("Config must be a dict or EndpointConfig instance")

        # Initialize ServiceBackend with config and resilience components
        super().__init__(
            config=_config,
            circuit_breaker=circuit_breaker,
            retry_config=retry_config,
        )

        # Inject SecretStr directly to _api_key if provided
        if secret_api_key is not None:
            # F6: Validate and strip whitespace from SecretStr credentials
            raw_value = secret_api_key.get_secret_value()
            if not raw_value.strip():
                raise ValueError("api_key cannot be empty or whitespace")
            # Strip whitespace and store
            object.__setattr__(self.config, "_api_key", SecretStr(raw_value.strip()))

        logger.debug(
            f"Initialized Endpoint with provider={self.config.provider}, "
            f"endpoint={self.config.endpoint}, circuit_breaker={circuit_breaker is not None}, "
            f"retry_config={retry_config is not None}"
        )

    def _create_http_client(self):
        """Create a new HTTP client for requests."""
        import httpx

        return httpx.AsyncClient(
            timeout=self.config.timeout,
            **self.config.client_kwargs,
        )

    @property
    def event_type(self) -> type:
        """Return Event/Calling type for this backend."""
        return APICalling

    @property
    def full_url(self) -> str:
        """Return full URL from config."""
        return self.config.full_url

    def create_payload(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        # First, create headers
        headers = HeaderFactory.get_header(
            auth_type=self.config.auth_type,
            content_type=self.config.content_type,
            api_key=self.config._api_key,
            default_headers=self.config.default_headers,
        )
        if extra_headers:
            headers.update(extra_headers)

        # Convert request to dict if it's a BaseModel
        request = request if isinstance(request, dict) else request.model_dump(exclude_none=True)

        # Start with config defaults
        payload = self.config.kwargs.copy()

        # Update with request data
        payload.update(request)

        # Update with additional kwargs
        if kwargs:
            payload.update(kwargs)

        # Validate payload using request_options schema (required)
        if self.config.request_options is None:
            raise ValueError(
                f"Endpoint {self.config.name} must define request_options schema. "
                "All endpoint backends must use proper request validation."
            )

        # Get valid field names from the model
        valid_fields = set(self.config.request_options.model_fields.keys())

        # Filter payload to only include valid fields
        filtered_payload = {k: v for k, v in payload.items() if k in valid_fields}

        # Validate the filtered payload
        payload = self.config.validate_payload(filtered_payload)

        return (payload, headers)

    async def _call(self, payload: dict, headers: dict, **kwargs):
        return await self._call_http(payload=payload, headers=headers, **kwargs)

    async def call(
        self,
        request: dict | BaseModel,
        skip_payload_creation: bool = False,
        **kwargs,
    ):
        """
        Make a call to the endpoint.

        Args:
            request: The request parameters or model.
            skip_payload_creation: Whether to skip create_payload and treat request as ready payload.
            **kwargs: Additional keyword arguments for the request.

        Returns:
            NormalizedResponse from the endpoint.
        """
        # Extract extra_headers before passing to create_payload
        extra_headers = kwargs.pop("extra_headers", None)

        payload, headers = None, None
        if skip_payload_creation:
            # Treat request as ready payload - bypasses create_payload() validation and auth
            # Caller is responsible for providing valid payload and headers (if auth needed)
            payload = request if isinstance(request, dict) else request.model_dump()
            headers = extra_headers or {}
        else:
            payload, headers = self.create_payload(request, extra_headers=extra_headers, **kwargs)

        # Apply resilience patterns: CB wraps _call (inner), retry wraps CB (outer)
        # This ensures each retry attempt counts against circuit breaker metrics
        base_call = self._call

        # Step 1: Wrap _call with circuit breaker (if configured)
        if self.circuit_breaker:

            async def cb_wrapped_call(p, h, **kw):
                return await self.circuit_breaker.execute(base_call, p, h, **kw)

            inner_call = cb_wrapped_call
        else:
            inner_call = base_call

        # Step 2: Wrap (possibly CB-wrapped) call with retry (if configured)
        if self.retry_config:
            raw_response = await retry_with_backoff(
                inner_call, payload, headers, **kwargs, **self.retry_config.as_kwargs()
            )
        else:
            raw_response = await inner_call(payload, headers, **kwargs)

        # Wrap response in NormalizedResponse
        return self.normalize_response(raw_response)

    async def _call_http(self, payload: dict, headers: dict, **kwargs):
        import httpx

        # Create a new client for this request
        async with self._create_http_client() as client:
            response = await client.request(
                method=self.config.method,
                url=self.config.full_url,
                headers=headers,
                json=payload,
                **kwargs,
            )

            # Check for rate limit or server errors that should be retried
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            elif response.status_code != 200:
                # Try to get error details from response body
                try:
                    error_body = response.json()
                    error_message = (
                        f"Request failed with status {response.status_code}: {error_body}"
                    )
                except Exception:
                    error_message = f"Request failed with status {response.status_code}"

                raise httpx.HTTPStatusError(
                    message=error_message,
                    request=response.request,
                    response=response,
                )

            # Extract and return the JSON response
            return response.json()

    async def stream(
        self,
        request: dict | BaseModel,
        extra_headers: dict | None = None,
        **kwargs,
    ):
        """
        Stream responses from the endpoint.

        Args:
            request: The request parameters or model.
            extra_headers: Additional headers for the request.
            **kwargs: Additional keyword arguments for the request.

        Yields:
            Streaming chunks from the API.
        """
        payload, headers = self.create_payload(request, extra_headers, **kwargs)

        # Direct streaming without context manager
        async for chunk in self._stream_http(payload=payload, headers=headers, **kwargs):
            yield chunk

    async def _stream_http(self, payload: dict, headers: dict, **kwargs):
        """
        Stream responses using httpx with a fresh client.

        Args:
            payload: The request payload.
            headers: The request headers.
            **kwargs: Additional keyword arguments for the request.

        Yields:
            Streaming chunks from the API.
        """
        import httpx

        # Ensure stream is enabled
        payload["stream"] = True

        # Create a new client for streaming
        async with (
            self._create_http_client() as client,
            client.stream(
                method=self.config.method,
                url=self.config.full_url,
                headers=headers,
                json=payload,
                **kwargs,
            ) as response,
        ):
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    message=f"Request failed with status {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if line:
                    yield line

    @field_serializer("circuit_breaker")
    def _serialize_circuit_breaker(
        self, circuit_breaker: CircuitBreaker | None
    ) -> dict[str, Any] | None:
        """Serialize circuit breaker to config dict."""
        if circuit_breaker is None:
            return None
        return circuit_breaker.to_dict()

    @field_serializer("retry_config")
    def _serialize_retry_config(self, retry_config: RetryConfig | None) -> dict[str, Any] | None:
        """Serialize retry config to dict."""
        if retry_config is None:
            return None
        return retry_config.to_dict()

    @field_validator("circuit_breaker", mode="before")
    @classmethod
    def _deserialize_circuit_breaker(cls, v: Any) -> CircuitBreaker | None:
        """Reconstruct CircuitBreaker from config dict."""
        if v is None:
            return None

        if isinstance(v, CircuitBreaker):
            return v

        if not isinstance(v, dict):
            raise ValueError("circuit_breaker must be a dict or CircuitBreaker instance")

        return CircuitBreaker(**v)

    @field_validator("retry_config", mode="before")
    @classmethod
    def _deserialize_retry_config(cls, v: Any) -> RetryConfig | None:
        """Reconstruct RetryConfig from dict."""
        if v is None:
            return None

        if isinstance(v, RetryConfig):
            return v

        if not isinstance(v, dict):
            raise ValueError("retry_config must be a dict or RetryConfig instance")

        return RetryConfig(**v)


class APICalling(Calling):
    """API call via Endpoint.

    Extends Calling (Event-based) to execute API requests through Endpoint.
    Stores request payload and headers separately for clean separation.

    Usage:
        endpoint = Endpoint(config=config)
        calling = APICalling(
            backend=endpoint,
            payload={"model": "gpt-4", "messages": [...]},
            headers={"X-Custom": "value"},
            timeout=30.0
        )
        await calling.invoke()
        response = calling.execution.response  # NormalizedResponse
    """

    backend: Endpoint = Field(exclude=True)
    headers: dict = Field(exclude=True)

    @property
    def required_tokens(self) -> int | None:
        """Estimate tokens for rate limiting. None skips token tracking."""
        # Check if backend requires token tracking
        if (
            hasattr(self.backend.config, "requires_tokens")
            and not self.backend.config.requires_tokens
        ):
            return None

        # Provider-specific token calculation
        # For OpenAI/Anthropic chat completions
        if "messages" in self.payload:
            return self._estimate_message_tokens(self.payload["messages"])

        # For embeddings
        if "input" in self.payload:
            return self._estimate_text_tokens(self.payload["input"])

        # Default: None (no token tracking)
        return None

    def _estimate_message_tokens(self, messages: list[dict]) -> int:
        """Estimate token usage for chat messages.

        Simple approximation: ~4 chars per token.
        For production, use tiktoken or provider-specific libraries.
        """
        total_chars = sum(
            len(str(msg.get("content", ""))) + len(msg.get("role", "")) for msg in messages
        )
        return int(total_chars / 4) + 10  # +10 for message overhead

    def _estimate_text_tokens(self, text: str | list[str]) -> int:
        """Estimate token usage for text input."""
        if isinstance(text, str):
            return int(len(text) / 4)
        return sum(int(len(t) / 4) for t in text)

    @property
    def request(self) -> dict:
        """Get permission data for Processor.request_permission(**event.request).

        Used for rate limiting checks before processing.

        Returns:
            Dict with required_tokens for rate limit checking
        """
        return {
            "required_tokens": self.required_tokens,
        }

    @property
    def call_args(self) -> dict:
        """Get arguments for backend.call(**self.call_args).

        Returns:
            Dict with request payload, headers, and skip_payload_creation flag
        """
        return {
            "request": self.payload,
            "extra_headers": self.headers,
            "skip_payload_creation": True,
        }
