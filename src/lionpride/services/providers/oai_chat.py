# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from lionpride.services.types import NormalizedResponse

from ..types.endpoint import Endpoint, EndpointConfig

__all__ = (
    "OAIChatEndpoint",
    "create_oai_chat",
)

_PROVIDER_MAP: MappingProxyType | None = None


def _get_providers() -> MappingProxyType:
    """Get mapping of provider names to base URLs.

    Returns:
        Immutable mapping of provider name to base_url for OpenAI-compatible providers:
        - openai: OpenAI official API
        - groq: Groq fast inference
        - openrouter: Multi-provider gateway
        - nvidia_nim: NVIDIA Inference Microservices
    """
    global _PROVIDER_MAP
    if _PROVIDER_MAP is not None:
        return _PROVIDER_MAP

    _PROVIDER_MAP = MappingProxyType(
        {
            "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
            "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
            "nvidia_nim": ("https://integrate.api.nvidia.com/v1", "NVIDIA_NIM_API_KEY"),
        }
    )
    return _PROVIDER_MAP


def _ensure_request_options(config: dict) -> dict:
    if config.get("request_options") is not None:
        return config

    from ..third_party.openai_models import OpenAIChatCompletionsRequest

    return {**config, "request_options": OpenAIChatCompletionsRequest}


def create_oai_chat(
    provider: str = "openai",
    name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    endpoint: str = "chat/completions",
) -> dict:
    """Factory for OpenAI-compatible Chat API config.

    Args:
        provider: Provider name (openai, groq, openrouter, nvidia_nim)
        name: Config name (default: "{provider}_chat")
        base_url: Base API URL (optional, auto-detected from provider)
        api_key: API key or env var name (default: "{PROVIDER}_API_KEY")
        endpoint: Endpoint path
    """

    # Get provider base URL
    base_url = base_url or _get_providers().get(provider, (None,))[0]
    if base_url is None:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(_get_providers().keys())}"
        )

    # Set default name
    if name is None:
        name = f"{provider}_chat"

    # Set default API key env var
    if api_key is None:
        api_key = _get_providers().get(provider, (None, None))[1]

    return {
        "name": name,
        "provider": provider,
        "base_url": base_url,
        "endpoint": endpoint,
        "api_key": api_key,
    }


class OAIChatEndpoint(Endpoint):
    """OpenAI Chat Completions API endpoint.

    Supports OpenAI and OpenAI-compatible providers.

    Usage:
        endpoint = OpenAIChatEndpoint()
        response = await endpoint.call({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}]
        })
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with OpenAI config."""
        if config is None:
            # Extract provider-specific kwargs for create_oai_chat
            config = create_oai_chat(
                provider=kwargs.pop("provider", "openai"),
                name=kwargs.pop("name", None),
                base_url=kwargs.pop("base_url", None),
                api_key=kwargs.pop("api_key", None),
                endpoint=kwargs.pop("endpoint", "chat/completions"),
            )
        elif isinstance(config, EndpointConfig):
            config = config.model_dump()
        if not isinstance(config, dict):
            raise ValueError("Provided config must be a dict or EndpointConfig instance")

        if kwargs.get("request_options") is None:
            config = _ensure_request_options(config)

        super().__init__(config=config, circuit_breaker=circuit_breaker, **kwargs)

    def normalize_response(self, raw_response: dict[str, Any]) -> NormalizedResponse:
        """Normalize OpenAI response to standard format.

        Extracts:
        - Text from choices[0].message.content
        - Usage stats
        - Model info
        - Finish reason
        - Tool calls (if present)
        """
        # Extract text from first choice
        text = ""
        choices = raw_response.get("choices")
        if choices and len(choices) > 0:
            choice = choices[0]
            message = choice.get("message", {})
            text = message.get("content") or ""

        # Extract metadata
        metadata: dict[str, Any] = {
            k: raw_response[k] for k in ("model", "usage", "id") if k in raw_response
        }

        if choices and len(choices) > 0:
            choice = choices[0]
            metadata.update({k: choice[k] for k in ("finish_reason",) if k in choice})

            # Extract tool calls if present
            message = choice.get("message", {})
            metadata.update({k: message[k] for k in ("tool_calls",) if k in message})

        return NormalizedResponse(
            status="success",
            data=text,
            raw_response=raw_response,
            metadata=metadata,
        )
