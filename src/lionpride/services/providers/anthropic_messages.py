# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from lionpride.services.types import NormalizedResponse

from ..types.endpoint import Endpoint, EndpointConfig

__all__ = (
    "AnthropicMessagesEndpoint",
    "create_anthropic_config",
)


def create_anthropic_config(
    name: str = "anthropic_messages",
    api_key: str | None = None,
    base_url: str = "https://api.anthropic.com/v1",
    endpoint: str = "messages",
    anthropic_version: str = "2023-06-01",
) -> dict:
    """Factory for Anthropic Messages API config.

    Args:
        api_key: API key or env var name (default: "ANTHROPIC_API_KEY")
        base_url: Base API URL
        endpoint: Endpoint path
        anthropic_version: API version header

    Returns:
        Config dict

    Example:
        >>> config = create_anthropic_config()
        >>> endpoint = AnthropicMessagesEndpoint(config=config)
        >>> response = await endpoint.call(
        ...     {"model": "claude-sonnet-4-5-20250929", "messages": [...]}
        ... )
    """
    if api_key is None:
        api_key = "ANTHROPIC_API_KEY"

    return {
        "name": name,
        "provider": "anthropic",
        "base_url": base_url,
        "endpoint": endpoint,
        "api_key": api_key,
        "auth_type": "x-api-key",
        "default_headers": {"anthropic-version": anthropic_version},
    }


class AnthropicMessagesEndpoint(Endpoint):
    """Anthropic Messages API endpoint.

    Supports Anthropic-specific response normalization.

    Usage:
        endpoint = AnthropicMessagesEndpoint()
        response = await endpoint.call({
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024
        })
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with Anthropic config."""
        if config is None:
            config = create_anthropic_config(**kwargs)
        elif isinstance(config, EndpointConfig):
            config = config.model_dump()
        if not isinstance(config, dict):
            raise ValueError("Provided config must be a dict or EndpointConfig instance")

        # Ensure request_options is set
        if config.get("request_options") is None:
            from ..third_party.anthropic_models import CreateMessageRequest

            config["request_options"] = CreateMessageRequest

        super().__init__(config=config, circuit_breaker=circuit_breaker, **kwargs)

    def normalize_response(self, response: dict[str, Any]) -> NormalizedResponse:
        """Normalize Anthropic response to standard format.

        Extracts:
        - Text from content blocks
        - Thinking blocks (if extended thinking enabled)
        - Usage stats
        - Stop reason
        - Model info
        - Tool uses
        """
        # Extract text and thinking from content blocks
        text_parts = []
        thinking_parts = []

        content = response.get("content")
        if content:
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    thinking_parts.append(block.get("thinking", ""))

        # Combine text
        text = "\n\n".join(text_parts)

        # Extract metadata
        metadata: dict[str, Any] = {
            k: response[k]
            for k in ("model", "usage", "stop_reason", "stop_sequence", "id")
            if k in response
        }

        # Add thinking if present
        if thinking_parts:
            metadata["thinking"] = "\n\n".join(thinking_parts)

        # Add tool use blocks if present
        tool_uses = [
            block for block in response.get("content", []) if block.get("type") == "tool_use"
        ]
        if tool_uses:
            metadata["tool_uses"] = tool_uses

        return NormalizedResponse(
            status="success",
            data=text,
            raw_response=response,
            metadata=metadata,
        )
