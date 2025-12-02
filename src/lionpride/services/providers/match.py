# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings

from lionpride.services.types.endpoint import Endpoint

KNOWN_PROVIDERS = frozenset(
    {
        "anthropic",
        "openai",
        "groq",
        "openrouter",
        "nvidia_nim",
        "claude_code",
    }
)


def match_endpoint(
    provider: str,
    endpoint: str,
    **kwargs,
) -> Endpoint:
    """Match provider and endpoint to appropriate Endpoint class.

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "groq", "openrouter", "nvidia_nim", "claude_code")
        endpoint: Endpoint name (e.g., "messages", "chat/completions", "query_cli")
        **kwargs: Additional kwargs passed to Endpoint constructor

    Returns:
        Endpoint instance configured for the provider

    Example:
        >>> endpoint = match_endpoint("anthropic", "messages", api_key="...")
        >>> endpoint = match_endpoint("openai", "chat/completions")
        >>> endpoint = match_endpoint("claude_code", "query_cli")
    """
    if provider == "anthropic" and ("messages" in endpoint or "chat" in endpoint):
        from .anthropic_messages import AnthropicMessagesEndpoint

        return AnthropicMessagesEndpoint(None, **kwargs)

    if provider == "claude_code":
        from .claude_code import ClaudeCodeEndpoint

        return ClaudeCodeEndpoint(None, **kwargs)

    if provider == "gemini_code":
        from .gemini import GeminiCodeEndpoint

        return GeminiCodeEndpoint(None, **kwargs)

    if provider in ("openai", "groq", "openrouter", "nvidia_nim") and "chat" in endpoint:
        from .oai_chat import OAIChatEndpoint

        return OAIChatEndpoint(None, provider=provider, endpoint=endpoint, **kwargs)

    # OpenAI-compatible fallback with warning for unknown providers
    if provider not in KNOWN_PROVIDERS:
        warnings.warn(
            f"Unknown provider '{provider}', falling back to OpenAI-compatible endpoint. "
            f"Known providers: {sorted(KNOWN_PROVIDERS)}",
            UserWarning,
            stacklevel=2,
        )

    from .oai_chat import OAIChatEndpoint

    return OAIChatEndpoint(None, provider=provider, endpoint=endpoint, **kwargs)
