# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for OAI Chat provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lionpride.services.providers.oai_chat import (
    OAIChatEndpoint,
    create_oai_chat,
)
from lionpride.services.types import NormalizedResponse


class TestCreateOAIChat:
    """Test create_oai_chat factory function."""

    def test_create_oai_chat_defaults(self):
        """Test factory with default provider (openai)."""
        config = create_oai_chat()

        assert config["provider"] == "openai"
        assert config["base_url"] == "https://api.openai.com/v1"
        assert config["endpoint"] == "chat/completions"
        assert config["api_key"] == "OPENAI_API_KEY"

    def test_create_oai_chat_groq_provider(self):
        """Test factory with groq provider."""
        config = create_oai_chat(provider="groq")

        assert config["provider"] == "groq"
        assert config["base_url"] == "https://api.groq.com/openai/v1"
        assert config["api_key"] == "GROQ_API_KEY"

    def test_create_oai_chat_openrouter_provider(self):
        """Test factory with openrouter provider."""
        config = create_oai_chat(provider="openrouter")

        assert config["provider"] == "openrouter"
        assert config["base_url"] == "https://openrouter.ai/api/v1"
        assert config["api_key"] == "OPENROUTER_API_KEY"

    def test_create_oai_chat_nvidia_nim_provider(self):
        """Test factory with nvidia_nim provider."""
        config = create_oai_chat(provider="nvidia_nim")

        assert config["provider"] == "nvidia_nim"
        assert config["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert config["api_key"] == "NVIDIA_NIM_API_KEY"

    def test_create_oai_chat_custom_values(self):
        """Test factory with custom values."""
        config = create_oai_chat(
            provider="openai",
            api_key="custom_key",
            endpoint="custom/endpoint",
        )

        assert config["provider"] == "openai"
        assert config["endpoint"] == "custom/endpoint"
        assert config["api_key"] == "custom_key"

    def test_create_oai_chat_custom_base_url(self):
        """Test factory with custom base_url."""
        config = create_oai_chat(base_url="https://custom.api.com")
        assert config["base_url"] == "https://custom.api.com"

    def test_create_oai_chat_invalid_provider(self):
        """Test factory raises error for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_oai_chat(provider="invalid_provider")


class TestOAIChatEndpointInit:
    """Test OAIChatEndpoint initialization."""

    def test_init_with_none_config(self):
        """Test initialization with config=None uses factory defaults."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")

        assert endpoint.config.provider == "openai"
        assert endpoint.config.base_url == "https://api.openai.com/v1"
        assert endpoint.config.endpoint == "chat/completions"
        assert endpoint.config.request_options is not None

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = {
            "name": "test-openai",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "chat/completions",
            "api_key": "test_key",
        }
        endpoint = OAIChatEndpoint(config=config_dict)

        assert endpoint.config.provider == "openai"
        assert endpoint.config.api_key is None  # Cleared (was raw credential)
        assert endpoint.config._api_key.get_secret_value() == "test_key"

    def test_init_with_circuit_breaker(self):
        """Test initialization with circuit breaker."""
        from lionpride.services.utilities.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = OAIChatEndpoint(config=None, name="test-default", circuit_breaker=cb)

        assert endpoint.circuit_breaker is not None
        assert endpoint.circuit_breaker.failure_threshold == 3


class TestOAIChatEndpointNormalizeResponse:
    """Test OAIChatEndpoint.normalize_response method."""

    def test_normalize_response_text_only(self):
        """Test normalize_response with text content only."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = endpoint.normalize_response(response)

        assert isinstance(normalized, NormalizedResponse)
        assert normalized.data == "Hello, world!"
        assert normalized.metadata["model"] == "gpt-4o-mini"
        assert normalized.metadata["usage"]["prompt_tokens"] == 10
        assert normalized.metadata["finish_reason"] == "stop"
        assert normalized.metadata["id"] == "chatcmpl-123"
        assert normalized.raw_response == response

    def test_normalize_response_empty_content(self):
        """Test normalize_response with empty content."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {"choices": [{"index": 0, "message": {"role": "assistant", "content": None}}]}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""

    def test_normalize_response_no_choices(self):
        """Test normalize_response with no choices."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {"id": "chatcmpl-123", "choices": []}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {"id": "chatcmpl-123"}

    def test_normalize_response_missing_choices(self):
        """Test normalize_response with missing choices field."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {"id": "chatcmpl-123"}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {"id": "chatcmpl-123"}

    def test_normalize_response_with_tool_calls(self):
        """Test normalize_response with tool_calls."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me call a tool.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": '{"a":1,"b":2}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "Let me call a tool."
        assert "tool_calls" in normalized.metadata
        assert len(normalized.metadata["tool_calls"]) == 1
        assert normalized.metadata["tool_calls"][0]["id"] == "call_123"
        assert normalized.metadata["finish_reason"] == "tool_calls"

    def test_normalize_response_multiple_tool_calls(self):
        """Test normalize_response with multiple tool_calls."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "tool1"}},
                            {"id": "call_2", "type": "function", "function": {"name": "tool2"}},
                        ],
                    }
                }
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert len(normalized.metadata["tool_calls"]) == 2
        assert normalized.metadata["tool_calls"][0]["id"] == "call_1"
        assert normalized.metadata["tool_calls"][1]["id"] == "call_2"

    def test_normalize_response_all_metadata_fields(self):
        """Test normalize_response extracts all metadata fields."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.metadata["id"] == "chatcmpl-123"
        assert normalized.metadata["model"] == "gpt-4o"
        assert normalized.metadata["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
        assert normalized.metadata["finish_reason"] == "stop"

    def test_normalize_response_extracts_first_choice_only(self):
        """Test normalize_response uses first choice only."""
        endpoint = OAIChatEndpoint(config=None, name="test-openai")
        response = {
            "choices": [
                {"message": {"content": "First choice"}, "finish_reason": "stop"},
                {"message": {"content": "Second choice"}, "finish_reason": "length"},
            ]
        }

        normalized = endpoint.normalize_response(response)

        # Should only extract from first choice
        assert normalized.data == "First choice"
        assert normalized.metadata["finish_reason"] == "stop"


class TestOAIChatEndpointIntegration:
    """Integration tests with mocked httpx."""

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful API call with mocked httpx."""
        endpoint = OAIChatEndpoint(config=None, name="test-default", api_key="test_key")

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "Hello from GPT!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

        assert result.data == "Hello from GPT!"
        assert result.metadata["model"] == "gpt-4o-mini"
        assert result.metadata["usage"]["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_call_with_tool_calls(self):
        """Test API call with tool_calls response."""
        endpoint = OAIChatEndpoint(config=None, name="test-default", api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "What's the weather?"}],
                }
            )

        assert "tool_calls" in result.metadata
        assert result.metadata["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result.metadata["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_call_http_error(self):
        """Test API call with HTTP error."""
        endpoint = OAIChatEndpoint(config=None, name="test-default", api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError, match="401"),
        ):
            await endpoint.call(
                request={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )


class TestSecurityProviderMapImmutability:
    """Security tests for provider map immutability (BS3)."""

    def test_provider_map_immutable(self):
        """Verify _PROVIDER_MAP cannot be mutated (security - CVSS 7.2).

        This test prevents attackers from redirecting API calls by mutating
        the global _PROVIDER_MAP to point to malicious endpoints.
        """
        import lionpride.services.providers.oai_chat as oai

        # Get provider map (triggers initialization if needed)
        providers = oai._get_providers()

        # Attempt mutation should raise TypeError
        with pytest.raises(TypeError, match="does not support item assignment"):
            providers["openai"] = ("https://evil.com", "EVIL_KEY")

        # Verify provider map is still valid after failed mutation attempt
        assert providers["openai"] == ("https://api.openai.com/v1", "OPENAI_API_KEY")

    def test_provider_map_global_mutation_blocked(self):
        """Verify direct mutation of module-level _PROVIDER_MAP is blocked."""
        import lionpride.services.providers.oai_chat as oai

        # Trigger initialization
        _ = oai._get_providers()

        # Direct mutation of global should also fail
        with pytest.raises(TypeError, match="does not support item assignment"):
            oai._PROVIDER_MAP["groq"] = ("https://attacker.com", "BAD_KEY")

    def test_provider_map_no_mutating_methods(self):
        """Verify MappingProxyType doesn't expose mutating methods."""
        import lionpride.services.providers.oai_chat as oai

        providers = oai._get_providers()

        # MappingProxyType should not have mutating methods like update
        assert not hasattr(providers, "update")
        assert not hasattr(providers, "pop")
        assert not hasattr(providers, "popitem")
        assert not hasattr(providers, "clear")

    def test_provider_map_returns_consistent_instance(self):
        """Verify _get_providers() returns same immutable instance."""
        import lionpride.services.providers.oai_chat as oai

        # Get providers twice
        providers1 = oai._get_providers()
        providers2 = oai._get_providers()

        # Should be the exact same object (cached)
        assert providers1 is providers2

        # Both should be immutable
        from types import MappingProxyType

        assert isinstance(providers1, MappingProxyType)
        assert isinstance(providers2, MappingProxyType)
