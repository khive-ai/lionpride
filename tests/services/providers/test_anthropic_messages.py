# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Anthropic Messages provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lionpride.services.providers.anthropic_messages import (
    AnthropicMessagesEndpoint,
    create_anthropic_config,
)
from lionpride.services.types import NormalizedResponse


class TestCreateAnthropicConfig:
    """Test create_anthropic_config factory function."""

    def test_create_anthropic_config_defaults(self):
        """Test factory with default values."""
        config = create_anthropic_config()

        assert config["provider"] == "anthropic"
        assert config["base_url"] == "https://api.anthropic.com/v1"
        assert config["endpoint"] == "messages"
        assert config["api_key"] == "ANTHROPIC_API_KEY"
        assert config["auth_type"] == "x-api-key"
        assert config["default_headers"]["anthropic-version"] == "2023-06-01"

    def test_create_anthropic_config_custom_values(self):
        """Test factory with custom values."""
        config = create_anthropic_config(
            api_key="custom_key",
            base_url="https://custom.api.com",
            endpoint="custom/endpoint",
            anthropic_version="2024-01-01",
        )

        assert config["provider"] == "anthropic"
        assert config["base_url"] == "https://custom.api.com"
        assert config["endpoint"] == "custom/endpoint"
        assert config["api_key"] == "custom_key"
        assert config["default_headers"]["anthropic-version"] == "2024-01-01"

    def test_create_anthropic_config_api_key_none(self):
        """Test factory with api_key=None uses env var name."""
        config = create_anthropic_config(api_key=None)
        assert config["api_key"] == "ANTHROPIC_API_KEY"


class TestAnthropicMessagesEndpointInit:
    """Test AnthropicMessagesEndpoint initialization."""

    def test_init_with_none_config(self):
        """Test initialization with config=None uses factory defaults."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config.base_url == "https://api.anthropic.com/v1"
        assert endpoint.config.endpoint == "messages"
        assert endpoint.config.request_options is not None

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "endpoint": "messages",
            "api_key": "test_key",
            "name": "test",
            "auth_type": "x-api-key",
        }
        endpoint = AnthropicMessagesEndpoint(config=config_dict)

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config.api_key is None  # Cleared (was raw credential)
        assert endpoint.config._api_key.get_secret_value() == "test_key"

    def test_init_with_circuit_breaker(self):
        """Test initialization with circuit breaker."""
        from lionpride.services.utilities.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = AnthropicMessagesEndpoint(config=None, name="test", circuit_breaker=cb)

        assert endpoint.circuit_breaker is not None
        assert endpoint.circuit_breaker.failure_threshold == 3


class TestAnthropicMessagesEndpointNormalizeResponse:
    """Test AnthropicMessagesEndpoint.normalize_response method."""

    def test_normalize_response_text_only(self):
        """Test normalize_response with text content only."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {
            "id": "msg_123",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        normalized = endpoint.normalize_response(response)

        assert isinstance(normalized, NormalizedResponse)
        assert normalized.data == "Hello, world!"
        assert normalized.metadata["model"] == "claude-sonnet-4-5-20250929"
        assert normalized.metadata["usage"]["input_tokens"] == 10
        assert normalized.metadata["stop_reason"] == "end_turn"
        assert normalized.metadata["id"] == "msg_123"
        assert normalized.raw_response == response

    def test_normalize_response_multiple_text_blocks(self):
        """Test normalize_response with multiple text blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"},
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "First part\n\nSecond part"

    def test_normalize_response_with_thinking(self):
        """Test normalize_response with thinking blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Here's my answer"},
            ]
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "Here's my answer"
        assert "thinking" in normalized.metadata
        assert normalized.metadata["thinking"] == "Let me think..."

    def test_normalize_response_with_tool_use(self):
        """Test normalize_response with tool_use blocks."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {
            "content": [
                {"type": "text", "text": "Let me use a tool"},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "calculator",
                    "input": {"a": 1, "b": 2},
                },
            ],
            "stop_reason": "tool_use",
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.data == "Let me use a tool"
        assert "tool_uses" in normalized.metadata
        assert len(normalized.metadata["tool_uses"]) == 1
        assert normalized.metadata["tool_uses"][0]["name"] == "calculator"
        assert normalized.metadata["stop_reason"] == "tool_use"

    def test_normalize_response_empty_content(self):
        """Test normalize_response with empty content."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {"id": "msg_123", "content": []}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {"id": "msg_123"}

    def test_normalize_response_no_content(self):
        """Test normalize_response with missing content field."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {"id": "msg_123"}

        normalized = endpoint.normalize_response(response)

        assert normalized.data == ""
        assert normalized.metadata == {"id": "msg_123"}


class TestAnthropicMessagesEndpointIntegration:
    """Integration tests with mocked httpx."""

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful API call with mocked httpx."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test", api_key="test_key")

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint.call(
                request={
                    "model": "claude-sonnet-4-5-20250929",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                }
            )

        assert result.data == "Hello from Claude!"
        assert result.metadata["model"] == "claude-sonnet-4-5-20250929"
        assert result.metadata["usage"]["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_call_http_error(self):
        """Test API call with HTTP error."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test", api_key="test_key")

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
                    "model": "claude-sonnet-4-5-20250929",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                }
            )
