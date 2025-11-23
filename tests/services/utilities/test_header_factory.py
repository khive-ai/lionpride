# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for header_factory.py to achieve 100% coverage."""

import pytest
from pydantic import SecretStr

from lionpride.services.utilities.header_factory import AUTH_TYPES, HeaderFactory


class TestHeaderFactory:
    """Test suite for HeaderFactory class."""

    def test_get_content_type_header_default(self):
        """Test default content type header."""
        result = HeaderFactory.get_content_type_header()
        assert result == {"Content-Type": "application/json"}

    def test_get_content_type_header_custom(self):
        """Test custom content type header."""
        result = HeaderFactory.get_content_type_header("text/plain")
        assert result == {"Content-Type": "text/plain"}

    def test_get_bearer_auth_header(self):
        """Test bearer authentication header."""
        result = HeaderFactory.get_bearer_auth_header("test-api-key-123")
        assert result == {"Authorization": "Bearer test-api-key-123"}

    def test_get_x_api_key_header(self):
        """Test x-api-key header."""
        result = HeaderFactory.get_x_api_key_header("test-api-key-456")
        assert result == {"x-api-key": "test-api-key-456"}

    def test_get_header_none_auth_with_content_type(self):
        """Test get_header with no authentication and content type."""
        result = HeaderFactory.get_header(auth_type="none", content_type="application/json")
        assert result == {"Content-Type": "application/json"}

    def test_get_header_none_auth_without_content_type(self):
        """Test get_header with no authentication and no content type."""
        result = HeaderFactory.get_header(auth_type="none", content_type=None)
        assert result == {}

    def test_get_header_bearer_with_string_api_key(self):
        """Test get_header with bearer auth and string API key."""
        result = HeaderFactory.get_header(
            auth_type="bearer", api_key="my-secret-key", content_type="application/json"
        )
        assert result == {
            "Content-Type": "application/json",
            "Authorization": "Bearer my-secret-key",
        }

    def test_get_header_bearer_with_secret_str(self):
        """Test get_header with bearer auth and SecretStr API key."""
        secret_key = SecretStr("my-secret-key")
        result = HeaderFactory.get_header(
            auth_type="bearer", api_key=secret_key, content_type="application/json"
        )
        assert result == {
            "Content-Type": "application/json",
            "Authorization": "Bearer my-secret-key",
        }

    def test_get_header_x_api_key_with_string(self):
        """Test get_header with x-api-key auth and string API key."""
        result = HeaderFactory.get_header(
            auth_type="x-api-key", api_key="my-api-key-789", content_type="application/json"
        )
        assert result == {"Content-Type": "application/json", "x-api-key": "my-api-key-789"}

    def test_get_header_x_api_key_with_secret_str(self):
        """Test get_header with x-api-key auth and SecretStr API key."""
        secret_key = SecretStr("my-api-key-789")
        result = HeaderFactory.get_header(auth_type="x-api-key", api_key=secret_key)
        assert result == {"Content-Type": "application/json", "x-api-key": "my-api-key-789"}

    def test_get_header_bearer_missing_api_key(self):
        """Test get_header with bearer auth but missing API key."""
        with pytest.raises(ValueError, match="API key is required for authentication"):
            HeaderFactory.get_header(auth_type="bearer", api_key=None)

    def test_get_header_x_api_key_missing_api_key(self):
        """Test get_header with x-api-key auth but missing API key."""
        with pytest.raises(ValueError, match="API key is required for authentication"):
            HeaderFactory.get_header(auth_type="x-api-key", api_key=None)

    def test_get_header_bearer_empty_string_api_key(self):
        """Test get_header with bearer auth but empty string API key."""
        with pytest.raises(ValueError, match="API key is required for authentication"):
            HeaderFactory.get_header(auth_type="bearer", api_key="")

    def test_get_header_with_default_headers(self):
        """Test get_header with additional default headers."""
        result = HeaderFactory.get_header(
            auth_type="bearer",
            api_key="test-key",
            content_type="application/json",
            default_headers={"X-Custom-Header": "value", "X-Request-ID": "123"},
        )
        assert result == {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-key",
            "X-Custom-Header": "value",
            "X-Request-ID": "123",
        }

    def test_get_header_default_headers_override(self):
        """Test that default headers can override existing headers."""
        result = HeaderFactory.get_header(
            auth_type="bearer",
            api_key="test-key",
            content_type="application/json",
            default_headers={"Content-Type": "text/plain"},
        )
        # default_headers should override content_type
        assert result["Content-Type"] == "text/plain"
        assert result["Authorization"] == "Bearer test-key"

    def test_get_header_no_content_type_with_default_headers(self):
        """Test get_header with no content type but with default headers."""
        result = HeaderFactory.get_header(
            auth_type="none", content_type=None, default_headers={"X-Custom": "header"}
        )
        assert result == {"X-Custom": "header"}

    def test_get_header_unsupported_auth_type(self):
        """Test get_header with unsupported auth type (bypassing type checking)."""
        # This tests the runtime error for unsupported auth types
        # In practice, this is prevented by the Literal type, but can happen at runtime
        with pytest.raises(ValueError, match="Unsupported auth type"):
            HeaderFactory.get_header(
                auth_type="invalid",  # type: ignore
                api_key="test-key",
            )

    def test_get_header_whitespace_secret_str(self):
        """Test that whitespace-only SecretStr raises ValueError."""
        from pydantic import SecretStr

        with pytest.raises(ValueError, match="API key is required"):
            HeaderFactory.get_header(
                auth_type="bearer",
                api_key=SecretStr("   "),  # Whitespace only
            )

    def test_get_header_secret_str_with_whitespace_stripped(self):
        """Test that SecretStr with leading/trailing whitespace is stripped."""
        from pydantic import SecretStr

        result = HeaderFactory.get_header(
            auth_type="bearer",
            api_key=SecretStr("  test-key  "),  # Whitespace around key
        )
        assert result["Authorization"] == "Bearer test-key"  # Stripped
