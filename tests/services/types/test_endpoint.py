# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive coverage tests for Endpoint module."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from lionpride.errors import ConnectionError
from lionpride.services.types.endpoint import APICalling, Endpoint, EndpointConfig
from lionpride.services.utilities.resilience import CircuitBreaker, RetryConfig


# Test Request Models
class SimpleRequest(BaseModel):
    """Simple request model for testing."""

    message: str
    temperature: float = 0.7


class TestEndpointConfig:
    """Test EndpointConfig validation and properties."""

    def test_validate_kwargs_moves_extra_fields(self):
        """Test that extra fields are moved to kwargs dict."""
        data = {
            "name": "test",
            "provider": "test_provider",
            "endpoint": "/test",
            "extra_field": "extra_value",
            "another_field": 123,
        }
        config = EndpointConfig(**data)
        assert config.kwargs["extra_field"] == "extra_value"
        assert config.kwargs["another_field"] == 123

    def test_validate_api_key_from_secret_str(self):
        """Test API key validation from SecretStr (via Endpoint)."""
        secret = SecretStr("secret_key_123")
        # SecretStr must go through Endpoint.__init__, not EndpointConfig directly
        endpoint = Endpoint(
            config={
                "name": "test",
                "provider": "test",
                "endpoint": "/test",
                "api_key": secret,
            }
        )
        assert endpoint.config.api_key is None  # Cleared (was raw credential via SecretStr)
        assert endpoint.config._api_key.get_secret_value() == "secret_key_123"

    def test_validate_api_key_from_string_literal(self):
        """Test API key validation from string (not env var)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            api_key="literal_key_456",
        )
        # String that's not an env var → treated as raw credential, cleared
        assert config.api_key is None
        assert config._api_key.get_secret_value() == "literal_key_456"

    def test_validate_api_key_from_env_var(self):
        """Test API key validation from environment variable."""
        os.environ["TEST_API_KEY"] = "env_key_789"
        try:
            config = EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                api_key="TEST_API_KEY",
            )
            # Successfully resolved from env → keep env var name
            assert config.api_key == "TEST_API_KEY"
            assert config._api_key.get_secret_value() == "env_key_789"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_validate_provider_empty_raises(self):
        """Test that empty provider raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="String should have at least 4 characters"):
            EndpointConfig(
                name="test",
                provider="",
                endpoint="/test",
            )

    def test_full_url_with_endpoint_params(self):
        """Test full_url property with endpoint params."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="api/{version}/test",
            base_url="https://api.test.com",
            endpoint_params=["version"],
            params={"version": "v1"},
            request_options=SimpleRequest,
        )
        # Test on both config and endpoint
        assert config.full_url == "https://api.test.com/api/v1/test"

        endpoint = Endpoint(config=config)
        assert endpoint.full_url == "https://api.test.com/api/v1/test"

    def test_validate_request_options_none(self):
        """Test request_options validator with None."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,
        )
        assert config.request_options is None

    def test_validate_request_options_pydantic_class(self):
        """Test request_options validator with Pydantic class."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        assert config.request_options == SimpleRequest

    def test_validate_request_options_pydantic_instance(self):
        """Test request_options validator with Pydantic instance."""
        instance = SimpleRequest(message="test")
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=instance,
        )
        assert config.request_options == SimpleRequest

    def test_validate_request_options_dict_with_schema_gen(self):
        """Test request_options with dict when schema-gen IS installed."""
        # Check if datamodel-code-generator is actually installed by trying to use it
        try:
            from lionpride import schema_handlers

            test_schema = {"type": "object", "properties": {"test": {"type": "string"}}}
            schema_handlers.load_pydantic_model_from_schema(test_schema)
        except ImportError:
            pytest.skip("datamodel-code-generator not installed (required for schema-gen)")
            return

        # schema-gen is available and working, test should succeed
        schema = {
            "type": "object",
            "properties": {"message": {"type": "string"}, "temperature": {"type": "number"}},
            "required": ["message"],
        }
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=schema,
        )
        # Should successfully create a Pydantic model from schema
        assert config.request_options is not None
        assert issubclass(config.request_options, BaseModel)

    def test_validate_request_options_dict_no_schema_gen(self):
        """Test request_options with dict when schema-gen NOT installed.

        This tests the graceful degradation when datamodel-code-generator
        is not installed. We mock the load_pydantic_model_from_schema to
        raise ImportError to simulate the missing dependency.
        """
        from lionpride.libs import schema_handlers

        original_func = schema_handlers.load_pydantic_model_from_schema

        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'datamodel_code_generator'")

        # Temporarily replace the function
        schema_handlers.load_pydantic_model_from_schema = raise_import_error

        try:
            with patch("lionpride.services.types.backend.logger") as mock_logger:
                config = EndpointConfig(
                    name="test",
                    provider="test",
                    endpoint="/test",
                    request_options={"type": "object", "properties": {}},
                )
                # Should log warning and return None (graceful degradation)
                assert config.request_options is None
                mock_logger.warning.assert_called_once()
                assert "datamodel-code-generator not installed" in str(
                    mock_logger.warning.call_args
                )
        finally:
            # Restore original function
            schema_handlers.load_pydantic_model_from_schema = original_func

    def test_full_url_without_endpoint_params(self):
        """Test full_url property without endpoint params."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
        )
        # Test when endpoint_params is None (line 80)
        assert config.endpoint_params is None
        assert config.full_url == "https://api.test.com/test"

    def test_validate_request_options_generic_exception(self):
        """Test request_options with exception during validation."""

        # Create a mock that raises a non-ImportError exception
        class BadModel:
            """A class that will cause issues during validation."""

            pass

        with pytest.raises(ValueError, match="Invalid request options"):
            EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                request_options=BadModel,  # Invalid - not a BaseModel
            )

    def test_validate_request_options_invalid_type(self):
        """Test request_options with invalid type raises."""
        with pytest.raises(ValueError, match="Invalid request options"):
            EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                request_options=12345,  # Invalid type
            )

    def test_serialize_request_options_none(self):
        """Test serializing None request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,
        )
        serialized = config.model_dump()
        # request_options should be None in serialization
        assert serialized.get("request_options") is None

    def test_serialize_request_options_with_model(self):
        """Test serializing request_options with model."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        serialized = config.model_dump()
        assert serialized["request_options"] is not None
        assert "properties" in serialized["request_options"]

    def test_validate_payload_no_request_options(self):
        """Test validate_payload returns data when no request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,
        )
        data = {"test": "data"}
        result = config.validate_payload(data)
        assert result == data

    def test_validate_payload_with_validation(self):
        """Test validate_payload validates against request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        data = {"message": "hello", "temperature": 0.5}
        result = config.validate_payload(data)
        assert result == data

    def test_validate_payload_invalid_raises(self):
        """Test validate_payload raises on invalid data."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        data = {"temperature": "not_a_float"}  # Invalid
        with pytest.raises(ValueError, match="Invalid payload"):
            config.validate_payload(data)

    # F6: Empty/Whitespace Credential Validation Tests
    def test_validate_api_key_empty_string(self):
        """F6: Test that empty string api_key raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            EndpointConfig(name="test", provider="test", endpoint="/test", api_key="")

    def test_validate_api_key_whitespace_only(self):
        """F6: Test that whitespace-only api_key raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            EndpointConfig(name="test", provider="test", endpoint="/test", api_key="   ")

    def test_validate_api_key_empty_env_var(self):
        """F6: Test that empty env var value raises ValueError."""
        os.environ["EMPTY_VAR"] = ""
        try:
            with pytest.raises(ValueError, match="is empty or whitespace"):
                EndpointConfig(name="test", provider="test", endpoint="/test", api_key="EMPTY_VAR")
        finally:
            del os.environ["EMPTY_VAR"]

    def test_validate_api_key_whitespace_env_var(self):
        """F6: Test that whitespace-only env var value raises ValueError."""
        os.environ["WHITESPACE_VAR"] = "   "
        try:
            with pytest.raises(ValueError, match="is empty or whitespace"):
                EndpointConfig(
                    name="test", provider="test", endpoint="/test", api_key="WHITESPACE_VAR"
                )
        finally:
            del os.environ["WHITESPACE_VAR"]

    def test_validate_api_key_strips_whitespace(self):
        """F6: Test that whitespace is stripped from credentials."""
        endpoint = Endpoint(
            config={
                "name": "test",
                "provider": "test",
                "endpoint": "/test",
                "api_key": SecretStr("  sk-123  "),
            }
        )
        # Whitespace should be stripped
        assert endpoint.config._api_key.get_secret_value() == "sk-123"

    def test_validate_api_key_strips_whitespace_from_env_var(self):
        """F6: Test that whitespace is stripped from env var values."""
        os.environ["PADDED_KEY"] = "  sk-456  "
        try:
            config = EndpointConfig(
                name="test", provider="test", endpoint="/test", api_key="PADDED_KEY"
            )
            # Whitespace should be stripped from resolved value
            assert config._api_key.get_secret_value() == "sk-456"
        finally:
            del os.environ["PADDED_KEY"]

    # V1: Env Var Collision Prevention Tests
    def test_v1_system_env_var_blocked(self):
        """V1: Test that system env vars are blocked."""
        with pytest.raises(ValueError, match="is a system environment variable"):
            EndpointConfig(name="test", provider="test", endpoint="/test", api_key="HOME")

    def test_v1_uppercase_without_env_var_treated_as_credential(self):
        """V1: Test UPPERCASE pattern without env var is treated as raw credential."""
        config = EndpointConfig(
            name="test", provider="test", endpoint="/test", api_key="MYUPPERKEY"
        )
        # UPPERCASE pattern but no env var exists → treat as raw credential
        assert config.api_key is None  # Cleared (was raw credential)
        assert config._api_key.get_secret_value() == "MYUPPERKEY"
        assert config.api_key_is_env is False

    def test_v1_lowercase_always_treated_as_credential(self):
        """V1: Test lowercase credentials are always treated as raw credentials."""
        config = EndpointConfig(
            name="test", provider="test", endpoint="/test", api_key="sk-lowercase-123"
        )
        assert config.api_key is None  # Cleared
        assert config._api_key.get_secret_value() == "sk-lowercase-123"
        assert config.api_key_is_env is False

    def test_v1_multiple_system_vars_blocked(self):
        """V1: Test multiple system env vars are blocked."""
        system_vars = ["PATH", "USER", "SHELL", "HOME", "PYTHONPATH"]
        for var in system_vars:
            with pytest.raises(ValueError, match="is a system environment variable"):
                EndpointConfig(name="test", provider="test", endpoint="/test", api_key=var)

    # V2: api_key_is_env Metadata Flag Tests
    def test_v2_api_key_is_env_true_for_resolved_env_var(self):
        """V2: Test api_key_is_env is True for resolved env var."""
        os.environ["TEST_KEY"] = "value123"
        try:
            config = EndpointConfig(
                name="test", provider="test", endpoint="/test", api_key="TEST_KEY"
            )
            assert config.api_key == "TEST_KEY"  # Kept (env var name)
            assert config.api_key_is_env is True
            assert config._api_key.get_secret_value() == "value123"
        finally:
            del os.environ["TEST_KEY"]

    def test_v2_api_key_is_env_false_for_raw_credential(self):
        """V2: Test api_key_is_env is False for raw credential."""
        config = EndpointConfig(name="test", provider="test", endpoint="/test", api_key="sk-raw")
        assert config.api_key is None  # Cleared
        assert config.api_key_is_env is False

    def test_v2_deserialization_fails_if_env_var_missing(self):
        """V2: Test deserialization raises if env var referenced but missing."""
        # Simulate deserialization with missing env var
        with pytest.raises(ValueError, match="not found during deserialization"):
            EndpointConfig.model_validate(
                {
                    "name": "test",
                    "provider": "test",
                    "endpoint": "/test",
                    "api_key": "MISSING_VAR",
                    "api_key_is_env": True,
                }
            )

    def test_v2_deserialization_succeeds_with_existing_env_var(self):
        """V2: Test deserialization succeeds when env var exists."""
        os.environ["EXISTING_VAR"] = "secret_value"
        try:
            config = EndpointConfig.model_validate(
                {
                    "name": "test",
                    "provider": "test",
                    "endpoint": "/test",
                    "api_key": "EXISTING_VAR",
                    "api_key_is_env": True,
                }
            )
            assert config.api_key == "EXISTING_VAR"
            assert config.api_key_is_env is True
            # Note: _api_key won't be set during deserialization unless we run the model_validator
            # which happens automatically when we construct the config
        finally:
            del os.environ["EXISTING_VAR"]

    def test_v2_api_key_is_env_false_for_secretstr_input(self):
        """V2: Test api_key_is_env is False when using SecretStr input."""
        endpoint = Endpoint(
            config={
                "name": "test",
                "provider": "test",
                "endpoint": "/test",
                "api_key": SecretStr("secret_key_789"),
            }
        )
        # SecretStr → raw credential → api_key_is_env should be False
        assert endpoint.config.api_key is None  # Cleared
        assert endpoint.config.api_key_is_env is False

    # Line 86: api_key_env property test
    def test_api_key_env_property(self):
        """Test api_key_env property returns api_key value."""
        os.environ["TEST_API_KEY_ENV"] = "test_value"
        try:
            config = EndpointConfig(
                name="test",
                provider="test",
                endpoint="/test",
                api_key="TEST_API_KEY_ENV",
            )
            # api_key_env is an alias for api_key
            assert config.api_key_env == "TEST_API_KEY_ENV"
            assert config.api_key_env == config.api_key
        finally:
            del os.environ["TEST_API_KEY_ENV"]

    # Line 162: Invalid params validation
    def test_endpoint_params_invalid_params_raises(self):
        """Test that invalid params raise ValueError (line 162)."""
        with pytest.raises(ValueError, match="Invalid params"):
            EndpointConfig(
                name="test",
                provider="test",
                endpoint="/api/{version}/{resource}",
                endpoint_params=["version", "resource"],
                params={
                    "version": "v1",
                    "invalid_key": "value",
                },  # invalid_key not in endpoint_params
            )

    # Line 169: Missing params warning
    def test_endpoint_params_missing_params_warns(self):
        """Test that missing params trigger warning (line 169)."""
        with patch("lionpride.services.types.endpoint.logger") as mock_logger:
            _ = EndpointConfig(
                name="test",
                provider="test",
                endpoint="/api/{version}/{resource}",
                endpoint_params=["version", "resource"],
                params={"version": "v1"},  # Missing 'resource'
            )
            # Should have logged a warning about missing 'resource'
            mock_logger.warning.assert_called_once()
            warning_msg = str(mock_logger.warning.call_args)
            assert "resource" in warning_msg
            assert "missing" in warning_msg.lower() or "not provided" in warning_msg.lower()


class TestEndpoint:
    """Test Endpoint class."""

    def test_init_with_dict_config(self):
        """Test Endpoint initialization with dict config."""
        config_dict = {
            "name": "test",
            "provider": "test",
            "endpoint": "/test",
            "base_url": "https://api.test.com",
            "request_options": SimpleRequest,
        }
        endpoint = Endpoint(config=config_dict)
        assert endpoint.config.name == "test"
        assert endpoint.config.provider == "test"

    def test_init_with_endpoint_config(self):
        """Test Endpoint initialization with EndpointConfig instance."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)
        assert endpoint.config.name == "test"

    def test_init_with_invalid_config_type(self):
        """Test Endpoint initialization with invalid config type."""
        with pytest.raises(ValueError, match="Config must be a dict or EndpointConfig"):
            Endpoint(config="invalid")

    def test_request_options_property(self):
        """Test request_options property."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)
        assert endpoint.request_options == SimpleRequest

    def test_create_payload_with_extra_headers(self):
        """Test create_payload with extra headers."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
            auth_type="bearer",
        )
        endpoint = Endpoint(config=config)
        request = {"message": "test", "temperature": 0.8}
        extra_headers = {"X-Custom": "header"}

        _payload, headers = endpoint.create_payload(request, extra_headers=extra_headers)

        assert "X-Custom" in headers
        assert headers["X-Custom"] == "header"
        assert "Authorization" in headers

    def test_create_payload_with_kwargs(self):
        """Test create_payload merges kwargs."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)
        request = {"message": "test"}

        payload, _headers = endpoint.create_payload(
            request, extra_param="extra_value", temperature=0.9
        )

        assert payload["message"] == "test"
        assert payload["temperature"] == 0.9  # From kwargs

    def test_create_payload_no_request_options_raises(self):
        """Test create_payload raises if no request_options."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=None,  # No schema
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        with pytest.raises(ValueError, match="must define request_options"):
            endpoint.create_payload({"data": "test"})

    def test_create_payload_no_request_options_error_message_helpful(self):
        """Test error message is helpful for operators when request_options is None."""
        config = EndpointConfig(
            name="my_custom_endpoint",
            provider="test_provider",
            endpoint="/api/test",
            request_options=None,  # No schema
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        with pytest.raises(ValueError) as exc_info:
            endpoint.create_payload({"data": "test"})

        error_msg = str(exc_info.value)

        # Verify error message contains:
        # 1. Which endpoint failed
        assert "my_custom_endpoint" in error_msg, "Error should include endpoint name"

        # 2. What the problem is
        assert "request_options" in error_msg, "Error should mention request_options"

        # 3. What to do about it
        assert "must define" in error_msg or "must use" in error_msg, (
            "Error should provide guidance"
        )

        # 4. Context about validation requirement
        assert "proper request validation" in error_msg, (
            "Error should explain why request_options is needed"
        )

    def test_create_payload_no_request_options_different_endpoints(self):
        """Test error message includes correct endpoint name for different endpoints."""
        endpoint_names = ["auth_endpoint", "data_processor", "payment_gateway"]

        for name in endpoint_names:
            config = EndpointConfig(
                name=name,
                provider="test",
                endpoint="/test",
                request_options=None,
                api_key="test_key",
            )
            endpoint = Endpoint(config=config)

            with pytest.raises(ValueError) as exc_info:
                endpoint.create_payload({"data": "test"})

            error_msg = str(exc_info.value)
            assert name in error_msg, f"Error message should include endpoint name '{name}'"

    @pytest.mark.asyncio
    async def test_no_request_options_no_cascading_failures(self):
        """Test that request_options=None error doesn't cause cascading failures."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=None,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # First call should raise ValueError
        with pytest.raises(ValueError, match="must define request_options"):
            await endpoint.call(request={"data": "test"})

        # Endpoint should still be usable - config should not be corrupted
        assert endpoint.config.name == "test_endpoint"
        assert endpoint.config.provider == "test"
        assert endpoint.config.request_options is None

        # Second call should raise the same error (not a different error)
        with pytest.raises(ValueError, match="must define request_options"):
            await endpoint.call(request={"data": "test2"})

        # Verify the endpoint's internal state is still intact
        assert endpoint.full_url == "https://api.test.com/test"
        assert endpoint.config._api_key.get_secret_value() == "test_key"

    @pytest.mark.asyncio
    async def test_call_skip_payload_creation_dict(self):
        """Test call with skip_payload_creation and dict request."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock _call
        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(
            request={"message": "test", "temperature": 0.5},
            skip_payload_creation=True,
        )

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_call_skip_payload_creation_basemodel(self):
        """Test call with skip_payload_creation and BaseModel request."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock _call
        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        request_model = SimpleRequest(message="test", temperature=0.5)
        result = await endpoint.call(
            request=request_model,
            skip_payload_creation=True,
        )

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_call_with_retry_only(self):
        """Test call with retry_config only (no circuit breaker)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01)
        endpoint = Endpoint(config=config, retry_config=retry_config)

        call_count = 0

        async def mock_call(payload, headers, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry me")
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker_only(self):
        """Test call with circuit_breaker only (no retry)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = Endpoint(config=config, circuit_breaker=circuit_breaker)

        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_call_with_both_retry_and_circuit_breaker(self):
        """Test call with both retry_config and circuit_breaker."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01)
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time=1.0)
        endpoint = Endpoint(
            config=config, retry_config=retry_config, circuit_breaker=circuit_breaker
        )

        call_count = 0

        async def mock_call(payload, headers, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry error")
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_call_http_success(self):
        """Test _call_http with successful response."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_call_http_429_raises(self):
        """Test _call_http with 429 rate limit raises."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

    @pytest.mark.asyncio
    async def test_call_http_500_raises(self):
        """Test _call_http with 500 server error raises."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

    @pytest.mark.asyncio
    async def test_call_http_non_200_with_json_error(self):
        """Test _call_http with non-200 status and JSON error body."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request", "code": "INVALID"}
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError, match="400"),
        ):
            await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

    @pytest.mark.asyncio
    async def test_call_http_non_200_without_json_error(self):
        """Test _call_http with non-200 status and no JSON error body."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.side_effect = Exception("Not JSON")
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError, match="403"),
        ):
            await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

    @pytest.mark.asyncio
    async def test_stream(self):
        """Test stream method."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        async def mock_stream_http(payload, headers, **kwargs):
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        endpoint._stream_http = mock_stream_http

        chunks = []
        async for chunk in endpoint.stream(request={"message": "test", "temperature": 0.7}):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_stream_http(self):
        """Test _stream_http method."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            yield "line1"
            yield ""  # Empty line (should be skipped)
            yield "line2"
            yield "line3"

        mock_response.aiter_lines = mock_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            chunks = []
            async for chunk in endpoint._stream_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            ):
                chunks.append(chunk)

        # Empty line should be filtered out
        assert chunks == ["line1", "line2", "line3"]

    @pytest.mark.asyncio
    async def test_stream_http_error(self):
        """Test _stream_http with error response."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.request = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError, match="500"),
        ):
            async for _ in endpoint._stream_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            ):
                pass

    def test_to_dict_with_retry_config(self):
        """Test to_dict with retry_config."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        retry_config = RetryConfig(max_retries=3, initial_delay=0.1)
        endpoint = Endpoint(config=config, retry_config=retry_config)

        result = endpoint.to_dict()

        assert result["retry_config"] is not None
        assert result["retry_config"]["max_retries"] == 3

    def test_to_dict_with_circuit_breaker(self):
        """Test to_dict with circuit_breaker."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time=2.0)
        endpoint = Endpoint(config=config, circuit_breaker=circuit_breaker)

        result = endpoint.to_dict()

        assert result["circuit_breaker"] is not None
        assert result["circuit_breaker"]["failure_threshold"] == 5

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "config": {
                "name": "test",
                "provider": "test",
                "endpoint": "/test",
                "request_options": SimpleRequest.model_json_schema(),
            },
            "retry_config": {
                "max_retries": 3,
                "initial_delay": 0.1,
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_time": 2.0,
            },
        }

        endpoint = Endpoint.from_dict(data)

        assert endpoint.config.name == "test"
        assert endpoint.retry_config is not None
        assert endpoint.retry_config.max_retries == 3
        assert endpoint.circuit_breaker is not None
        assert endpoint.circuit_breaker.failure_threshold == 5

    def test_from_dict_invalid_type(self):
        """Test from_dict with invalid type raises."""
        # Element.from_dict() expects dict, raises AttributeError for non-dict
        with pytest.raises(AttributeError, match="'str' object has no attribute 'copy'"):
            Endpoint.from_dict("not a dict")

    def test_create_http_client(self):
        """Test _create_http_client creates httpx.AsyncClient."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            timeout=120,
            client_kwargs={"follow_redirects": True},
        )
        endpoint = Endpoint(config=config)

        # Create client
        client = endpoint._create_http_client()

        # Verify it's an httpx.AsyncClient
        assert client.__class__.__name__ == "AsyncClient"
        # Timeout should be configured
        assert client.timeout.read == 120

    def test_from_dict_with_none_values(self):
        """Test from_dict with None retry_config and circuit_breaker."""
        data = {
            "config": {
                "name": "test",
                "provider": "test",
                "endpoint": "test",
                "request_options": SimpleRequest.model_json_schema(),
            },
            "retry_config": None,
            "circuit_breaker": None,
        }

        endpoint = Endpoint.from_dict(data)

        assert endpoint.config.name == "test"
        assert endpoint.retry_config is None
        assert endpoint.circuit_breaker is None

    # Line 224: Empty SecretStr validation in Endpoint.__init__
    def test_init_with_empty_secretstr_raises(self):
        """Test Endpoint init with empty SecretStr raises ValueError (line 224)."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            Endpoint(
                config={
                    "name": "test",
                    "provider": "test",
                    "endpoint": "/test",
                    "api_key": SecretStr(""),  # Empty SecretStr
                }
            )

    def test_init_with_whitespace_secretstr_raises(self):
        """Test Endpoint init with whitespace-only SecretStr raises ValueError (line 224)."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            Endpoint(
                config={
                    "name": "test",
                    "provider": "test",
                    "endpoint": "/test",
                    "api_key": SecretStr("   "),  # Whitespace-only SecretStr
                }
            )

    # Line 490: Invalid circuit_breaker type
    def test_deserialize_circuit_breaker_invalid_type_raises(self):
        """Test circuit_breaker validator with invalid type (line 490)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="circuit_breaker must be a dict"):
            Endpoint.from_dict(
                {
                    "config": {
                        "name": "test",
                        "provider": "test",
                        "endpoint": "/test",
                        "request_options": SimpleRequest.model_json_schema(),
                    },
                    "circuit_breaker": "invalid_string",  # Invalid type
                }
            )

    # Line 505: Invalid retry_config type
    def test_deserialize_retry_config_invalid_type_raises(self):
        """Test retry_config validator with invalid type (line 505)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="retry_config must be a dict"):
            Endpoint.from_dict(
                {
                    "config": {
                        "name": "test",
                        "provider": "test",
                        "endpoint": "/test",
                        "request_options": SimpleRequest.model_json_schema(),
                    },
                    "retry_config": 12345,  # Invalid type (int)
                }
            )

    # Line 246: event_type property
    def test_event_type_property(self):
        """Test event_type property returns APICalling (line 246)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)

        # event_type should return APICalling class
        assert endpoint.event_type == APICalling


class TestAPICalling:
    """Test APICalling class."""

    @pytest.mark.asyncio
    async def test_invoke(self):
        """Test _invoke method calls backend.call."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock _call_http instead since call is an async method we can't easily patch on Pydantic
        async def mock_call_http(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call_http = mock_call_http

        calling = APICalling(
            backend=endpoint,
            payload={"message": "test", "temperature": 0.7},
            headers={"X-Custom": "header"},
        )

        result = await calling._invoke()

        # Should get the normalized response back
        assert result.status == "success"
        assert result.data == {"result": "success"}

    # Lines 567-568: Token estimation with messages
    def test_required_tokens_with_messages(self):
        """Test required_tokens calculates tokens from messages (lines 567-568)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ]
            },
            headers={},
        )

        tokens = calling.required_tokens
        # Should estimate tokens from messages
        assert tokens is not None
        assert tokens > 0

    # Lines 571-572: Token estimation with input (embeddings)
    def test_required_tokens_with_input_string(self):
        """Test required_tokens calculates tokens from string input (lines 571-572)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"input": "This is a test string for embeddings"},
            headers={},
        )

        tokens = calling.required_tokens
        # Should estimate tokens from input
        assert tokens is not None
        assert tokens > 0

    def test_required_tokens_with_input_list(self):
        """Test required_tokens calculates tokens from list input (line 592)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"input": ["First text", "Second text", "Third text"]},
            headers={},
        )

        tokens = calling.required_tokens
        # Should estimate tokens from list of texts
        assert tokens is not None
        assert tokens > 0

    # Line 575: Default return None
    def test_required_tokens_returns_none_for_unknown_payload(self):
        """Test required_tokens returns None for unknown payload format (line 575)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"custom_field": "unknown format"},  # No 'messages' or 'input'
            headers={},
        )

        tokens = calling.required_tokens
        # Should return None for unknown payload format
        assert tokens is None

    def test_required_tokens_returns_none_when_not_required(self):
        """Test required_tokens returns None when requires_tokens=False."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=False,  # Token tracking disabled
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"messages": [{"role": "user", "content": "test"}]},
            headers={},
        )

        tokens = calling.required_tokens
        # Should return None when token tracking is disabled
        assert tokens is None

    # Lines 583-586: _estimate_message_tokens method
    def test_estimate_message_tokens(self):
        """Test _estimate_message_tokens calculates correctly (lines 583-586)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={},
            headers={},
        )

        # Test with sample messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        tokens = calling._estimate_message_tokens(messages)
        # Should return ~4 chars per token + overhead
        assert tokens > 0
        # Roughly: (5 + 4 + 8 + 9) / 4 + 10 = ~16 tokens
        assert tokens >= 10

    # Lines 590-592: _estimate_text_tokens method
    def test_estimate_text_tokens_string(self):
        """Test _estimate_text_tokens with string input (line 590-591)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={},
            headers={},
        )

        tokens = calling._estimate_text_tokens("This is a test string")
        # Should return ~chars / 4
        assert tokens > 0
        assert tokens == len("This is a test string") // 4

    def test_estimate_text_tokens_list(self):
        """Test _estimate_text_tokens with list input (line 592)."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={},
            headers={},
        )

        texts = ["First text", "Second text", "Third text"]
        tokens = calling._estimate_text_tokens(texts)
        # Should sum tokens from all texts
        assert tokens > 0
        expected = sum(len(t) // 4 for t in texts)
        assert tokens == expected

    def test_permission_request_property(self):
        """Test permission_request returns required_tokens."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"messages": [{"role": "user", "content": "test"}]},
            headers={},
        )

        # request returns permission data for rate limiting
        request = calling.request
        assert "required_tokens" in request
        assert request["required_tokens"] is not None

    def test_call_args_returns_backend_arguments(self):
        """Test call_args property returns arguments for backend.call()."""
        config = EndpointConfig(
            name="test",
            provider="test",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"message": "test"},
            headers={"X-Custom": "value"},
        )

        # call_args should return backend.call() arguments
        call_args = calling.call_args
        assert call_args["request"] == {"message": "test"}
        assert call_args["extra_headers"] == {"X-Custom": "value"}
        assert call_args["skip_payload_creation"] is True
