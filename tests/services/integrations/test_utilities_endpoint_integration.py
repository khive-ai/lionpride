# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for utilities with Endpoint."""

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from lionpride.services.types import Endpoint, EndpointConfig
from lionpride.services.utilities import (
    CircuitBreaker,
    CircuitState,
    RetryConfig,
)


class TestEndpointIntegration:
    """Test Endpoint integration with resilience patterns."""

    @pytest.mark.asyncio
    async def test_endpoint_with_circuit_breaker(self):
        """Endpoint integrates CircuitBreaker correctly."""
        config = EndpointConfig(
            name="test-endpoint",
            provider="test",
            base_url="https://example.com",
            endpoint="api/test",
            method="POST",
        )

        cb = CircuitBreaker(failure_threshold=2, recovery_time=1.0, name="test")
        endpoint = Endpoint(config=config, circuit_breaker=cb)

        assert endpoint.circuit_breaker is cb

    @pytest.mark.asyncio
    async def test_endpoint_with_retry_config(self):
        """Endpoint integrates RetryConfig correctly."""
        config = EndpointConfig(
            name="test-endpoint",
            provider="test",
            base_url="https://example.com",
            endpoint="api/test",
            method="POST",
        )

        retry = RetryConfig(max_retries=3, initial_delay=0.1)
        endpoint = Endpoint(config=config, retry_config=retry)

        assert endpoint.retry_config is retry

    @pytest.mark.asyncio
    async def test_endpoint_resilience_layering(self):
        """Endpoint layers retry → circuit breaker → HTTP correctly."""

        # Define minimal request schema
        class TestRequest(BaseModel):
            pass

        config = {
            "name": "test-endpoint",
            "provider": "test",
            "base_url": "https://httpbin.org",
            "endpoint": "status/500",  # Returns 500 error
            "method": "GET",
            "api_key": SecretStr("dummy_key"),
            "request_options": TestRequest,
        }

        cb = CircuitBreaker(
            failure_threshold=1, recovery_time=10.0, name="test"
        )  # Open after 1 failure
        retry = RetryConfig(max_retries=1, initial_delay=0.01)

        endpoint = Endpoint(config=config, circuit_breaker=cb, retry_config=retry)

        # This should fail after retries and open circuit
        with pytest.raises(httpx.HTTPStatusError):
            await endpoint.call(request={})

        # Circuit should be OPEN after the failure (threshold=1)
        assert cb.state == CircuitState.OPEN
