# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for src/lionpride/services/types/backend.py

Targets 100% coverage of:
- NormalizedResponse class
- Calling class (Event-based async invoke/stream)
- ServiceBackend abstract class
- ServiceConfig
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lionpride import EventStatus
from lionpride.services.types.backend import (
    Calling,
    NormalizedResponse,
    ServiceBackend,
    ServiceConfig,
)
from lionpride.services.types.hook import HookEvent, HookPhase, HookRegistry

# =============================================================================
# Test Fixtures
# =============================================================================


class PydanticTestModel(BaseModel):
    """Test Pydantic model for normalize_response tests."""

    name: str
    value: int


class BrokenPydanticModel(BaseModel):
    """Model that raises exception in model_dump()."""

    name: str

    def model_dump(self, **kwargs):
        raise RuntimeError("model_dump failed")


class MockCalling(Calling):
    """Mock Calling implementation for testing.

    Uses parent Calling._invoke() which calls backend.call(**self.call_args).
    MockServiceBackend.call() can be configured to return different responses.
    """

    @property
    def call_args(self) -> dict:
        """Return payload as call arguments."""
        return self.payload

    async def _stream(self):
        """Mock stream implementation."""
        raise NotImplementedError("Stream not implemented")


class MockServiceBackend(ServiceBackend):
    """Mock ServiceBackend for testing properties."""

    result_value: str = "test_result"
    should_fail: bool = False
    should_cancel: bool = False

    @property
    def event_type(self) -> type[Calling]:
        """Return MockCalling type."""
        return MockCalling

    async def call(self, *args, **kw) -> NormalizedResponse:
        """Mock call implementation."""
        if self.should_cancel:
            import asyncio

            raise asyncio.CancelledError("Test cancellation")
        if self.should_fail:
            raise RuntimeError("Test failure")

        return NormalizedResponse(
            status="success",
            data=self.result_value,
            raw_response={"result": self.result_value},
        )


@pytest.fixture
def hook_registry():
    """Create empty hook registry."""
    return HookRegistry()


@pytest.fixture
def mock_backend():
    """Create mock service backend."""
    config = ServiceConfig(provider="test", name="test")
    return MockServiceBackend(config=config)


@pytest.fixture
def mock_calling(mock_backend):
    """Create mock calling instance."""
    return MockCalling(backend=mock_backend, payload={})


# =============================================================================
# NormalizedResponse Tests
# =============================================================================


class TestNormalizedResponse:
    """Test NormalizedResponse class."""

    def test_normalized_response_basic(self):
        """Test basic NormalizedResponse creation."""
        response = NormalizedResponse(
            status="success",
            data={"key": "value"},
            raw_response={"original": "data"},
        )

        assert response.status == "success"
        assert response.data == {"key": "value"}
        assert response.error is None
        assert response.raw_response == {"original": "data"}
        assert response.metadata is None

    def test_normalized_response_with_error(self):
        """Test NormalizedResponse with error."""
        response = NormalizedResponse(
            status="error",
            error="Something went wrong",
            raw_response={"error": "details"},
        )

        assert response.status == "error"
        assert response.error == "Something went wrong"
        assert response.data is None

    def test_to_dict_excludes_none(self):
        """Test to_dict() excludes None values."""
        response = NormalizedResponse(
            status="success",
            data="result",
            raw_response={"result": "test"},
        )

        result = response.to_dict()
        assert "error" not in result
        assert "metadata" not in result
        assert result["status"] == "success"
        assert result["data"] == "result"

    def test_to_dict_includes_metadata(self):
        """Test to_dict() includes metadata when present."""
        response = NormalizedResponse(
            status="success",
            data="result",
            raw_response={"result": "test"},
            metadata={"usage": {"tokens": 100}},
        )

        result = response.to_dict()
        assert result["metadata"] == {"usage": {"tokens": 100}}


# =============================================================================
# Calling Class Tests
# =============================================================================


class TestCalling:
    """Test Calling class invoke() and hook integration."""

    @pytest.mark.asyncio
    async def test_invoke_success_no_hooks(self, mock_calling):
        """Test successful invoke without hooks."""
        from lionpride import Unset

        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.COMPLETED
        assert mock_calling.execution.response.status == "success"
        assert mock_calling.execution.response.data == "test_result"
        # Error is Unset (sentinel value) or None after successful completion
        assert mock_calling.execution.error in (None, Unset)
        assert mock_calling.execution.duration > 0

    @pytest.mark.asyncio
    async def test_invoke_failure(self, mock_calling):
        """Test invoke with failure."""
        mock_calling.backend.should_fail = True
        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.FAILED
        assert str(mock_calling.execution.error) == "Test failure"

    @pytest.mark.asyncio
    async def test_invoke_cancellation(self, mock_calling):
        """Test invoke with cancellation."""
        import asyncio

        mock_calling.backend.should_cancel = True

        with pytest.raises(asyncio.CancelledError):
            await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.CANCELLED
        assert isinstance(mock_calling.execution.error, asyncio.CancelledError)

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_success(self, mock_calling):
        """Test invoke with pre-invoke hook that succeeds."""
        # Create a mock hook event
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook.execution.error = None
        mock_hook._should_exit = False
        mock_hook.invoke = AsyncMock()

        mock_calling._pre_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionpride.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.COMPLETED
        mock_hook.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_failed(self, mock_calling):
        """Test invoke when pre-invoke hook fails."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def failing_hook():
            mock_hook.execution.status = EventStatus.FAILED
            mock_hook.execution.error = "Hook failed"

        mock_hook.invoke = failing_hook

        mock_calling._pre_invoke_hook_event = mock_hook

        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Pre-invoke hook failed:" in str(mock_calling.execution.error)

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_cancelled(self, mock_calling):
        """Test invoke when pre-invoke hook is cancelled."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def cancelled_hook():
            mock_hook.execution.status = EventStatus.CANCELLED
            mock_hook.execution.error = "Hook cancelled"

        mock_hook.invoke = cancelled_hook

        mock_calling._pre_invoke_hook_event = mock_hook

        await mock_calling.invoke()

        # Pre-hook with CANCELLED status → Calling._invoke raises RuntimeError
        # RuntimeError is Exception → Event.invoke sets status=FAILED (not CANCELLED)
        assert mock_calling.execution.status == EventStatus.FAILED
        assert isinstance(mock_calling.execution.error, RuntimeError)
        assert "Pre-invoke hook cancelled:" in str(mock_calling.execution.error)

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_exit_with_cause(self, mock_calling):
        """Test invoke when pre-invoke hook requests exit with cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = RuntimeError("Exit requested")
        mock_hook.invoke = AsyncMock()

        mock_calling._pre_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionpride.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Exit requested" in str(mock_calling.execution.error)

    @pytest.mark.asyncio
    async def test_invoke_with_pre_hook_exit_without_cause(self, mock_calling):
        """Test invoke when pre-invoke hook requests exit without cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = None
        mock_hook.invoke = AsyncMock()

        mock_calling._pre_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionpride.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "requested exit without a cause" in str(mock_calling.execution.error)

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_success(self, mock_calling):
        """Test invoke with post-invoke hook that succeeds."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook.execution.error = None
        mock_hook._should_exit = False
        mock_hook.invoke = AsyncMock()

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionpride.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.COMPLETED
        assert mock_calling.execution.response.status == "success"
        assert mock_calling.execution.response.data == "test_result"
        mock_hook.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_failed(self, mock_calling):
        """Test invoke when post-invoke hook fails."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def failing_hook():
            mock_hook.execution.status = EventStatus.FAILED
            mock_hook.execution.error = "Hook failed"

        mock_hook.invoke = failing_hook

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent validation error on mock_hook
        with patch("lionpride.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Post-hook failure only logs warning when _should_exit=False
        # Overall call succeeds with COMPLETED status
        assert mock_calling.execution.status == EventStatus.COMPLETED
        assert mock_calling.execution.response.status == "success"
        assert mock_calling.execution.response.data == "test_result"

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_cancelled(self, mock_calling):
        """Test invoke when post-invoke hook is cancelled.

        Post-hook cancellation logs warning but doesn't fail the overall call
        unless broadcast fails.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.PENDING
        mock_hook._should_exit = False

        async def cancelled_hook():
            mock_hook.execution.status = EventStatus.CANCELLED
            mock_hook.execution.error = "Hook cancelled"

        mock_hook.invoke = cancelled_hook

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock broadcast to raise exception (simulating broadcast failure)
        with patch(
            "lionpride.services.types.backend.HookBroadcaster.broadcast",
            side_effect=RuntimeError("Broadcast failed"),
        ):
            await mock_calling.invoke()

        # Overall call fails due to broadcast exception after post-hook cancellation
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Broadcast failed" in str(mock_calling.execution.error)

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_exit_with_cause(self, mock_calling):
        """Test invoke when post-invoke hook requests exit with cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = RuntimeError("Exit requested")
        mock_hook.invoke = AsyncMock()

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionpride.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "Exit requested" in str(mock_calling.execution.error)

    @pytest.mark.asyncio
    async def test_invoke_with_post_hook_exit_without_cause(self, mock_calling):
        """Test invoke when post-invoke hook requests exit without cause."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_hook = MagicMock()
        mock_hook.execution = MagicMock()
        mock_hook.execution.status = EventStatus.COMPLETED
        mock_hook._should_exit = True
        mock_hook._exit_cause = None
        mock_hook.invoke = AsyncMock()

        mock_calling._post_invoke_hook_event = mock_hook

        # Mock HookBroadcaster to prevent actual broadcast calls
        with patch("lionpride.services.types.backend.HookBroadcaster.broadcast", new=AsyncMock()):
            await mock_calling.invoke()

        # Exception caught and converted to FAILED status
        assert mock_calling.execution.status == EventStatus.FAILED
        assert "requested exit without a cause" in str(mock_calling.execution.error)

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self, mock_calling):
        """Test that _stream raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await mock_calling._stream()


# =============================================================================
# ServiceConfig Tests
# =============================================================================


class TestServiceConfig:
    """Test ServiceConfig class."""

    def test_service_config_basic(self):
        """Test basic ServiceConfig creation."""
        config = ServiceConfig(provider="test", name="test_service")

        assert config.provider == "test"
        assert config.name == "test_service"
        assert config.request_options is None

    def test_service_config_with_request_options(self):
        """Test ServiceConfig with request_options."""

        class TestOptions(BaseModel):
            timeout: int = 30

        config = ServiceConfig(provider="test", name="test_service", request_options=TestOptions)

        assert config.request_options == TestOptions


# =============================================================================
# ServiceBackend Tests
# =============================================================================


class TestServiceBackend:
    """Test ServiceBackend abstract class properties."""

    def test_provider_property(self):
        """Test provider property."""
        config = ServiceConfig(provider="test_provider", name="test")
        backend = MockServiceBackend(config=config)

        assert backend.provider == "test_provider"

    def test_name_property(self):
        """Test name property."""
        config = ServiceConfig(provider="test", name="test_name")
        backend = MockServiceBackend(config=config)

        assert backend.name == "test_name"

    def test_version_property_from_config(self):
        """Test version property when config has version attribute."""

        # Create a config class that supports version
        class VersionedConfig(ServiceConfig):
            version: str | None = None

        config = VersionedConfig(provider="test", name="test", version="1.0.0")
        backend = MockServiceBackend(config=config)

        assert backend.version == "1.0.0"

    def test_tags_property_from_config_list(self):
        """Test tags property when config has tags as list."""

        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: list[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags=["tag1", "tag2"])
        backend = MockServiceBackend(config=config)

        assert backend.tags == {"tag1", "tag2"}

    def test_tags_property_from_config_tuple(self):
        """Test tags property when config has tags as tuple."""

        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: tuple[str, ...] | None = None

        config = TaggedConfig(provider="test", name="test", tags=("tag1", "tag2"))
        backend = MockServiceBackend(config=config)

        assert backend.tags == {"tag1", "tag2"}

    def test_tags_property_from_config_set(self):
        """Test tags property when config has tags as set."""

        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: set[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags={"tag1", "tag2"})
        backend = MockServiceBackend(config=config)

        assert backend.tags == {"tag1", "tag2"}

    def test_tags_property_from_config_none(self):
        """Test tags property when config.tags is None."""

        # Create a config class that supports tags
        class TaggedConfig(ServiceConfig):
            tags: list[str] | None = None

        config = TaggedConfig(provider="test", name="test", tags=None)
        backend = MockServiceBackend(config=config)

        assert backend.tags == set()

    def test_tags_property_from_metadata_empty(self):
        """Test tags property from metadata when empty."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        assert backend.tags == set()

    def test_request_options_property_getter(self):
        """Test request_options property getter."""

        class TestOptions(BaseModel):
            timeout: int = 30

        config = ServiceConfig(provider="test", name="test", request_options=TestOptions)
        backend = MockServiceBackend(config=config)

        assert backend.request_options == TestOptions

    def test_request_options_property_getter_none(self):
        """Test request_options property getter when None."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        assert backend.request_options is None

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self):
        """Test that stream() raises NotImplementedError."""
        config = ServiceConfig(provider="test", name="test")
        backend = MockServiceBackend(config=config)

        with pytest.raises(NotImplementedError, match="does not support streaming calls"):
            await backend.stream()


# =============================================================================
# ServiceConfig Tests - request_options validation
# =============================================================================


class TestServiceConfigRequestOptions:
    """Test ServiceConfig request_options validation (lines 74-75)."""

    def test_validate_request_options_when_invalid_type_then_raises(self):
        """Test _validate_request_options raises ValueError for invalid types (line 74-75).

        When request_options is not a Pydantic model, BaseModel instance, or dict/str,
        the validator raises ValueError with 'Invalid request options' message.
        """

        class NotPydantic:
            """Class that's not a Pydantic BaseModel."""

            pass

        # Create an instance that's not a dict, str, or BaseModel
        # The validator checks: isinstance(v, type) and issubclass(v, BaseModel)
        # For a non-BaseModel type, it will fall through to raise ValueError
        with pytest.raises(ValueError, match="Invalid request options"):
            ServiceConfig(provider="test", name="test_service", request_options=NotPydantic())

    def test_validate_request_options_when_exception_in_schema_load(self):
        """Test _validate_request_options raises ValueError when schema loading fails.

        This tests line 74-75: except Exception as e: raise ValueError(...) from e
        """
        # Pass an invalid type that will cause the schema loading to fail
        # A number is not a valid request_options (not dict, not str, not BaseModel)
        with pytest.raises(ValueError, match="Invalid request options"):
            ServiceConfig(provider="test", name="test_service", request_options=12345)


# =============================================================================
# Calling Tests - response property
# =============================================================================


class TestCallingResponseProperty:
    """Test Calling.response property (lines 125-127)."""

    def test_response_property_when_unset_then_returns_unset(self, mock_calling):
        """Test response property returns Unset when execution.response is Unset (line 125-127).

        Before invoke() is called, execution.response is Unset (sentinel value).
        The response property should return Unset in this case.
        """
        from lionpride.types import Unset, is_sentinel

        # Before invoke(), execution.response should be Unset
        assert is_sentinel(mock_calling.execution.response)

        # Access response property - should return Unset
        response = mock_calling.response
        assert response is Unset

    @pytest.mark.asyncio
    async def test_response_property_when_completed_then_returns_response(self, mock_calling):
        """Test response property returns NormalizedResponse after successful invoke."""
        await mock_calling.invoke()

        response = mock_calling.response
        assert response is not None
        assert response.status == "success"
        assert response.data == "test_result"
