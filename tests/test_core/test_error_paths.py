# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Error hierarchy tests for lionpride.

Note: Error paths for individual classes (Pile, Progression, Flow, Graph)
are tested in their respective test files. This file tests the error
hierarchy itself - base class behavior, retryable semantics, serialization.
"""

import pytest

from lionpride.core import Element
from lionpride.errors import (
    ConfigurationError,
    ConnectionError,
    ExecutionError,
    ExistsError,
    LionprideError,
    NotFoundError,
    QueueFullError,
    TimeoutError,
    ValidationError,
)

# =============================================================================
# LionprideError Base Class (unique - tests error class behavior)
# =============================================================================


class TestLionprideErrorBase:
    """Tests for LionprideError base class behavior."""

    def test_default_message(self):
        """Default message is used when none provided."""
        err = LionprideError()
        assert err.message == "lionpride error"

    def test_custom_message(self):
        """Custom message overrides default."""
        err = LionprideError("custom error message")
        assert err.message == "custom error message"

    def test_details_dict(self):
        """Details dict is stored and accessible."""
        details = {"key": "value", "count": 42}
        err = LionprideError("test", details=details)
        assert err.details == details

    def test_retryable_default(self):
        """Default retryable flag from class attribute."""
        err = LionprideError()
        assert err.retryable is True

    def test_cause_chaining(self):
        """Cause exception is preserved for traceback."""
        original = ValueError("original error")
        err = LionprideError("wrapped error", cause=original)
        assert err.__cause__ is original

    def test_to_dict_serialization(self):
        """Error serializes to dict with all fields."""
        err = LionprideError("test message", details={"key": "value"}, retryable=False)
        data = err.to_dict()
        assert data["error"] == "LionprideError"
        assert data["message"] == "test message"
        assert data["retryable"] is False
        assert data["details"] == {"key": "value"}


# =============================================================================
# Specialized Errors (unique - tests error hierarchy properties)
# =============================================================================


class TestSpecializedErrorsRetryable:
    """Tests for specialized error subclass retryable defaults."""

    def test_validation_error_not_retryable(self):
        """ValidationError is not retryable by default."""
        err = ValidationError("validation failed")
        assert err.retryable is False

    def test_not_found_error_not_retryable(self):
        """NotFoundError is not retryable by default."""
        err = NotFoundError("item not found")
        assert err.retryable is False

    def test_exists_error_not_retryable(self):
        """ExistsError is not retryable by default."""
        err = ExistsError("item exists")
        assert err.retryable is False

    def test_timeout_error_retryable(self):
        """TimeoutError is retryable by default."""
        err = TimeoutError("operation timed out")
        assert err.retryable is True

    def test_connection_error_retryable(self):
        """ConnectionError is retryable by default."""
        err = ConnectionError("connection lost")
        assert err.retryable is True

    def test_inheritance_hierarchy(self):
        """All specialized errors inherit from LionprideError."""
        errors = [
            ValidationError(),
            ConfigurationError(),
            ExecutionError(),
            ConnectionError(),
            TimeoutError(),
            NotFoundError(),
            ExistsError(),
            QueueFullError(),
        ]
        for err in errors:
            assert isinstance(err, LionprideError)


# =============================================================================
# Retryable Consistency (unique - semantic consistency check)
# =============================================================================


class TestRetryableConsistency:
    """Tests for retryable flag consistency across error types."""

    def test_transient_errors_are_retryable(self):
        """Transient errors are retryable."""
        assert ConnectionError().retryable is True
        assert TimeoutError().retryable is True
        assert QueueFullError().retryable is True

    def test_permanent_errors_are_not_retryable(self):
        """Permanent errors are not retryable."""
        assert ValidationError().retryable is False
        assert ConfigurationError().retryable is False
        assert NotFoundError().retryable is False
        assert ExistsError().retryable is False


# =============================================================================
# Element Error Paths (unique - to_dict invalid mode)
# =============================================================================


class TestElementErrorPaths:
    """Tests for errors in Element operations."""

    def test_to_dict_invalid_mode(self):
        """Element.to_dict() raises ValueError for invalid mode."""
        elem = Element()
        with pytest.raises(ValueError):
            elem.to_dict(mode="invalid")
