# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Validator enhancements from lionagi v0.2.2.

Updated to use the new RuleRegistry-based API.
"""

import pytest

from lionpride.rules import RuleRegistry, ValidationError, Validator
from lionpride.rules.number import NumberRule
from lionpride.rules.string import StringRule


class TestValidationLog:
    """Test validation_log feature for tracking errors."""

    @pytest.mark.asyncio
    async def test_validation_log_initialized(self):
        """Test that validation_log is initialized as empty list."""
        validator = Validator()
        assert hasattr(validator, "validation_log")
        assert isinstance(validator.validation_log, list)
        assert len(validator.validation_log) == 0

    @pytest.mark.asyncio
    async def test_log_validation_error(self):
        """Test log_validation_error method logs errors."""
        validator = Validator()
        validator.log_validation_error("name", "invalid", "Value too short")

        assert len(validator.validation_log) == 1
        log_entry = validator.validation_log[0]
        assert log_entry["field"] == "name"
        assert log_entry["value"] == "invalid"
        assert log_entry["error"] == "Value too short"
        assert "timestamp" in log_entry

    @pytest.mark.asyncio
    async def test_multiple_validation_errors_logged(self):
        """Test that multiple errors are logged."""
        validator = Validator()
        validator.log_validation_error("name", "x", "Too short")
        validator.log_validation_error("age", -5, "Must be positive")

        assert len(validator.validation_log) == 2


class TestRuleRegistry:
    """Test RuleRegistry for type→Rule mapping."""

    @pytest.mark.asyncio
    async def test_registry_type_registration(self):
        """Test that rules are registered for types."""
        registry = RuleRegistry()
        registry.register(str, StringRule())
        registry.register(int, NumberRule())

        assert registry.has_rule(str)
        assert registry.has_rule(int)

    @pytest.mark.asyncio
    async def test_registry_get_rule(self):
        """Test getting rule for type."""
        registry = RuleRegistry()
        string_rule = StringRule()
        registry.register(str, string_rule)

        assert registry.get_rule(base_type=str) is string_rule

    @pytest.mark.asyncio
    async def test_registry_field_name_priority(self):
        """Test field name takes priority over type."""
        registry = RuleRegistry()
        type_rule = StringRule()
        name_rule = StringRule(min_length=5)

        registry.register(str, type_rule)
        registry.register("special", name_rule)

        # Field name should take priority
        rule = registry.get_rule(base_type=str, field_name="special")
        assert rule is name_rule


class TestStrictMode:
    """Test strict parameter in validate_field."""

    @pytest.mark.asyncio
    async def test_strict_true_raises_on_no_rule(self):
        """Test that strict=True raises ValidationError when no rule applies."""
        validator = Validator()

        # Use a type that no rule applies to
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_field("unknown", [], list, strict=True)

        assert "no rule found" in str(exc_info.value).lower()
        assert len(validator.validation_log) == 1

    @pytest.mark.asyncio
    async def test_strict_false_returns_value_on_no_rule(self):
        """Test that strict=False returns value when no rule applies."""
        validator = Validator()

        # Use a type that no rule applies to
        result = await validator.validate_field("unknown", [1, 2, 3], list, strict=False)

        # Should return the original value
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_strict_true_default_in_validate_field(self):
        """Test that strict=True is the default."""
        validator = Validator()

        # Should raise by default (strict=True)
        with pytest.raises(ValidationError):
            await validator.validate_field("unknown", [], list)

    @pytest.mark.asyncio
    async def test_strict_false_doesnt_log_error(self):
        """Test that strict=False doesn't log error when no rule applies."""
        validator = Validator()

        result = await validator.validate_field("unknown", [], list, strict=False)

        # No error should be logged
        assert len(validator.validation_log) == 0
        assert result == []

    @pytest.mark.asyncio
    async def test_strict_mode_with_validation_failure(self):
        """Test that strict mode still raises on validation failure."""
        # Configure custom registry with StringRule with min_length=1
        registry = RuleRegistry()
        registry.register(str, StringRule(min_length=1))
        validator = Validator(registry=registry)

        # String rule should apply but validation should fail due to min_length
        with pytest.raises(ValidationError):
            await validator.validate_field(
                "name",
                "",  # Empty string fails min_length=1 even after auto-fix
                str,
                strict=True,
            )


class TestLogDuringValidation:
    """Test that errors are logged during validation failures."""

    @pytest.mark.asyncio
    async def test_error_logged_on_validation_failure(self):
        """Test that validation errors are logged automatically."""
        # Configure custom registry with StringRule with min_length=1
        registry = RuleRegistry()
        registry.register(str, StringRule(min_length=1))
        validator = Validator(registry=registry)

        try:
            await validator.validate_field(
                "name",
                "",  # Empty string fails min_length=1 even after auto-fix
                str,
                strict=True,
            )
        except ValidationError:
            pass

        # Error should be logged
        assert len(validator.validation_log) >= 1

    @pytest.mark.asyncio
    async def test_error_logged_on_strict_failure(self):
        """Test that strict mode errors are logged."""
        validator = Validator()

        try:
            await validator.validate_field("unknown", [], list, strict=True)
        except ValidationError:
            pass

        # Error should be logged
        assert len(validator.validation_log) == 1
        log_entry = validator.validation_log[0]
        assert log_entry["field"] == "unknown"
        assert "no rule found" in log_entry["error"].lower()


class TestValidatorIntegration:
    """Integration tests for enhanced Validator."""

    @pytest.mark.asyncio
    async def test_enhanced_validator_with_all_features(self):
        """Test validator with registry, strict mode, and logging."""
        validator = Validator()

        # Valid string should pass
        result = await validator.validate_field("name", "Ocean", str, strict=True)
        assert result == "Ocean"

        # Invalid type with strict=False should return as-is
        result = await validator.validate_field("data", [], list, strict=False)
        assert result == []

        # list type has no rule - strict=True should raise
        with pytest.raises(ValidationError):
            await validator.validate_field("data", [], list, strict=True)

        # Check that errors were logged
        assert len(validator.validation_log) >= 1

    @pytest.mark.asyncio
    async def test_clear_validation_log(self):
        """Test clearing validation log manually."""
        validator = Validator()
        validator.log_validation_error("field1", "value1", "error1")
        validator.log_validation_error("field2", "value2", "error2")

        assert len(validator.validation_log) == 2

        validator.validation_log.clear()
        assert len(validator.validation_log) == 0

    @pytest.mark.asyncio
    async def test_get_validation_summary(self):
        """Test getting summary of validation errors."""
        validator = Validator()

        # Log some errors
        validator.log_validation_error("field1", "value1", "error1")
        validator.log_validation_error("field2", "value2", "error2")

        # Get summary
        summary = validator.get_validation_summary()

        assert isinstance(summary, dict)
        assert "total_errors" in summary
        assert summary["total_errors"] == 2
        assert "fields_with_errors" in summary
        assert set(summary["fields_with_errors"]) == {"field1", "field2"}
