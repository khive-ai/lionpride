# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Validator enhancements from lionagi v0.2.2."""

import pytest

from lionpride.rules import ValidationError, Validator
from lionpride.rules.base import RuleParams
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


class TestRuleOrder:
    """Test rule_order parameter for rule precedence."""

    @pytest.mark.asyncio
    async def test_rule_order_initialized(self):
        """Test that rule_order is stored in validator."""
        order = ["number", "string"]
        validator = Validator(rule_order=order)
        assert hasattr(validator, "rule_order")
        assert validator.rule_order == order

    @pytest.mark.asyncio
    async def test_rule_order_default(self):
        """Test default rule_order when not specified."""
        validator = Validator()
        assert hasattr(validator, "rule_order")
        assert isinstance(validator.rule_order, list)
        # Default order should be list of rule names
        assert len(validator.rule_order) > 0

    @pytest.mark.asyncio
    async def test_rules_applied_in_order(self):
        """Test that rules are applied in specified order."""
        # Create a custom rule that tracks calls
        call_order = []

        class TrackingStringRule(StringRule):
            async def validate(self, v, t, **kw):
                call_order.append("string")
                await super().validate(v, t, **kw)

        class TrackingNumberRule(NumberRule):
            async def validate(self, v, t, **kw):
                call_order.append("number")
                await super().validate(v, t, **kw)

        # Create validator with specific order
        validator = Validator(
            rule_order=["number", "string"],
            rules={
                "number": TrackingNumberRule(params=RuleParams(apply_types={int, float})),
                "string": TrackingStringRule(params=RuleParams(apply_types={str})),
            },
        )

        # Clear and validate a string - should check in order: number, string
        call_order.clear()
        await validator.validate_field("test", "hello", str)
        # String rule applies, so only string should be in order
        assert "string" in call_order


class TestStrictMode:
    """Test strict parameter in validate_field."""

    @pytest.mark.asyncio
    async def test_strict_true_raises_on_no_rule(self):
        """Test that strict=True raises ValidationError when no rule applies."""
        validator = Validator()

        # Use a type that no rule applies to
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_field("unknown", [], list, strict=True)

        assert "no rule applied" in str(exc_info.value).lower()
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
        # Configure StringRule with min_length=1 to reject empty strings
        # Even with auto_fix=True (default), re-validation after fix catches the error
        validator = Validator(rules={"string": StringRule(min_length=1)})

        # String rule should apply but validation should fail due to min_length
        # (empty string -> fixed to empty string -> re-validation fails)
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
        # Configure StringRule with min_length=1 to reject empty strings
        # Re-validation after auto-fix catches the error and logs it
        validator = Validator(rules={"string": StringRule(min_length=1)})

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
            await validator.validate_field("unknown", {}, dict, strict=True)
        except ValidationError:
            pass

        # Error should be logged
        assert len(validator.validation_log) == 1
        log_entry = validator.validation_log[0]
        assert log_entry["field"] == "unknown"
        assert "no rule applied" in log_entry["error"].lower()


class TestValidatorIntegration:
    """Integration tests for enhanced Validator."""

    @pytest.mark.asyncio
    async def test_enhanced_validator_with_all_features(self):
        """Test validator with rule_order, strict mode, and logging."""
        order = ["string", "number"]
        validator = Validator(rule_order=order)

        # Valid string should pass
        result = await validator.validate_field("name", "Ocean", str, strict=True)
        assert result == "Ocean"

        # Invalid type with strict=False should return as-is
        result = await validator.validate_field("data", [], list, strict=False)
        assert result == []

        # Invalid type with strict=True should raise
        with pytest.raises(ValidationError):
            await validator.validate_field("data", {}, dict, strict=True)

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
