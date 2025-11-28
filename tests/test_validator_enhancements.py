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


class TestValidatorIntegration:
    """Integration tests for enhanced Validator."""

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


class TestValidatorEdgeCases:
    """Tests for validator edge cases - covers uncovered lines."""

    @pytest.mark.asyncio
    async def test_none_value_strict_false_no_default(self):
        """Test None value with strict=False and no default returns None (line 144)."""
        from lionpride.types import Operable, Spec

        # Non-nullable spec with no default
        spec = Spec(str, name="required_field")
        operable = Operable([spec])

        validator = Validator()
        # With strict=False, should return None instead of raising
        validated = await validator.validate_operable(
            data={"required_field": None},
            operable=operable,
            strict=False,
        )

        assert validated["required_field"] is None

    @pytest.mark.asyncio
    async def test_listable_spec_rule_exception(self):
        """Test exception in rule.invoke for listable items (lines 167-169)."""
        from lionpride.types import Operable, Spec

        # Use a number rule with strict constraints that will fail
        failing_rule = NumberRule(ge=0.0, le=1.0)
        spec = Spec(float, name="scores", listable=True, rule=failing_rule)
        operable = Operable([spec])

        validator = Validator()
        # Value 5.0 is outside [0, 1] range - should raise and log error
        with pytest.raises(ValidationError):
            await validator.validate_operable(
                data={"scores": [0.5, 5.0]},  # 5.0 will fail
                operable=operable,
                auto_fix=True,
            )

        # Error should be logged
        assert len(validator.validation_log) == 1
        assert "scores[1]" in validator.validation_log[0]["field"]

    @pytest.mark.asyncio
    async def test_listable_spec_no_rule(self):
        """Test listable spec with no rule passes items through (lines 170-171)."""
        from lionpride.types import Operable, Spec

        # Create a custom type with no rule in default registry
        class CustomType:
            def __init__(self, value):
                self.value = value

        spec = Spec(CustomType, name="items", listable=True)
        operable = Operable([spec])

        # Use empty registry so no rule is found
        empty_registry = RuleRegistry()
        validator = Validator(registry=empty_registry)

        item1 = CustomType("a")
        item2 = CustomType("b")

        # With strict=False, items should pass through unchanged
        validated = await validator.validate_operable(
            data={"items": [item1, item2]},
            operable=operable,
            strict=False,
        )

        assert validated["items"] == [item1, item2]

    @pytest.mark.asyncio
    async def test_validator_list_with_non_callable(self):
        """Test validator list with non-callable is skipped (line 204)."""
        from unittest.mock import MagicMock

        from lionpride.types import Operable, Spec

        def uppercase_validator(v):
            return v.upper()

        # Create a valid spec first
        spec = Spec(str, name="code", validator=uppercase_validator)

        # Mock spec.get to return a list with non-callable items
        original_get = spec.get

        def mock_get(key, default=None):
            if key == "validator":
                # Return list with callable and non-callable items
                return [uppercase_validator, "not_callable", 42]
            return original_get(key, default)

        # Apply mock
        mock_spec = MagicMock(wraps=spec)
        mock_spec.get = mock_get
        mock_spec.name = "code"
        mock_spec.base_type = str
        mock_spec.is_nullable = False
        mock_spec.is_listable = False

        async def mock_acreate_default_value():
            raise ValueError("No default")

        mock_spec.acreate_default_value = mock_acreate_default_value

        # Create operable with actual spec (for model creation)
        Operable([spec])  # Ensure model is created

        validator = Validator()
        # Directly call validate_spec with our mock
        result = await validator.validate_spec(mock_spec, "abc", auto_fix=True, strict=True)

        # The callable validator should still work, non-callables skipped
        assert result == "ABC"

    @pytest.mark.asyncio
    async def test_custom_validator_exception(self):
        """Test exception in custom validator (lines 213-216)."""
        from lionpride.types import Operable, Spec

        def failing_validator(v):
            raise ValueError("Validator intentionally failed")

        spec = Spec(str, name="field", validator=failing_validator)
        operable = Operable([spec])

        validator = Validator()
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_operable(
                data={"field": "test"},
                operable=operable,
            )

        assert "Custom validator failed" in str(exc_info.value)
        assert "Validator intentionally failed" in str(exc_info.value)

        # Error should be logged
        assert len(validator.validation_log) == 1
        assert validator.validation_log[0]["field"] == "field"

    @pytest.mark.asyncio
    async def test_validate_with_capabilities_filter(self):
        """Test _validate_data skips fields not in capabilities (line 255)."""
        from lionpride.types import Operable, Spec

        spec1 = Spec(str, name="included")
        spec2 = Spec(str, name="excluded")
        operable = Operable([spec1, spec2])

        validator = Validator()

        # Use validate() method with limited capabilities
        result = await validator.validate(
            data={"included": "yes", "excluded": "no"},
            operable=operable,
            capabilities={"included"},  # Only include "included"
        )

        # Only "included" should be in the result
        assert hasattr(result, "included")
        assert result.included == "yes"
        # "excluded" should not be validated/included

    @pytest.mark.asyncio
    async def test_spec_without_name_skipped(self):
        """Test spec without name is skipped in _validate_data (line 251)."""
        from unittest.mock import MagicMock

        from lionpride.types import Operable, Spec

        # Create a real operable for model creation
        spec_with_name = Spec(str, name="named_field")
        Operable([spec_with_name])  # Ensure model is created

        # Create mock operable with a spec that has no name
        mock_spec_no_name = MagicMock()
        mock_spec_no_name.name = None  # No name
        mock_spec_no_name.base_type = str

        mock_operable = MagicMock()
        mock_operable.get_specs.return_value = [mock_spec_no_name, spec_with_name]
        mock_operable.allowed.return_value = {"named_field"}

        validator = Validator()
        # Call _validate_data directly with mock operable
        validated = await validator._validate_data(
            data={"named_field": "value"},
            operable=mock_operable,
            capabilities={"named_field"},
        )

        # Only the named field should be in results
        assert validated["named_field"] == "value"

    @pytest.mark.asyncio
    async def test_no_rule_strict_mode_single_value(self):
        """Test no rule found for single value in strict mode (lines 178-184)."""
        from lionpride.types import Operable, Spec

        # Create a custom type with no rule in registry
        class UnknownType:
            pass

        spec = Spec(UnknownType, name="unknown")
        operable = Operable([spec])

        # Use empty registry
        empty_registry = RuleRegistry()
        validator = Validator(registry=empty_registry)

        # With strict=True (default), should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_operable(
                data={"unknown": UnknownType()},
                operable=operable,
                strict=True,
            )

        assert "No rule found" in str(exc_info.value)
        assert "unknown" in str(exc_info.value)

        # Error should be logged
        assert len(validator.validation_log) == 1
        assert validator.validation_log[0]["field"] == "unknown"

    def test_validator_repr(self):
        """Test Validator.__repr__ (lines 339-340)."""
        validator = Validator()

        repr_str = repr(validator)

        assert "Validator" in repr_str
        assert "registry_types" in repr_str
        # Default registry has str, int, float, bool, dict, etc.
        assert "str" in repr_str

    def test_validator_repr_custom_registry(self):
        """Test Validator.__repr__ with custom registry."""
        registry = RuleRegistry()
        registry.register(str, StringRule())

        validator = Validator(registry=registry)
        repr_str = repr(validator)

        assert "Validator" in repr_str
        assert "str" in repr_str


class TestAsyncValidatorEdgeCases:
    """Tests for async validator edge cases."""

    @pytest.mark.asyncio
    async def test_async_custom_validator_exception(self):
        """Test exception in async custom validator."""
        from lionpride.types import Operable, Spec

        async def failing_async_validator(v):
            raise RuntimeError("Async validator failed")

        spec = Spec(str, name="async_field", validator=failing_async_validator)
        operable = Operable([spec])

        validator = Validator()
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_operable(
                data={"async_field": "test"},
                operable=operable,
            )

        assert "Custom validator failed" in str(exc_info.value)
        assert "Async validator failed" in str(exc_info.value)
