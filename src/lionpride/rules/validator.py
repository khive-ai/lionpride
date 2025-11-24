# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Validator engine - applies rules to validate and fix data structures.

Enhancements from lionagi v0.2.2:
- validation_log: Track validation attempts and errors
- rule_order: Control rule precedence (deterministic iteration)
- log_validation_error(): Log errors with timestamp
- strict mode: Fail or return value if no rule applies
- get_validation_summary(): Summary of validation history
"""

from datetime import datetime
from typing import Any

from .base import Rule, ValidationError
from .boolean import BooleanRule
from .choice import ChoiceRule
from .number import NumberRule
from .string import StringRule

__all__ = ("Validator",)


class Validator:
    """Validation engine using rule system.

    Applies rules to data based on Operable specs.
    This is the core of IPU's validation → structure → usefulness pipeline.

    Enhancements:
    - validation_log: List of validation attempts and errors (for auditing)
    - rule_order: Control order of rule application (deterministic)
    - strict mode: Raise if no rule applies or return value as-is
    - Logging: Track all validation errors with timestamps

    Usage:
        validator = Validator(rule_order=["string", "number"])
        validated = await validator.validate(
            data={"confidence": "0.95", "output": 42},
            spec=operable,
            auto_fix=True,
            strict=True
        )
    """

    def __init__(
        self,
        rules: dict[str, Rule] | None = None,
        rule_order: list[str] | None = None,
    ):
        """Initialize validator with rules and rule order.

        Args:
            rules: Custom rules dict (uses defaults if None)
            rule_order: List of rule names defining application order
                       (if None, uses dict key order)
        """
        self.rules = rules or self._get_default_rules()
        # Set rule order - if not provided, use rule keys in order
        self.rule_order = rule_order or list(self.rules.keys())
        # Validation log for tracking attempts and errors
        self.validation_log: list[dict[str, Any]] = []

    def _get_default_rules(self) -> dict[str, Rule]:
        """Get default validation rules.

        Returns:
            Dict of rule name → Rule instance
        """
        return {
            "string": StringRule(),
            "number": NumberRule(),
            "boolean": BooleanRule(),
            "choice": ChoiceRule(choices=[]),  # Empty choices, subclass for specific
        }

    def log_validation_error(self, field: str, value: Any, error: str) -> None:
        """Log a validation error with timestamp.

        Args:
            field: Field name that failed validation
            value: Value that failed validation
            error: Error message
        """
        log_entry = {
            "field": field,
            "value": value,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        self.validation_log.append(log_entry)

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of validation log.

        Returns:
            Dict with total_errors, fields_with_errors, and error_details
        """
        fields_with_errors = set()
        for entry in self.validation_log:
            if "field" in entry:
                fields_with_errors.add(entry["field"])

        return {
            "total_errors": len(self.validation_log),
            "fields_with_errors": sorted(list(fields_with_errors)),
            "error_entries": self.validation_log,
        }

    def add_rule(self, name: str, rule: Rule, replace: bool = False) -> None:
        """Add custom rule to validator.

        Args:
            name: Rule name
            rule: Rule instance
            replace: Allow replacing existing rule

        Raises:
            ValueError: If rule exists and replace=False
        """
        if name in self.rules and not replace:
            raise ValueError(f"Rule '{name}' already exists (use replace=True)")
        self.rules[name] = rule
        # Add to rule_order if not already there
        if name not in self.rule_order:
            self.rule_order.append(name)

    def remove_rule(self, name: str) -> Rule:
        """Remove rule from validator.

        Args:
            name: Rule name

        Returns:
            Removed rule

        Raises:
            KeyError: If rule doesn't exist
        """
        if name not in self.rules:
            raise KeyError(f"Rule '{name}' not found")
        return self.rules.pop(name)

    async def validate_field(
        self,
        field_name: str,
        field_value: Any,
        field_type: type | None = None,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> Any:
        """Validate single field using applicable rules (in rule_order).

        Args:
            field_name: Field name
            field_value: Field value
            field_type: Expected type (optional)
            auto_fix: Enable auto-correction
            strict: If True, raise ValidationError if no rule applies
                   If False, return value as-is

        Returns:
            Validated (and possibly fixed) value

        Raises:
            ValidationError: If validation fails or (strict=True and no rule applies)
        """
        # Try rules in order specified by rule_order (deterministic)
        rule_applied = False
        last_error = None

        for rule_name in self.rule_order:
            if rule_name not in self.rules:
                continue  # Skip rules not in current rules dict

            rule = self.rules[rule_name]

            try:
                if await rule.apply(field_name, field_value, field_type):
                    rule_applied = True
                    # Apply rule with auto-fix setting
                    if not auto_fix:
                        # Disable auto_fix temporarily
                        original_auto_fix = rule.auto_fix
                        rule.params = rule.params.with_updates(auto_fix=False)
                        try:
                            result = await rule.invoke(field_name, field_value, field_type)
                        finally:
                            # Restore original setting
                            rule.params = rule.params.with_updates(auto_fix=original_auto_fix)
                        return result
                    else:
                        return await rule.invoke(field_name, field_value, field_type)
            except Exception as e:
                # Log rule application error
                last_error = str(e)
                self.log_validation_error(field_name, field_value, str(e))
                # Continue to next rule
                continue

        # No rule applied
        if not rule_applied:
            if strict:
                error_msg = (
                    f"Failed to validate {field_name} because no rule applied. "
                    f"To return the original value directly when no rule applies, "
                    f"set strict=False."
                )
                self.log_validation_error(field_name, field_value, error_msg)
                raise ValidationError(error_msg)
            else:
                # Return value as-is
                return field_value

        # If we get here, all rules failed (no successful validation)
        if last_error and strict:
            raise ValidationError(f"Failed to validate {field_name}: {last_error}")

        return field_value

    async def validate(
        self,
        data: dict[str, Any],
        operable: Any = None,  # Operable type
        auto_fix: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Validate data structure using rules against Operable spec.

        Args:
            data: Field → value dict to validate
            operable: Operable spec defining expected structure (optional)
            auto_fix: Enable auto-correction
            strict: If True, raise ValidationError if no rule applies to a field
                   If False, return field values as-is

        Returns:
            Validated (and possibly fixed) data

        Raises:
            ValidationError: If validation fails and (auto_fix disabled or strict=True)
        """

        validated = {}

        # If Operable provided, validate against its __op_fields__
        if operable is not None and hasattr(operable, "__op_fields__"):
            for field_spec in operable.__op_fields__:
                # Get field name from spec metadata
                field_name = field_spec["name"] if "name" in field_spec.metadata else None
                if field_name is None:
                    continue

                # Get value from data
                value = data.get(field_name)

                # Handle nullable
                nullable = field_spec.get("nullable", False)
                if value is None and nullable:
                    validated[field_name] = None
                    continue

                # Handle default
                if value is None:
                    if "default" in field_spec.metadata:
                        value = field_spec["default"]
                    elif "default_factory" in field_spec.metadata:
                        factory = field_spec["default_factory"]
                        value = factory() if callable(factory) else value

                # Get expected type
                field_type = field_spec.base_type

                # Validate field
                validated[field_name] = await self.validate_field(
                    field_name, value, field_type, auto_fix, strict
                )
        else:
            # No spec - validate all fields without type info
            for field_name, value in data.items():
                validated[field_name] = await self.validate_field(
                    field_name, value, None, auto_fix, strict
                )

        return validated

    def __repr__(self) -> str:
        """String representation."""
        return f"Validator(rules={list(self.rules.keys())})"
