# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Validator engine - applies rules to validate and fix data structures."""

from typing import Any

from .base import Rule
from .boolean import BooleanRule
from .choice import ChoiceRule
from .number import NumberRule
from .string import StringRule

__all__ = ("Validator",)


class Validator:
    """Validation engine using rule system.

    Applies rules to data based on Operable specs.
    This is the core of IPU's validation → structure → usefulness pipeline.

    Usage:
        validator = Validator()
        validated = await validator.validate(
            data={"confidence": "0.95", "output": 42},
            spec=operable,
            auto_fix=True
        )
    """

    def __init__(self, rules: dict[str, Rule] | None = None):
        """Initialize validator with rules.

        Args:
            rules: Custom rules dict (uses defaults if None)
        """
        self.rules = rules or self._get_default_rules()

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
    ) -> Any:
        """Validate single field using applicable rules.

        Args:
            field_name: Field name
            field_value: Field value
            field_type: Expected type (optional)
            auto_fix: Enable auto-correction

        Returns:
            Validated (and possibly fixed) value

        Raises:
            ValidationError: If validation fails and auto_fix disabled
        """
        # Find first applicable rule
        for _rule_name, rule in self.rules.items():
            if await rule.apply(field_name, field_value, field_type):
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

        # No rule applied - return value as-is
        return field_value

    async def validate(
        self,
        data: dict[str, Any],
        operable: Any = None,  # Operable type
        auto_fix: bool = True,
    ) -> dict[str, Any]:
        """Validate data structure using rules against Operable spec.

        Args:
            data: Field → value dict to validate
            operable: Operable spec defining expected structure (optional)
            auto_fix: Enable auto-correction

        Returns:
            Validated (and possibly fixed) data

        Raises:
            ValidationError: If validation fails and auto_fix disabled
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
                    field_name, value, field_type, auto_fix
                )
        else:
            # No spec - validate all fields without type info
            for field_name, value in data.items():
                validated[field_name] = await self.validate_field(field_name, value, None, auto_fix)

        return validated

    def __repr__(self) -> str:
        """String representation."""
        return f"Validator(rules={list(self.rules.keys())})"
