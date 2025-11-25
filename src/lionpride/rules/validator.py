# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Validator engine - applies rules to validate and fix data structures.

Core of the IPU validation pipeline:
    Spec.base_type → auto Rule assignment → validate spec-by-spec → Operable.create_model()

Features:
- Auto Rule assignment from Spec.base_type via RuleRegistry
- Spec metadata override for custom rules
- validation_log for error tracking
- strict mode control
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import Rule, ValidationError
from .registry import RuleRegistry, get_default_registry

if TYPE_CHECKING:
    from lionpride.types import Operable, Spec

__all__ = ("Validator",)


class Validator:
    """Validation engine using rule system.

    Validates data spec-by-spec using auto-assigned Rules based on Spec.base_type.
    This is the core of IPU's validation → structure → usefulness pipeline.

    Flow:
        1. Operable.create_model(request_specs) → RequestModel for LLM
        2. LLM returns raw response
        3. Validator.validate_operable(raw_data, operable) → validated dict
           - For each Spec: get Rule from base_type → validate → fix
        4. Operable.create_model(output_specs) → OutputModel
        5. OutputModel.model_validate(validated) → final structured output

    Usage:
        validator = Validator()

        # Validate raw LLM response against operable specs
        validated = await validator.validate_operable(
            data={"confidence": "0.95", "output": 42},
            operable=my_operable,
            auto_fix=True
        )

        # Create output model and validate
        OutputModel = operable.create_model()
        result = OutputModel.model_validate(validated)
    """

    def __init__(
        self,
        registry: RuleRegistry | None = None,
        rules: dict[str, Rule] | None = None,
    ):
        """Initialize validator with rule registry.

        Args:
            registry: RuleRegistry for type→Rule lookup (uses default if None)
            rules: Legacy dict-based rules (for backwards compatibility)
        """
        self.registry = registry or get_default_registry()
        # Legacy support: dict-based rules
        self._legacy_rules = rules
        # Validation log for tracking attempts and errors
        self.validation_log: list[dict[str, Any]] = []

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
            Dict with total_errors, fields_with_errors, and error_entries
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

    def get_rule_for_spec(self, spec: Spec) -> Rule | None:
        """Get Rule for a Spec based on base_type or metadata override.

        Priority:
        1. Spec metadata "rule" override (explicit Rule instance)
        2. Registry lookup by field name
        3. Registry lookup by base_type

        Args:
            spec: Spec to get rule for

        Returns:
            Rule instance or None if not found
        """
        # Priority 1: Explicit rule override in metadata
        override = spec.get("rule")
        if override is not None and isinstance(override, Rule):
            return override

        # Priority 2 & 3: Registry lookup (name then type)
        return self.registry.get_rule(
            base_type=spec.base_type,
            field_name=spec.name if spec.name else None,
        )

    async def validate_spec(
        self,
        spec: Spec,
        value: Any,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> Any:
        """Validate a single value against a Spec.

        Handles:
        - nullable: Returns None if value is None and spec is nullable
        - default: Uses sync/async default factory if value is None
        - listable: Validates each item in list against base_type
        - validator: Applies Spec's custom validators after rule validation

        Args:
            spec: Spec defining the field
            value: Value to validate
            auto_fix: Enable auto-correction
            strict: Raise if no rule applies

        Returns:
            Validated (and possibly fixed) value

        Raises:
            ValidationError: If validation fails
        """
        field_name = spec.name or "<unnamed>"

        # Handle nullable/default
        if value is None:
            if spec.is_nullable:
                return None
            # Try default (supports async default factories)
            try:
                value = await spec.acreate_default_value()
            except ValueError:
                if strict:
                    error_msg = f"Field '{field_name}' is None but not nullable and has no default"
                    self.log_validation_error(field_name, value, error_msg)
                    raise ValidationError(error_msg)
                return value

        # Get rule for this spec (priority: metadata override > name > base_type)
        rule = self.get_rule_for_spec(spec)

        # Handle listable specs - validate each item
        if spec.is_listable:
            if not isinstance(value, list):
                if auto_fix:
                    value = [value]  # Wrap single value in list
                else:
                    error_msg = f"Field '{field_name}' expected list, got {type(value).__name__}"
                    self.log_validation_error(field_name, value, error_msg)
                    raise ValidationError(error_msg)

            validated_items = []
            for i, item in enumerate(value):
                item_name = f"{field_name}[{i}]"
                if rule is not None:
                    try:
                        validated_item = await rule.invoke(
                            item_name, item, spec.base_type, auto_fix=auto_fix
                        )
                    except Exception as e:
                        self.log_validation_error(item_name, item, str(e))
                        raise
                else:
                    validated_item = item
                validated_items.append(validated_item)

            value = validated_items
        else:
            # Single value validation
            if rule is None:
                if strict:
                    error_msg = (
                        f"No rule found for field '{field_name}' with type {spec.base_type}. "
                        f"Register a rule or set strict=False."
                    )
                    self.log_validation_error(field_name, value, error_msg)
                    raise ValidationError(error_msg)
            else:
                try:
                    value = await rule.invoke(field_name, value, spec.base_type, auto_fix=auto_fix)
                except Exception as e:
                    self.log_validation_error(field_name, value, str(e))
                    raise

        # Apply Spec's custom validators (after rule validation)
        custom_validators = spec.get("validator")
        # Check for sentinel values (Undefined, Unset) - they're not callable
        if custom_validators is not None and callable(custom_validators):
            validators = [custom_validators]
        elif isinstance(custom_validators, list):
            validators = custom_validators
        else:
            validators = []

        for validator_fn in validators:
            if not callable(validator_fn):
                continue
            try:
                # Support both sync and async validators
                from lionpride.libs.concurrency import is_coro_func

                if is_coro_func(validator_fn):
                    value = await validator_fn(value)
                else:
                    value = validator_fn(value)
            except Exception as e:
                error_msg = f"Custom validator failed for '{field_name}': {e}"
                self.log_validation_error(field_name, value, error_msg)
                raise ValidationError(error_msg) from e

        return value

    async def validate_operable(
        self,
        data: dict[str, Any],
        operable: Operable,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Validate data spec-by-spec against an Operable.

        This is the main validation method for the IPU pipeline.

        Args:
            data: Raw data dict (e.g., from LLM response)
            operable: Operable defining expected structure
            auto_fix: Enable auto-correction for each field
            strict: Raise if validation fails

        Returns:
            Dict of validated field values

        Raises:
            ValidationError: If any field validation fails
        """
        validated: dict[str, Any] = {}

        for spec in operable.get_specs():
            field_name = spec.name
            if field_name is None:
                continue

            # Get value from data
            value = data.get(field_name)

            # Validate against spec
            validated[field_name] = await self.validate_spec(
                spec, value, auto_fix=auto_fix, strict=strict
            )

        return validated

    # === Legacy API (backwards compatibility) ===

    async def validate_field(
        self,
        field_name: str,
        field_value: Any,
        field_type: type | None = None,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> Any:
        """Validate single field using registry lookup.

        Legacy method for backwards compatibility.
        Prefer validate_spec() for new code.

        Args:
            field_name: Field name
            field_value: Field value
            field_type: Expected type (optional)
            auto_fix: Enable auto-correction
            strict: Raise if no rule applies

        Returns:
            Validated (and possibly fixed) value
        """
        # Get rule from registry
        rule = self.registry.get_rule(base_type=field_type, field_name=field_name)

        # Fallback to legacy rules if available
        if rule is None and self._legacy_rules:
            for legacy_rule in self._legacy_rules.values():
                try:
                    if await legacy_rule.apply(field_name, field_value, field_type):
                        rule = legacy_rule
                        break
                except Exception:
                    continue

        if rule is None:
            if strict:
                error_msg = (
                    f"No rule found for field '{field_name}' with type {field_type}. "
                    f"Register a rule or set strict=False."
                )
                self.log_validation_error(field_name, field_value, error_msg)
                raise ValidationError(error_msg)
            return field_value

        try:
            return await rule.invoke(field_name, field_value, field_type, auto_fix=auto_fix)
        except Exception as e:
            self.log_validation_error(field_name, field_value, str(e))
            raise

    async def validate(
        self,
        data: dict[str, Any],
        operable: Operable | None = None,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Validate data structure.

        If operable provided, validates spec-by-spec.
        Otherwise, validates each field using registry lookup.

        Args:
            data: Field → value dict to validate
            operable: Operable spec (optional)
            auto_fix: Enable auto-correction
            strict: Raise if validation fails

        Returns:
            Validated data dict
        """
        if operable is not None:
            return await self.validate_operable(data, operable, auto_fix, strict)

        # No operable - validate each field by inferred type
        validated: dict[str, Any] = {}
        for field_name, value in data.items():
            field_type = type(value) if value is not None else None
            validated[field_name] = await self.validate_field(
                field_name, value, field_type, auto_fix, strict
            )
        return validated

    def __repr__(self) -> str:
        """String representation."""
        types = self.registry.list_types()
        return f"Validator(registry_types={[t.__name__ for t in types]})"
