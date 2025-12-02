# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import Rule, ValidationError
from .registry import RuleRegistry, get_default_registry

if TYPE_CHECKING:
    from lionpride.types import Operable, Spec

__all__ = ("Validator",)


class Validator:
    """Validates data spec-by-spec using auto-assigned Rules from Spec.base_type."""

    def __init__(
        self,
        registry: RuleRegistry | None = None,
    ):
        """Initialize validator with rule registry.

        Args:
            registry: RuleRegistry for type→Rule lookup (uses default if None)
        """
        self.registry = registry or get_default_registry()
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

    async def _validate_data(
        self,
        data: dict[str, Any],
        operable: Operable,
        capabilities: set[str],
        auto_fix: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Validate data spec-by-spec against an Operable.

        Only validates fields that are in the capabilities set.
        Empty capabilities = nothing validated.

        Args:
            data: Raw data dict (e.g., from LLM response)
            operable: Operable defining expected structure
            capabilities: Set of field names allowed to be validated
            auto_fix: Enable auto-correction for each field
            strict: Raise if validation fails

        Returns:
            Dict of validated field values (only fields in capabilities)

        Raises:
            ValidationError: If any field validation fails
        """
        validated: dict[str, Any] = {}

        for spec in operable.get_specs():
            field_name = spec.name
            if field_name is None:
                continue

            # Only validate fields in capabilities
            if field_name not in capabilities:
                continue

            # Get value from data
            value = data.get(field_name)

            # Validate against spec
            validated[field_name] = await self.validate_spec(
                spec, value, auto_fix=auto_fix, strict=strict
            )

        return validated

    async def validate_operable(
        self,
        data: dict[str, Any],
        operable: Operable,
        auto_fix: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Validate data spec-by-spec against an Operable.

        Validates all fields defined in the operable. This is a convenience
        wrapper around _validate_data that validates all fields.

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
        # Validate all fields in the operable
        capabilities = operable.allowed()
        return await self._validate_data(
            data, operable, capabilities, auto_fix=auto_fix, strict=strict
        )

    async def validate(
        self,
        data: dict[str, Any],
        operable: Operable,
        capabilities: set[str],
        auto_fix: bool = True,
        strict: bool = True,
    ) -> Any:
        """Validate data and return a model instance.

        This is the security microkernel - all capability-based access control
        flows through these few lines.

        Flow:
        1. Validate data field-by-field with rules (respects capabilities)
        2. Create model from operable with allowed capabilities
        3. Validate dict into model instance

        Args:
            data: Raw data dict (e.g., from LLM response)
            operable: Operable defining expected structure (carries its own adapter)
            capabilities: Set of field names allowed
            auto_fix: Enable auto-correction for each field
            strict: Raise if validation fails

        Returns:
            Validated model instance

        Raises:
            ValidationError: If validation fails
        """
        # 1. Validate data with rules
        validated_dict = await self._validate_data(
            data, operable, capabilities, auto_fix=auto_fix, strict=strict
        )

        # 2. Create model with capabilities (operable uses its own adapter)
        Model = operable.create_model(include=capabilities)
        return operable.adapter.validate_model(Model, validated_dict)

    def __repr__(self) -> str:
        """String representation."""
        types = self.registry.list_types()
        return f"Validator(registry_types={[t.__name__ for t in types]})"
