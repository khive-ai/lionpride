# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Rule

__all__ = ("RuleRegistry", "get_default_registry")

# Lazy initialization to avoid circular imports
_default_registry: RuleRegistry | None = None


class RuleRegistry:
    """Registry mapping types to Rule classes/instances.

    Provides automatic Rule assignment based on Spec.base_type.
    Supports inheritance-based lookup (subclasses inherit parent rules).

    Usage:
        registry = RuleRegistry()
        registry.register(str, StringRule(min_length=1))
        registry.register(int, NumberRule(ge=0))

        rule = registry.get_rule(str)  # Returns StringRule instance
        rule = registry.get_rule(MyStr)  # Returns StringRule (inherited)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._type_rules: dict[type, Rule] = {}
        self._name_rules: dict[str, Rule] = {}

    def register(
        self,
        key: type | str,
        rule: Rule,
        *,
        replace: bool = False,
    ) -> None:
        """Register a Rule for a type or field name.

        Args:
            key: Type or field name to register
            rule: Rule instance to use
            replace: Allow replacing existing registration

        Raises:
            ValueError: If key already registered and replace=False
        """
        if isinstance(key, str):
            if key in self._name_rules and not replace:
                raise ValueError(f"Rule already registered for field '{key}'")
            self._name_rules[key] = rule
        else:
            if key in self._type_rules and not replace:
                raise ValueError(f"Rule already registered for type {key}")
            self._type_rules[key] = rule

    def get_rule(
        self,
        base_type: type | None = None,
        field_name: str | None = None,
    ) -> Rule | None:
        """Get Rule for a type or field name.

        Priority:
        1. Exact field name match
        2. Exact type match
        3. Inheritance-based type match (check base classes)

        Args:
            base_type: Type to look up
            field_name: Field name to look up

        Returns:
            Rule instance or None if not found
        """
        # Priority 1: Exact field name match
        if field_name and field_name in self._name_rules:
            return self._name_rules[field_name]

        # Priority 2: Exact type match
        if base_type and base_type in self._type_rules:
            return self._type_rules[base_type]

        # Priority 3: Inheritance-based lookup
        if base_type:
            for registered_type, rule in self._type_rules.items():
                try:
                    if issubclass(base_type, registered_type):
                        return rule
                except TypeError:
                    # base_type might not be a class (e.g., generic)
                    continue

        return None

    def has_rule(self, key: type | str) -> bool:
        """Check if a rule is registered for type or name."""
        if isinstance(key, str):
            return key in self._name_rules
        return key in self._type_rules or any(
            issubclass(key, t) for t in self._type_rules if isinstance(key, type)
        )

    def list_types(self) -> list[type]:
        """List all registered types."""
        return list(self._type_rules.keys())

    def list_names(self) -> list[str]:
        """List all registered field names."""
        return list(self._name_rules.keys())


def get_default_registry() -> RuleRegistry:
    """Get the default rule registry with standard rules.

    Standard mappings:
        str → StringRule
        int → NumberRule
        float → NumberRule
        bool → BooleanRule
        dict → MappingRule
        ActionRequest → ActionRequestRule
        Reason → ReasonRule
        BaseModel → BaseModelRule (catches all Pydantic models)

    Returns:
        RuleRegistry with default rules registered
    """
    global _default_registry

    if _default_registry is None:
        from pydantic import BaseModel

        from .action_request import ActionRequestRule
        from .basemodel import BaseModelRule
        from .boolean import BooleanRule
        from .mapping import MappingRule
        from .models import ActionRequest, Reason
        from .number import NumberRule
        from .reason import ReasonRule
        from .string import StringRule

        _default_registry = RuleRegistry()
        _default_registry.register(str, StringRule())
        _default_registry.register(int, NumberRule())
        _default_registry.register(float, NumberRule())
        _default_registry.register(bool, BooleanRule())
        _default_registry.register(dict, MappingRule())
        _default_registry.register(ActionRequest, ActionRequestRule())
        _default_registry.register(Reason, ReasonRule())
        _default_registry.register(BaseModel, BaseModelRule())

    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (for testing)."""
    global _default_registry
    _default_registry = None
