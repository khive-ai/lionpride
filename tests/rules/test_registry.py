# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuleRegistry coverage.

Target: 100% coverage for src/lionpride/rules/registry.py
Uncovered lines: 58, 62, 98-100, 107, 118, 162
"""

from typing import Generic, TypeVar
from unittest.mock import patch

import pytest

from lionpride.rules import StringRule
from lionpride.rules.registry import (
    RuleRegistry,
    get_default_registry,
    reset_default_registry,
)


class TestRuleRegistryDuplicateRegistration:
    """Test duplicate registration error paths."""

    def test_register_duplicate_field_name_raises(self):
        """Test registering duplicate field name raises ValueError (line 58)."""
        registry = RuleRegistry()
        rule1 = StringRule()
        rule2 = StringRule(min_length=5)

        # Register first rule for field name
        registry.register("username", rule1)

        # Attempt to register another rule for same field name
        with pytest.raises(ValueError, match="Rule already registered for field 'username'"):
            registry.register("username", rule2)

    def test_register_duplicate_field_name_with_replace(self):
        """Test registering duplicate field name with replace=True succeeds."""
        registry = RuleRegistry()
        rule1 = StringRule()
        rule2 = StringRule(min_length=5)

        registry.register("username", rule1)
        registry.register("username", rule2, replace=True)  # Should not raise

        # Verify replacement
        assert registry.get_rule(field_name="username") is rule2

    def test_register_duplicate_type_raises(self):
        """Test registering duplicate type raises ValueError (line 62)."""
        registry = RuleRegistry()
        rule1 = StringRule()
        rule2 = StringRule(min_length=10)

        # Register first rule for type
        registry.register(str, rule1)

        # Attempt to register another rule for same type
        with pytest.raises(ValueError, match="Rule already registered for type"):
            registry.register(str, rule2)

    def test_register_duplicate_type_with_replace(self):
        """Test registering duplicate type with replace=True succeeds."""
        registry = RuleRegistry()
        rule1 = StringRule()
        rule2 = StringRule(min_length=10)

        registry.register(str, rule1)
        registry.register(str, rule2, replace=True)  # Should not raise

        # Verify replacement
        assert registry.get_rule(base_type=str) is rule2


class TestRuleRegistryGetRuleTypeError:
    """Test get_rule handling of TypeError in issubclass (lines 98-100)."""

    def test_get_rule_with_generic_type_handles_type_error(self):
        """Test get_rule handles TypeError when base_type is generic (lines 98-100).

        When base_type is a generic type like List[str], issubclass() raises
        TypeError. The code should catch this and continue to next iteration.
        """
        registry = RuleRegistry()
        rule = StringRule()
        registry.register(str, rule)

        # Generic types cause TypeError in issubclass
        T = TypeVar("T")

        class MyGeneric(Generic[T]):
            pass

        # Parameterized generics raise TypeError with issubclass
        # The registry should handle this gracefully and return None
        result = registry.get_rule(base_type=MyGeneric[int])
        assert result is None

    def test_get_rule_with_union_type_handles_type_error(self):
        """Test get_rule handles TypeError with Union types."""
        from typing import Union

        registry = RuleRegistry()
        rule = StringRule()
        registry.register(str, rule)

        # Union types also cause TypeError in issubclass
        result = registry.get_rule(base_type=Union[str, int])
        assert result is None


class TestRuleRegistryHasRule:
    """Test has_rule with string keys (line 107)."""

    def test_has_rule_with_string_key_true(self):
        """Test has_rule returns True for registered field name (line 107)."""
        registry = RuleRegistry()
        rule = StringRule()
        registry.register("email", rule)

        assert registry.has_rule("email") is True

    def test_has_rule_with_string_key_false(self):
        """Test has_rule returns False for unregistered field name (line 107)."""
        registry = RuleRegistry()

        assert registry.has_rule("nonexistent_field") is False

    def test_has_rule_with_type_key(self):
        """Test has_rule with type key for completeness."""
        registry = RuleRegistry()
        rule = StringRule()
        registry.register(str, rule)

        assert registry.has_rule(str) is True
        assert registry.has_rule(int) is False


class TestRuleRegistryListNames:
    """Test list_names method (line 118)."""

    def test_list_names_empty_registry(self):
        """Test list_names returns empty list for empty registry (line 118)."""
        registry = RuleRegistry()

        assert registry.list_names() == []

    def test_list_names_with_field_rules(self):
        """Test list_names returns registered field names (line 118)."""
        registry = RuleRegistry()
        rule = StringRule()

        registry.register("username", rule)
        registry.register("email", rule)
        registry.register("password", rule)

        names = registry.list_names()
        assert len(names) == 3
        assert set(names) == {"username", "email", "password"}

    def test_list_names_excludes_type_rules(self):
        """Test list_names only returns field names, not types."""
        registry = RuleRegistry()
        rule = StringRule()

        # Register both type and field name rules
        registry.register(str, rule)
        registry.register("username", rule)

        names = registry.list_names()
        assert names == ["username"]


class TestRuleRegistryListTypes:
    """Test list_types method for completeness."""

    def test_list_types_empty_registry(self):
        """Test list_types returns empty list for empty registry."""
        registry = RuleRegistry()

        assert registry.list_types() == []

    def test_list_types_with_type_rules(self):
        """Test list_types returns registered types."""
        registry = RuleRegistry()
        rule = StringRule()

        registry.register(str, rule)
        registry.register(int, rule)

        types = registry.list_types()
        assert len(types) == 2
        assert set(types) == {str, int}


class TestResetDefaultRegistry:
    """Test reset_default_registry function (line 162)."""

    def test_reset_default_registry(self):
        """Test reset_default_registry clears the cached registry (line 162)."""
        # First call creates the default registry
        registry1 = get_default_registry()
        assert registry1 is not None

        # Reset the registry
        reset_default_registry()

        # Next call should create a new registry
        registry2 = get_default_registry()
        assert registry2 is not None

        # They should be different instances
        assert registry1 is not registry2

    def test_reset_default_registry_allows_fresh_initialization(self):
        """Test reset enables fresh default registry with all standard rules."""
        # Get initial registry
        get_default_registry()

        # Reset
        reset_default_registry()

        # Get fresh registry and verify standard rules are present
        fresh = get_default_registry()
        assert fresh.has_rule(str)
        assert fresh.has_rule(int)
        assert fresh.has_rule(float)
        assert fresh.has_rule(bool)
        assert fresh.has_rule(dict)


class TestGetDefaultRegistry:
    """Test get_default_registry function."""

    def test_get_default_registry_caching(self):
        """Test get_default_registry returns same instance on repeated calls."""
        # Reset to ensure clean state
        reset_default_registry()

        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_get_default_registry_has_standard_rules(self):
        """Test default registry includes all standard type rules."""
        reset_default_registry()
        registry = get_default_registry()

        # Verify standard types are registered
        assert registry.get_rule(base_type=str) is not None
        assert registry.get_rule(base_type=int) is not None
        assert registry.get_rule(base_type=float) is not None
        assert registry.get_rule(base_type=bool) is not None
        assert registry.get_rule(base_type=dict) is not None


class TestRuleRegistryGetRulePriority:
    """Test get_rule priority order."""

    def test_get_rule_field_name_priority(self):
        """Test field name has highest priority."""
        registry = RuleRegistry()
        type_rule = StringRule()
        field_rule = StringRule(min_length=5)

        registry.register(str, type_rule)
        registry.register("username", field_rule)

        # Field name should take priority
        result = registry.get_rule(base_type=str, field_name="username")
        assert result is field_rule

    def test_get_rule_type_when_no_field(self):
        """Test type lookup when no field name matches."""
        registry = RuleRegistry()
        rule = StringRule()
        registry.register(str, rule)

        result = registry.get_rule(base_type=str, field_name="nonexistent")
        assert result is rule

    def test_get_rule_inheritance_lookup(self):
        """Test inheritance-based lookup for subclasses."""
        registry = RuleRegistry()
        rule = StringRule()
        registry.register(str, rule)

        # Custom subclass of str
        class MyStr(str):
            pass

        result = registry.get_rule(base_type=MyStr)
        assert result is rule

    def test_get_rule_returns_none_when_not_found(self):
        """Test get_rule returns None when no matching rule."""
        registry = RuleRegistry()

        result = registry.get_rule(base_type=bytes)
        assert result is None
