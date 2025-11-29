# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuleParams and RuleQualifier edge cases.

Note: RuleParams._validate() has a known issue where super()._validate()
fails with frozen dataclass + slots. Lines 107-115 of base.py are currently
unreachable. This file tests valid construction patterns.
"""

import pytest

from lionpride.rules import RuleParams, RuleQualifier


class TestRuleQualifierFromStr:
    """Tests for RuleQualifier.from_str() method."""

    def test_from_str_field(self):
        """Test parsing FIELD qualifier."""
        assert RuleQualifier.from_str("FIELD") == RuleQualifier.FIELD
        assert RuleQualifier.from_str("field") == RuleQualifier.FIELD
        assert RuleQualifier.from_str("  FIELD  ") == RuleQualifier.FIELD

    def test_from_str_annotation(self):
        """Test parsing ANNOTATION qualifier."""
        assert RuleQualifier.from_str("ANNOTATION") == RuleQualifier.ANNOTATION
        assert RuleQualifier.from_str("annotation") == RuleQualifier.ANNOTATION

    def test_from_str_condition(self):
        """Test parsing CONDITION qualifier."""
        assert RuleQualifier.from_str("CONDITION") == RuleQualifier.CONDITION
        assert RuleQualifier.from_str("condition") == RuleQualifier.CONDITION

    def test_from_str_unknown_raises(self):
        """Test unknown qualifier string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown RuleQualifier"):
            RuleQualifier.from_str("unknown")


class TestRuleParamsValidConstruction:
    """Tests for valid RuleParams construction patterns."""

    def test_condition_qualifier_allows_neither_set(self):
        """Test CONDITION qualifier allows neither types nor fields."""
        params = RuleParams(default_qualifier=RuleQualifier.CONDITION)
        assert not params.apply_types
        assert not params.apply_fields
        assert params.default_qualifier == RuleQualifier.CONDITION

    def test_apply_types_only_valid(self):
        """Test apply_types only is valid."""
        params = RuleParams(apply_types={str, int})
        assert str in params.apply_types
        assert int in params.apply_types
        assert not params.apply_fields

    def test_apply_fields_only_valid(self):
        """Test apply_fields only is valid."""
        params = RuleParams(apply_fields={"name", "value"})
        assert "name" in params.apply_fields
        assert "value" in params.apply_fields
        assert not params.apply_types

    def test_auto_fix_default_false(self):
        """Test auto_fix defaults to False."""
        params = RuleParams(apply_types={str})
        assert params.auto_fix is False

    def test_auto_fix_explicit_true(self):
        """Test auto_fix can be set to True."""
        params = RuleParams(apply_types={str}, auto_fix=True)
        assert params.auto_fix is True

    def test_kw_defaults_empty(self):
        """Test kw defaults to empty dict."""
        params = RuleParams(apply_types={str})
        assert params.kw == {}

    def test_kw_can_be_set(self):
        """Test kw can be set with custom values."""
        params = RuleParams(apply_types={str}, kw={"min_length": 1})
        assert params.kw == {"min_length": 1}
