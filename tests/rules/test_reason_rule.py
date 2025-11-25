# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for ReasonRule coverage."""

import pytest

from lionpride.rules import Reason
from lionpride.rules.reason import ReasonRule, _get_reason_params


class TestGetReasonParams:
    """Test _get_reason_params function."""

    def test_returns_rule_params(self):
        """Test that _get_reason_params returns RuleParams with correct values."""
        params = _get_reason_params()

        assert params.apply_types == {Reason}
        assert params.apply_fields == set()
        assert params.auto_fix is True


class TestReasonRuleInit:
    """Test ReasonRule initialization."""

    def test_init_with_default_params(self):
        """Test initialization without params uses defaults."""
        rule = ReasonRule()

        assert rule.params.apply_types == {Reason}
        assert rule.params.auto_fix is True

    def test_init_with_custom_params(self):
        """Test initialization with custom params."""
        from lionpride.rules.base import RuleParams

        custom_params = RuleParams(
            apply_types={Reason},
            apply_fields={"custom_field"},
            auto_fix=False,
        )
        rule = ReasonRule(params=custom_params)

        assert rule.params.apply_fields == {"custom_field"}
        assert rule.params.auto_fix is False


class TestReasonRuleValidate:
    """Test ReasonRule.validate method."""

    @pytest.mark.asyncio
    async def test_validate_reason_instance_passes(self):
        """Test that Reason instance passes validation."""
        rule = ReasonRule()
        reason = Reason(reasoning="test reasoning")

        # Should not raise
        await rule.validate(reason, Reason)

    @pytest.mark.asyncio
    async def test_validate_non_reason_raises(self):
        """Test that non-Reason instance raises ValueError."""
        rule = ReasonRule()

        with pytest.raises(ValueError, match="Expected Reason instance"):
            await rule.validate("not a reason", Reason)

    @pytest.mark.asyncio
    async def test_validate_dict_raises(self):
        """Test that dict raises ValueError (needs perform_fix)."""
        rule = ReasonRule()

        with pytest.raises(ValueError, match="Expected Reason instance"):
            await rule.validate({"reasoning": "test"}, Reason)

    @pytest.mark.asyncio
    async def test_validate_confidence_in_range(self):
        """Test that valid confidence passes."""
        rule = ReasonRule()
        reason = Reason(reasoning="test", confidence=0.5)

        # Should not raise
        await rule.validate(reason, Reason)

    @pytest.mark.asyncio
    async def test_validate_confidence_at_boundaries(self):
        """Test confidence at 0.0 and 1.0 boundaries."""
        rule = ReasonRule()

        # Test 0.0
        reason_zero = Reason(reasoning="test", confidence=0.0)
        await rule.validate(reason_zero, Reason)

        # Test 1.0
        reason_one = Reason(reasoning="test", confidence=1.0)
        await rule.validate(reason_one, Reason)

    @pytest.mark.asyncio
    async def test_validate_confidence_none_passes(self):
        """Test that None confidence passes validation."""
        rule = ReasonRule()
        reason = Reason(reasoning="test", confidence=None)

        # Should not raise
        await rule.validate(reason, Reason)


class TestReasonRulePerformFix:
    """Test ReasonRule.perform_fix method."""

    @pytest.mark.asyncio
    async def test_perform_fix_reason_instance_passthrough(self):
        """Test that Reason instance passes through unchanged."""
        rule = ReasonRule()
        original = Reason(reasoning="test", confidence=0.8)

        result = await rule.perform_fix(original, Reason)

        assert result is original

    @pytest.mark.asyncio
    async def test_perform_fix_string_to_reason(self):
        """Test string is converted to Reason."""
        rule = ReasonRule()

        result = await rule.perform_fix("my reasoning text", Reason)

        assert isinstance(result, Reason)
        assert result.reasoning == "my reasoning text"
        assert result.confidence is None

    @pytest.mark.asyncio
    async def test_perform_fix_dict_with_reasoning(self):
        """Test dict with reasoning field is converted."""
        rule = ReasonRule()

        result = await rule.perform_fix({"reasoning": "from dict"}, Reason)

        assert isinstance(result, Reason)
        assert result.reasoning == "from dict"
        assert result.confidence is None

    @pytest.mark.asyncio
    async def test_perform_fix_dict_with_reasoning_and_confidence(self):
        """Test dict with reasoning and confidence fields."""
        rule = ReasonRule()

        result = await rule.perform_fix(
            {"reasoning": "test", "confidence": 0.75},
            Reason,
        )

        assert isinstance(result, Reason)
        assert result.reasoning == "test"
        assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_perform_fix_dict_empty_reasoning(self):
        """Test dict without reasoning field uses empty string."""
        rule = ReasonRule()

        result = await rule.perform_fix({}, Reason)

        assert isinstance(result, Reason)
        assert result.reasoning == ""

    @pytest.mark.asyncio
    async def test_perform_fix_dict_non_string_reasoning(self):
        """Test dict with non-string reasoning is converted to string."""
        rule = ReasonRule()

        result = await rule.perform_fix({"reasoning": 123}, Reason)

        assert isinstance(result, Reason)
        assert result.reasoning == "123"

    @pytest.mark.asyncio
    async def test_perform_fix_dict_none_reasoning(self):
        """Test dict with None reasoning uses empty string."""
        rule = ReasonRule()

        result = await rule.perform_fix({"reasoning": None}, Reason)

        assert isinstance(result, Reason)
        assert result.reasoning == ""

    @pytest.mark.asyncio
    async def test_perform_fix_dict_string_confidence(self):
        """Test dict with string confidence is converted to float."""
        rule = ReasonRule()

        result = await rule.perform_fix(
            {"reasoning": "test", "confidence": "0.5"},
            Reason,
        )

        assert isinstance(result, Reason)
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_perform_fix_dict_invalid_confidence_type(self):
        """Test dict with invalid confidence type results in None."""
        rule = ReasonRule()

        result = await rule.perform_fix(
            {"reasoning": "test", "confidence": "not a number"},
            Reason,
        )

        assert isinstance(result, Reason)
        assert result.confidence is None

    @pytest.mark.asyncio
    async def test_perform_fix_dict_confidence_out_of_range(self):
        """Test dict with out-of-range confidence results in None."""
        rule = ReasonRule()

        result = await rule.perform_fix(
            {"reasoning": "test", "confidence": 1.5},
            Reason,
        )

        assert isinstance(result, Reason)
        assert result.confidence is None

    @pytest.mark.asyncio
    async def test_perform_fix_dict_negative_confidence(self):
        """Test dict with negative confidence results in None."""
        rule = ReasonRule()

        result = await rule.perform_fix(
            {"reasoning": "test", "confidence": -0.5},
            Reason,
        )

        assert isinstance(result, Reason)
        assert result.confidence is None

    @pytest.mark.asyncio
    async def test_perform_fix_fallback_to_string(self):
        """Test fallback converts arbitrary types to string."""
        rule = ReasonRule()

        result = await rule.perform_fix(42, Reason)

        assert isinstance(result, Reason)
        assert result.reasoning == "42"

    @pytest.mark.asyncio
    async def test_perform_fix_fallback_with_list(self):
        """Test fallback converts list to string."""
        rule = ReasonRule()

        result = await rule.perform_fix([1, 2, 3], Reason)

        assert isinstance(result, Reason)
        assert result.reasoning == "[1, 2, 3]"

    @pytest.mark.asyncio
    async def test_perform_fix_conversion_failure(self):
        """Test conversion failure raises ValueError."""
        rule = ReasonRule()

        # Create an object that fails str() conversion
        class Unconvertible:
            def __str__(self):
                raise RuntimeError("Cannot convert")

        with pytest.raises(ValueError, match="Cannot convert Unconvertible to Reason"):
            await rule.perform_fix(Unconvertible(), Reason)
