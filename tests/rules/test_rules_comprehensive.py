# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for all rule types in the validation system."""

import pytest

from lionpride.rules import (
    ActionRequestRule,
    BooleanRule,
    ChoiceRule,
    MappingRule,
    NumberRule,
    Rule,
    RuleParams,
    RuleQualifier,
    StringRule,
    ValidationError,
    Validator,
)


class TestStringRule:
    """Tests for StringRule validation."""

    @pytest.mark.asyncio
    async def test_valid_string(self):
        """Test valid string passes validation."""
        rule = StringRule()
        result = await rule.invoke("name", "Ocean", str)
        assert result == "Ocean"

    @pytest.mark.asyncio
    async def test_min_length_valid(self):
        """Test string meeting min_length passes."""
        rule = StringRule(min_length=3)
        result = await rule.invoke("name", "abc", str)
        assert result == "abc"

    @pytest.mark.asyncio
    async def test_min_length_invalid(self):
        """Test string below min_length fails."""
        rule = StringRule(min_length=5)
        with pytest.raises(ValidationError):
            await rule.invoke("name", "abc", str)

    @pytest.mark.asyncio
    async def test_max_length_valid(self):
        """Test string within max_length passes."""
        rule = StringRule(max_length=10)
        result = await rule.invoke("name", "Ocean", str)
        assert result == "Ocean"

    @pytest.mark.asyncio
    async def test_max_length_invalid(self):
        """Test string exceeding max_length fails."""
        rule = StringRule(max_length=3)
        with pytest.raises(ValidationError):
            await rule.invoke("name", "Ocean", str)

    @pytest.mark.asyncio
    async def test_pattern_valid(self):
        """Test string matching pattern passes."""
        rule = StringRule(pattern=r"^[A-Za-z]+$")
        result = await rule.invoke("name", "Ocean", str)
        assert result == "Ocean"

    @pytest.mark.asyncio
    async def test_pattern_invalid(self):
        """Test string not matching pattern fails."""
        rule = StringRule(pattern=r"^[A-Za-z]+$")
        with pytest.raises(ValidationError):
            await rule.invoke("name", "Ocean123", str)

    @pytest.mark.asyncio
    async def test_auto_fix_from_int(self):
        """Test auto-fixing integer to string."""
        rule = StringRule()
        result = await rule.invoke("value", 42, str)
        assert result == "42"

    @pytest.mark.asyncio
    async def test_auto_fix_from_float(self):
        """Test auto-fixing float to string."""
        rule = StringRule()
        result = await rule.invoke("value", 3.14, str)
        assert result == "3.14"

    @pytest.mark.asyncio
    async def test_auto_fix_revalidates_length(self):
        """Test that auto-fix re-validates length constraints."""
        rule = StringRule(min_length=5)
        with pytest.raises(ValidationError):
            await rule.invoke("name", "", str)  # Empty string fails min_length


class TestNumberRule:
    """Tests for NumberRule validation."""

    @pytest.mark.asyncio
    async def test_valid_int(self):
        """Test valid integer passes."""
        rule = NumberRule()
        result = await rule.invoke("count", 42, int)
        assert result == 42

    @pytest.mark.asyncio
    async def test_valid_float(self):
        """Test valid float passes."""
        rule = NumberRule()
        result = await rule.invoke("score", 0.95, float)
        assert result == 0.95

    @pytest.mark.asyncio
    async def test_ge_valid(self):
        """Test number meeting ge constraint passes."""
        rule = NumberRule(ge=0)
        result = await rule.invoke("score", 0.5, float)
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_ge_invalid(self):
        """Test number below ge constraint fails."""
        rule = NumberRule(ge=0)
        with pytest.raises(ValidationError):
            await rule.invoke("score", -0.1, float)

    @pytest.mark.asyncio
    async def test_gt_valid(self):
        """Test number exceeding gt constraint passes."""
        rule = NumberRule(gt=0)
        result = await rule.invoke("count", 1, int)
        assert result == 1

    @pytest.mark.asyncio
    async def test_gt_invalid(self):
        """Test number not exceeding gt constraint fails."""
        rule = NumberRule(gt=0)
        with pytest.raises(ValidationError):
            await rule.invoke("count", 0, int)

    @pytest.mark.asyncio
    async def test_le_valid(self):
        """Test number meeting le constraint passes."""
        rule = NumberRule(le=1.0)
        result = await rule.invoke("confidence", 1.0, float)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_le_invalid(self):
        """Test number exceeding le constraint fails."""
        rule = NumberRule(le=1.0)
        with pytest.raises(ValidationError):
            await rule.invoke("confidence", 1.1, float)

    @pytest.mark.asyncio
    async def test_lt_valid(self):
        """Test number below lt constraint passes."""
        rule = NumberRule(lt=100)
        result = await rule.invoke("age", 99, int)
        assert result == 99

    @pytest.mark.asyncio
    async def test_lt_invalid(self):
        """Test number not below lt constraint fails."""
        rule = NumberRule(lt=100)
        with pytest.raises(ValidationError):
            await rule.invoke("age", 100, int)

    @pytest.mark.asyncio
    async def test_combined_constraints(self):
        """Test combined ge and le constraints (range)."""
        rule = NumberRule(ge=0.0, le=1.0)
        result = await rule.invoke("confidence", 0.5, float)
        assert result == 0.5

        with pytest.raises(ValidationError):
            await rule.invoke("confidence", 1.5, float)

    @pytest.mark.asyncio
    async def test_auto_fix_from_string(self):
        """Test auto-fixing string to number."""
        rule = NumberRule()
        result = await rule.invoke("score", "0.95", float)
        assert result == 0.95

    @pytest.mark.asyncio
    async def test_auto_fix_from_string_int(self):
        """Test auto-fixing string to int."""
        rule = NumberRule()
        result = await rule.invoke("count", "42", int)
        assert result == 42

    @pytest.mark.asyncio
    async def test_auto_fix_revalidates_constraints(self):
        """Test that auto-fix re-validates constraints after conversion."""
        rule = NumberRule(ge=0.0, le=1.0)
        with pytest.raises(ValidationError):
            await rule.invoke("confidence", "1.5", float)

    @pytest.mark.asyncio
    async def test_auto_fix_invalid_string(self):
        """Test auto-fix fails for non-numeric string."""
        rule = NumberRule()
        with pytest.raises(ValidationError):
            await rule.invoke("count", "not_a_number", int)


class TestBooleanRule:
    """Tests for BooleanRule validation."""

    @pytest.mark.asyncio
    async def test_valid_bool_true(self):
        """Test True passes validation."""
        rule = BooleanRule()
        result = await rule.invoke("active", True, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_valid_bool_false(self):
        """Test False passes validation."""
        rule = BooleanRule()
        result = await rule.invoke("active", False, bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_fix_string_true(self):
        """Test auto-fixing 'true' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "true", bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fix_string_false(self):
        """Test auto-fixing 'false' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "false", bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_fix_string_yes(self):
        """Test auto-fixing 'yes' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "yes", bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fix_string_no(self):
        """Test auto-fixing 'no' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "no", bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_fix_string_1(self):
        """Test auto-fixing '1' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "1", bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fix_string_0(self):
        """Test auto-fixing '0' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "0", bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_fix_string_on(self):
        """Test auto-fixing 'on' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "on", bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fix_string_off(self):
        """Test auto-fixing 'off' string."""
        rule = BooleanRule()
        result = await rule.invoke("active", "off", bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_fix_case_insensitive(self):
        """Test auto-fixing is case-insensitive."""
        rule = BooleanRule()
        assert await rule.invoke("active", "TRUE", bool) is True
        assert await rule.invoke("active", "FALSE", bool) is False
        assert await rule.invoke("active", "Yes", bool) is True
        assert await rule.invoke("active", "No", bool) is False

    @pytest.mark.asyncio
    async def test_auto_fix_int(self):
        """Test auto-fixing integers."""
        rule = BooleanRule()
        assert await rule.invoke("active", 1, bool) is True
        assert await rule.invoke("active", 0, bool) is False
        assert await rule.invoke("active", 42, bool) is True

    @pytest.mark.asyncio
    async def test_auto_fix_invalid_string(self):
        """Test auto-fix fails for invalid string."""
        rule = BooleanRule()
        with pytest.raises(ValidationError):
            await rule.invoke("active", "maybe", bool)


class TestChoiceRule:
    """Tests for ChoiceRule validation."""

    @pytest.mark.asyncio
    async def test_valid_choice(self):
        """Test valid choice passes."""
        rule = ChoiceRule(choices=["low", "medium", "high"])
        result = await rule.invoke("priority", "medium", str)
        assert result == "medium"

    @pytest.mark.asyncio
    async def test_invalid_choice(self):
        """Test invalid choice fails."""
        rule = ChoiceRule(choices=["low", "medium", "high"])
        with pytest.raises(ValidationError):
            await rule.invoke("priority", "urgent", str)

    @pytest.mark.asyncio
    async def test_case_sensitive_default(self):
        """Test case-sensitive matching by default."""
        rule = ChoiceRule(choices=["Low", "Medium", "High"])
        with pytest.raises(ValidationError):
            await rule.invoke("priority", "low", str)

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Test case-insensitive matching."""
        rule = ChoiceRule(choices=["Low", "Medium", "High"], case_sensitive=False)
        result = await rule.invoke("priority", "low", str)
        assert result == "Low"  # Returns canonical case

    @pytest.mark.asyncio
    async def test_case_insensitive_auto_fix(self):
        """Test case-insensitive auto-fix returns canonical case."""
        rule = ChoiceRule(choices=["LOW", "MEDIUM", "HIGH"], case_sensitive=False)
        result = await rule.invoke("priority", "Medium", str)
        assert result == "MEDIUM"

    @pytest.mark.asyncio
    async def test_numeric_choices(self):
        """Test numeric choices."""
        rule = ChoiceRule(choices=[1, 2, 3], apply_types={int})
        result = await rule.invoke("level", 2, int)
        assert result == 2

    @pytest.mark.asyncio
    async def test_apply_fields(self):
        """Test choice rule with apply_fields."""
        rule = ChoiceRule(choices=["draft", "review", "published"], apply_fields={"status"})
        assert await rule.apply("status", "draft", str) is True
        assert await rule.apply("other", "draft", str) is False


class TestMappingRule:
    """Tests for MappingRule validation."""

    @pytest.mark.asyncio
    async def test_valid_dict(self):
        """Test valid dict passes."""
        rule = MappingRule()
        result = await rule.invoke("config", {"key": "value"}, dict)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        """Test dict with required keys passes."""
        rule = MappingRule(required_keys={"name", "value"})
        result = await rule.invoke("config", {"name": "test", "value": 42}, dict)
        assert result == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_required_keys_missing(self):
        """Test dict missing required keys fails."""
        rule = MappingRule(required_keys={"name", "value"})
        with pytest.raises(ValidationError):
            await rule.invoke("config", {"name": "test"}, dict)

    @pytest.mark.asyncio
    async def test_auto_fix_json_string(self):
        """Test auto-fixing JSON string to dict."""
        rule = MappingRule()
        result = await rule.invoke("config", '{"key": "value"}', dict)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_auto_fix_invalid_json(self):
        """Test auto-fix fails for invalid JSON."""
        rule = MappingRule()
        with pytest.raises(ValidationError):
            await rule.invoke("config", "not json", dict)

    @pytest.mark.asyncio
    async def test_fuzzy_keys(self):
        """Test fuzzy key matching normalizes keys."""
        rule = MappingRule(required_keys={"name", "value"}, fuzzy_keys=True)
        result = await rule.invoke("config", {"NAME": "test", "Value": 42}, dict)
        assert result == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_fuzzy_keys_with_json(self):
        """Test fuzzy keys with JSON auto-fix."""
        rule = MappingRule(required_keys={"name"}, fuzzy_keys=True)
        result = await rule.invoke("config", '{"NAME": "test"}', dict)
        assert result == {"name": "test"}

    @pytest.mark.asyncio
    async def test_non_mapping_fails(self):
        """Test non-mapping type fails validation."""
        rule = MappingRule()
        with pytest.raises(ValidationError):
            await rule.invoke("config", [1, 2, 3], dict)


class TestActionRequestRule:
    """Tests for ActionRequestRule validation."""

    @pytest.mark.asyncio
    async def test_valid_action_request(self):
        """Test valid action request passes."""
        rule = ActionRequestRule()
        result = await rule.invoke(
            "action_request", {"function": "get_weather", "arguments": {"city": "NYC"}}, dict
        )
        assert result.function == "get_weather"
        assert result.arguments == {"city": "NYC"}

    @pytest.mark.asyncio
    async def test_valid_with_name_field(self):
        """Test action request with 'name' field auto-fixes to canonical format."""
        rule = ActionRequestRule()
        result = await rule.invoke(
            "action_request", {"name": "search", "arguments": {"query": "test"}}, dict
        )
        assert result.function == "search"
        assert result.arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_missing_function_fails(self):
        """Test action request without function fails."""
        rule = ActionRequestRule()
        with pytest.raises(ValidationError):
            await rule.invoke("action_request", {"arguments": {"city": "NYC"}}, dict)

    @pytest.mark.asyncio
    async def test_allowed_functions(self):
        """Test allowed functions validation."""
        rule = ActionRequestRule(allowed_functions={"get_weather", "search"})
        result = await rule.invoke(
            "action_request", {"function": "get_weather", "arguments": {}}, dict
        )
        assert result.function == "get_weather"

        with pytest.raises(ValidationError):
            await rule.invoke("action_request", {"function": "delete_all", "arguments": {}}, dict)

    @pytest.mark.asyncio
    async def test_auto_fix_json_string(self):
        """Test auto-fixing JSON string."""
        rule = ActionRequestRule()
        result = await rule.invoke(
            "action_request", '{"function": "search", "arguments": {"q": "test"}}', dict
        )
        assert result.function == "search"
        assert result.arguments == {"q": "test"}

    @pytest.mark.asyncio
    async def test_auto_fix_openai_format(self):
        """Test auto-fixing OpenAI tool_calls format with string arguments."""
        rule = ActionRequestRule()
        result = await rule.invoke(
            "tool_call", {"name": "get_weather", "arguments": '{"city": "NYC"}'}, dict
        )
        assert result.function == "get_weather"
        assert result.arguments == {"city": "NYC"}

    @pytest.mark.asyncio
    async def test_auto_fix_anthropic_format(self):
        """Test auto-fixing Anthropic tool_use format."""
        rule = ActionRequestRule()
        result = await rule.invoke(
            "tool_call", {"name": "search", "input": {"query": "test"}}, dict
        )
        assert result.function == "search"
        assert result.arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_empty_arguments_allowed(self):
        """Test action request with empty arguments passes."""
        rule = ActionRequestRule()
        result = await rule.invoke("action_request", {"function": "get_time"}, dict)
        assert result.function == "get_time"
        assert result.arguments == {}


class TestRuleApply:
    """Tests for Rule.apply() method and qualifiers."""

    @pytest.mark.asyncio
    async def test_apply_by_type(self):
        """Test rule applies by type annotation."""
        rule = StringRule()
        assert await rule.apply("any_field", "value", str) is True
        assert await rule.apply("any_field", 42, int) is False

    @pytest.mark.asyncio
    async def test_apply_by_field(self):
        """Test rule applies by field name."""
        rule = ChoiceRule(choices=["a", "b"], apply_fields={"status"})
        assert await rule.apply("status", "a", str) is True
        assert await rule.apply("other", "a", str) is False

    @pytest.mark.asyncio
    async def test_qualifier_precedence(self):
        """Test qualifier precedence order."""
        # FIELD > ANNOTATION > CONDITION
        rule = ChoiceRule(choices=["a", "b"], apply_fields={"status"}, apply_types={str})
        # Should match by FIELD first
        assert await rule.apply("status", "a", str) is True
        # Should match by ANNOTATION if field doesn't match
        assert await rule.apply("other", "a", str) is True


class TestValidatorIntegration:
    """Integration tests for Validator with all rules."""

    @pytest.mark.asyncio
    async def test_validator_default_registry(self):
        """Test validator uses default registry with standard rules."""
        validator = Validator()
        # Check registry has standard types registered
        assert validator.registry.has_rule(str)
        assert validator.registry.has_rule(int)
        assert validator.registry.has_rule(float)
        assert validator.registry.has_rule(bool)
        assert validator.registry.has_rule(dict)

    @pytest.mark.asyncio
    async def test_validator_summary(self):
        """Test validator summary generation."""
        validator = Validator()
        validator.log_validation_error("field1", "value1", "error1")
        validator.log_validation_error("field2", "value2", "error2")

        summary = validator.get_validation_summary()
        assert summary["total_errors"] == 2
        assert set(summary["fields_with_errors"]) == {"field1", "field2"}


class TestRuleInvokeAutoFixOverride:
    """Tests for Rule.invoke() auto_fix parameter override."""

    @pytest.mark.asyncio
    async def test_invoke_auto_fix_override_true(self):
        """Test invoke with auto_fix=True override."""
        # Create rule with auto_fix=False by default
        rule = NumberRule(
            params=RuleParams(
                apply_types={int, float},
                auto_fix=False,  # Default disabled
            )
        )

        # Override to enable auto_fix
        result = await rule.invoke("score", "0.5", float, auto_fix=True)
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_invoke_auto_fix_override_false(self):
        """Test invoke with auto_fix=False override."""
        # Create rule with auto_fix=True by default
        rule = NumberRule()  # Default has auto_fix=True

        # Override to disable auto_fix
        with pytest.raises(ValidationError):
            await rule.invoke("score", "0.5", float, auto_fix=False)

    @pytest.mark.asyncio
    async def test_invoke_no_override_uses_default(self):
        """Test invoke without override uses default setting."""
        rule_with_fix = NumberRule()  # auto_fix=True
        result = await rule_with_fix.invoke("score", "0.5", float)
        assert result == 0.5

        rule_without_fix = NumberRule(
            params=RuleParams(
                apply_types={int, float},
                auto_fix=False,
            )
        )
        with pytest.raises(ValidationError):
            await rule_without_fix.invoke("score", "0.5", float)
