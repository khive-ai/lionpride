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

    @pytest.mark.asyncio
    async def test_perform_fix_action_request_passthrough(self):
        """Test perform_fix passes through ActionRequest instance (lines 114-115).

        Call perform_fix directly since invoke() won't call it when
        ActionRequest already passes validate().
        """
        from lionpride.rules.models import ActionRequest

        rule = ActionRequestRule()
        action_req = ActionRequest(function="test_func", arguments={"key": "value"})
        # Call perform_fix directly to hit lines 114-115
        result = await rule.perform_fix(action_req, dict)
        assert result is action_req
        assert result.function == "test_func"
        assert result.arguments == {"key": "value"}

    @pytest.mark.asyncio
    async def test_perform_fix_invalid_json_string(self):
        """Test perform_fix fails on invalid JSON string (lines 121-122)."""
        rule = ActionRequestRule()
        with pytest.raises(ValidationError, match="Failed to parse action request JSON"):
            await rule.invoke("action_request", "not valid json {", dict)

    @pytest.mark.asyncio
    async def test_perform_fix_non_mapping_fails(self):
        """Test perform_fix fails on non-mapping value (line 125)."""
        rule = ActionRequestRule()
        with pytest.raises(ValidationError, match="Cannot convert list to action request"):
            await rule.invoke("action_request", [1, 2, 3], dict)

    @pytest.mark.asyncio
    async def test_perform_fix_arguments_invalid_json_string(self):
        """Test perform_fix wraps invalid JSON arguments as raw (lines 153-155)."""
        rule = ActionRequestRule()
        result = await rule.invoke(
            "action_request",
            {"function": "test_func", "arguments": "not valid json"},
            dict,
        )
        assert result.function == "test_func"
        assert result.arguments == {"raw": "not valid json"}

    @pytest.mark.asyncio
    async def test_perform_fix_arguments_non_mapping(self):
        """Test perform_fix wraps non-mapping arguments as value (line 158)."""
        rule = ActionRequestRule()
        result = await rule.invoke(
            "action_request",
            {"function": "test_func", "arguments": [1, 2, 3]},
            dict,
        )
        assert result.function == "test_func"
        assert result.arguments == {"value": [1, 2, 3]}


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


class TestRuleQualifierFromStr:
    """Tests for RuleQualifier.from_str() method."""

    def test_field_qualifier(self):
        """Test parsing FIELD qualifier."""
        assert RuleQualifier.from_str("FIELD") == RuleQualifier.FIELD
        assert RuleQualifier.from_str("field") == RuleQualifier.FIELD
        assert RuleQualifier.from_str("  Field  ") == RuleQualifier.FIELD

    def test_annotation_qualifier(self):
        """Test parsing ANNOTATION qualifier."""
        assert RuleQualifier.from_str("ANNOTATION") == RuleQualifier.ANNOTATION
        assert RuleQualifier.from_str("annotation") == RuleQualifier.ANNOTATION

    def test_condition_qualifier(self):
        """Test parsing CONDITION qualifier."""
        assert RuleQualifier.from_str("CONDITION") == RuleQualifier.CONDITION
        assert RuleQualifier.from_str("condition") == RuleQualifier.CONDITION

    def test_unknown_qualifier_raises(self):
        """Test unknown qualifier string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown RuleQualifier"):
            RuleQualifier.from_str("INVALID")
        with pytest.raises(ValueError, match="Unknown RuleQualifier"):
            RuleQualifier.from_str("unknown")


class TestDecideQualifierOrder:
    """Tests for _decide_qualifier_order() function."""

    @pytest.mark.asyncio
    async def test_string_qualifier_moves_to_front(self):
        """Test string qualifier is converted and moved to front."""
        # Use Rule.apply() which internally calls _decide_qualifier_order
        rule = ChoiceRule(choices=["a", "b"], apply_types={str})
        # Pass qualifier as string - tests lines 69-70
        assert await rule.apply("field", "a", str, qualifier="annotation") is True

    @pytest.mark.asyncio
    async def test_string_qualifier_condition(self):
        """Test string qualifier CONDITION is parsed."""
        rule = StringRule()
        # CONDITION qualifier will raise NotImplementedError since StringRule
        # doesn't implement rule_condition, but it should try other qualifiers
        assert await rule.apply("field", "test", str, qualifier="condition") is True


class TestRuleParamsValidation:
    """Tests for RuleParams._validate() method.

    Note: RuleParams._validate() has a bug where super()._validate() fails
    with frozen dataclass + slots. Lines 107-115 are unreachable until fixed.
    These tests verify the data structures can be created.
    """

    def test_params_with_apply_types(self):
        """Test RuleParams with apply_types set."""
        params = RuleParams(apply_types={str, int})
        assert str in params.apply_types
        assert int in params.apply_types

    def test_params_with_apply_fields(self):
        """Test RuleParams with apply_fields set."""
        params = RuleParams(apply_fields={"name", "value"})
        assert "name" in params.apply_fields
        assert "value" in params.apply_fields

    def test_params_with_condition_qualifier(self):
        """Test RuleParams with CONDITION qualifier allows no types/fields."""
        params = RuleParams(default_qualifier=RuleQualifier.CONDITION)
        assert params.default_qualifier == RuleQualifier.CONDITION
        assert not params.apply_types
        assert not params.apply_fields


class TestRuleInitWithKwargs:
    """Tests for Rule.__init__ with additional kwargs."""

    @pytest.mark.asyncio
    async def test_init_merges_kwargs(self):
        """Test Rule.__init__ merges kwargs with params.kw."""
        # Note: StringRule handles min_length/max_length as instance attrs,
        # so we use arbitrary kwargs to test the merge behavior in Rule.__init__
        params = RuleParams(apply_types={str}, kw={"existing_key": "original"})
        rule = StringRule(params=params, extra_key="merged_value")
        # Verify kwargs are merged in validation_kwargs
        assert rule.validation_kwargs.get("existing_key") == "original"
        assert rule.validation_kwargs.get("extra_key") == "merged_value"


class TestRuleDefaultQualifierProperty:
    """Tests for Rule.default_qualifier property."""

    def test_default_qualifier_property(self):
        """Test default_qualifier property returns params value."""
        params = RuleParams(apply_types={str}, default_qualifier=RuleQualifier.ANNOTATION)
        rule = StringRule(params=params)
        assert rule.default_qualifier == RuleQualifier.ANNOTATION


class TestRulePerformFixNotImplemented:
    """Tests for Rule.perform_fix() NotImplementedError."""

    @pytest.mark.asyncio
    async def test_perform_fix_not_implemented_raises(self):
        """Test perform_fix raises NotImplementedError when not implemented."""
        from typing import Any

        # Create a minimal concrete rule that doesn't implement perform_fix
        class MinimalRule(Rule):
            def __init__(self):
                super().__init__(RuleParams(apply_types={str}, auto_fix=True))

            async def validate(self, v: Any, t: type, **kw) -> None:
                if not isinstance(v, str):
                    raise ValueError("Not a string")

        rule = MinimalRule()
        # When validation fails and auto_fix is True, perform_fix is called
        # but MinimalRule doesn't implement it, so it should raise
        with pytest.raises(ValidationError, match="Failed to fix field"):
            await rule.invoke("field", 123, str)


class TestRuleRepr:
    """Tests for Rule.__repr__() method."""

    def test_repr_format(self):
        """Test __repr__ returns expected format."""
        rule = StringRule()
        repr_str = repr(rule)
        assert "StringRule(" in repr_str
        assert "types=" in repr_str
        assert "fields=" in repr_str
        assert "auto_fix=" in repr_str

    def test_repr_with_fields(self):
        """Test __repr__ with apply_fields."""
        rule = ChoiceRule(choices=["a", "b"], apply_fields={"status"})
        repr_str = repr(rule)
        assert "ChoiceRule(" in repr_str
        assert "status" in repr_str or "fields=" in repr_str


class TestRuleParamsValidationConstraints:
    """Tests for RuleParams dataclass constraints.

    NOTE: Lines 107-115 in base.py (RuleParams._validate) are currently DEAD CODE.
    The dataclass decorator on RuleParams generates __init__ without calling
    super().__init__(), so _validate() is never invoked. These tests verify
    the current behavior (no validation) rather than the intended behavior.
    """

    def test_params_both_types_and_fields_currently_allowed(self):
        """Verify both apply_types and apply_fields can be set (dead code issue).

        NOTE: This SHOULD raise ValueError but doesn't because _validate() is dead code.
        When fixed, this test should expect ValueError.
        """
        # Currently allowed due to dead code - this is a BUG to be fixed
        params = RuleParams(apply_types={str}, apply_fields={"name"})
        assert params.apply_types == {str}
        assert params.apply_fields == {"name"}

    def test_params_neither_set_currently_allowed(self):
        """Verify neither apply_types nor apply_fields can be empty (dead code issue).

        NOTE: This SHOULD raise ValueError for non-CONDITION qualifier but doesn't.
        """
        params = RuleParams()
        assert params.apply_types == set()
        assert params.apply_fields == set()

    def test_params_neither_with_condition_qualifier(self):
        """Test CONDITION qualifier with neither types nor fields."""
        params = RuleParams(default_qualifier=RuleQualifier.CONDITION)
        assert params.default_qualifier == RuleQualifier.CONDITION
        assert not params.apply_types
        assert not params.apply_fields


class TestBooleanRuleFallback:
    """Tests for BooleanRule.perform_fix() fallback (line 88)."""

    @pytest.mark.asyncio
    async def test_auto_fix_list_to_bool(self):
        """Test auto-fixing list to bool uses bool() fallback (line 88)."""
        rule = BooleanRule()
        # Non-empty list → True via bool()
        result = await rule.invoke("active", [1, 2, 3], bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fix_empty_list_to_bool(self):
        """Test auto-fixing empty list to bool uses bool() fallback."""
        rule = BooleanRule()
        # Empty list → False via bool()
        result = await rule.invoke("active", [], bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_fix_dict_to_bool(self):
        """Test auto-fixing dict to bool uses bool() fallback."""
        rule = BooleanRule()
        # Non-empty dict → True via bool()
        result = await rule.invoke("active", {"key": "value"}, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fix_empty_dict_to_bool(self):
        """Test auto-fixing empty dict to bool uses bool() fallback."""
        rule = BooleanRule()
        # Empty dict → False via bool()
        result = await rule.invoke("active", {}, bool)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_fix_tuple_to_bool(self):
        """Test auto-fixing tuple to bool uses bool() fallback."""
        rule = BooleanRule()
        result = await rule.invoke("active", (1,), bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fix_none_to_bool(self):
        """Test auto-fixing None to bool uses bool() fallback."""
        rule = BooleanRule()
        result = await rule.invoke("active", None, bool)
        assert result is False


class TestChoiceRulePerformFixAlreadyValid:
    """Tests for ChoiceRule.perform_fix() when value already valid (line 94)."""

    @pytest.mark.asyncio
    async def test_perform_fix_already_valid_passthrough(self):
        """Test perform_fix returns valid value unchanged (line 93-94).

        Call perform_fix directly since invoke() won't call it when
        value already passes validate().
        """
        rule = ChoiceRule(choices=["low", "medium", "high"])
        # Call perform_fix directly to hit line 94
        result = await rule.perform_fix("medium", str)
        assert result == "medium"

    @pytest.mark.asyncio
    async def test_perform_fix_numeric_already_valid(self):
        """Test perform_fix with numeric choice already valid."""
        rule = ChoiceRule(choices=[1, 2, 3], apply_types={int})
        result = await rule.perform_fix(2, int)
        assert result == 2


class TestMappingRuleKeepOriginalKey:
    """Tests for MappingRule.perform_fix() keeping original key (line 123)."""

    @pytest.mark.asyncio
    async def test_fuzzy_keys_unknown_key_preserved(self):
        """Test fuzzy key matching preserves unknown keys (line 122-123)."""
        rule = MappingRule(required_keys={"name"}, optional_keys={"value"}, fuzzy_keys=True)
        # "unknown_key" is not in required_keys or optional_keys, so it should be kept as-is
        result = await rule.invoke(
            "config", {"NAME": "test", "unknown_key": "preserved_value"}, dict
        )
        assert result == {"name": "test", "unknown_key": "preserved_value"}

    @pytest.mark.asyncio
    async def test_fuzzy_keys_multiple_unknown_keys_preserved(self):
        """Test fuzzy key matching preserves multiple unknown keys."""
        rule = MappingRule(required_keys={"name"}, fuzzy_keys=True)
        result = await rule.invoke(
            "config",
            {"NAME": "test", "extra1": "value1", "Extra2": "value2"},
            dict,
        )
        assert result["name"] == "test"
        assert result["extra1"] == "value1"
        assert result["Extra2"] == "value2"

    @pytest.mark.asyncio
    async def test_fuzzy_keys_only_unknown_keys(self):
        """Test fuzzy key matching with only unknown keys keeps them all."""
        rule = MappingRule(fuzzy_keys=True)
        result = await rule.invoke("config", {"random_key": "value1", "another": "value2"}, dict)
        assert result["random_key"] == "value1"
        assert result["another"] == "value2"


class TestReasonRuleConfidenceValidation:
    """Tests for ReasonRule confidence validation.

    NOTE: Lines 68-69 in reason.py are DEAD CODE because Pydantic's Reason model
    validates confidence range (ge=0.0, le=1.0) at construction time. It's impossible
    to create a Reason instance with invalid confidence, so the ReasonRule.validate()
    check can never be reached.
    """

    @pytest.mark.asyncio
    async def test_pydantic_prevents_invalid_confidence(self):
        """Verify Pydantic prevents creating Reason with invalid confidence.

        This shows why line 69 is unreachable - Pydantic raises before ReasonRule.validate().
        """
        import pydantic

        from lionpride.rules import Reason

        with pytest.raises(pydantic.ValidationError):
            Reason(reasoning="test", confidence=1.5)

        with pytest.raises(pydantic.ValidationError):
            Reason(reasoning="test", confidence=-0.5)

    @pytest.mark.asyncio
    async def test_valid_confidence_boundaries(self):
        """Test that valid confidence at boundaries passes."""
        from lionpride.rules import Reason
        from lionpride.rules.reason import ReasonRule

        rule = ReasonRule()

        # 0.0 boundary
        reason_zero = Reason(reasoning="test", confidence=0.0)
        await rule.validate(reason_zero, Reason)

        # 1.0 boundary
        reason_one = Reason(reasoning="test", confidence=1.0)
        await rule.validate(reason_one, Reason)

    @pytest.mark.asyncio
    async def test_none_confidence_valid(self):
        """Test that None confidence passes validation."""
        from lionpride.rules import Reason
        from lionpride.rules.reason import ReasonRule

        rule = ReasonRule()
        reason = Reason(reasoning="test", confidence=None)
        await rule.validate(reason, Reason)


class TestStringRulePerformFixException:
    """Tests for StringRule.perform_fix() exception handling (lines 98-99)."""

    @pytest.mark.asyncio
    async def test_perform_fix_str_conversion_failure(self):
        """Test perform_fix handles str() conversion failure (lines 98-99)."""

        class Unconvertible:
            """Object that raises exception when converted to str."""

            def __str__(self):
                raise RuntimeError("Cannot convert to string")

        rule = StringRule()
        with pytest.raises(ValidationError, match="Failed to fix field"):
            await rule.invoke("field", Unconvertible(), str)

    @pytest.mark.asyncio
    async def test_perform_fix_str_raises_type_error(self):
        """Test perform_fix handles TypeError from str() conversion."""

        class TypeErrorStr:
            """Object that raises TypeError when converted to str."""

            def __str__(self):
                raise TypeError("Type error during conversion")

        rule = StringRule()
        with pytest.raises(ValidationError, match="Failed to fix field"):
            await rule.invoke("field", TypeErrorStr(), str)


class TestBaseModelRule:
    """Tests for BaseModelRule validation."""

    @pytest.mark.asyncio
    async def test_validate_non_basemodel_type_raises(self):
        """Test validate raises when expected type is not BaseModel subclass."""
        from lionpride.rules import BaseModelRule

        rule = BaseModelRule()
        with pytest.raises(ValueError, match="must be a BaseModel subclass"):
            await rule.validate("value", str)  # str is not BaseModel

    @pytest.mark.asyncio
    async def test_validate_already_correct_type(self):
        """Test validate passes for already correct type."""
        from pydantic import BaseModel

        from lionpride.rules import BaseModelRule

        class MyModel(BaseModel):
            name: str

        rule = BaseModelRule()
        instance = MyModel(name="test")
        # Should not raise - already correct type
        await rule.validate(instance, MyModel)

    @pytest.mark.asyncio
    async def test_validate_dict_success(self):
        """Test validate passes for valid dict input."""
        from pydantic import BaseModel

        from lionpride.rules import BaseModelRule

        class MyModel(BaseModel):
            name: str
            value: int

        rule = BaseModelRule()
        await rule.validate({"name": "test", "value": 42}, MyModel)

    @pytest.mark.asyncio
    async def test_validate_dict_failure(self):
        """Test validate raises for invalid dict input."""
        from pydantic import BaseModel

        from lionpride.rules import BaseModelRule

        class MyModel(BaseModel):
            name: str
            value: int

        rule = BaseModelRule()
        with pytest.raises(ValueError, match="Dict validation failed"):
            await rule.validate({"name": "test"}, MyModel)  # Missing value

    @pytest.mark.asyncio
    async def test_validate_wrong_type(self):
        """Test validate raises for non-dict, non-model input."""
        from pydantic import BaseModel

        from lionpride.rules import BaseModelRule

        class MyModel(BaseModel):
            name: str

        rule = BaseModelRule()
        with pytest.raises(ValueError, match="Cannot validate"):
            await rule.validate(12345, MyModel)  # int cannot be validated

    @pytest.mark.asyncio
    async def test_perform_fix_uses_fuzzy_validate(self):
        """Test perform_fix uses fuzzy_validate_pydantic."""
        from pydantic import BaseModel

        from lionpride.rules import BaseModelRule

        class MyModel(BaseModel):
            name: str
            value: int

        rule = BaseModelRule(fuzzy_parse=True, fuzzy_match=False)
        result = await rule.perform_fix({"name": "test", "value": 42}, MyModel)
        assert isinstance(result, MyModel)
        assert result.name == "test"
        assert result.value == 42
