# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for the complete Spec → Rule → Validator flow.

This tests the IPU validation pipeline:
    1. Operable.create_model(request_specs) → RequestModel for LLM
    2. LLM returns raw response
    3. Validator.validate_operable(raw_data, operable) → validated dict
    4. Operable.create_model(output_specs) → OutputModel
    5. OutputModel.model_validate(validated) → final structured output
"""

import pytest

from lionpride.rules import (
    NumberRule,
    RuleRegistry,
    StringRule,
    ValidationError,
    Validator,
)
from lionpride.types import Operable, Spec


class TestSpecRuleMapping:
    """Tests for automatic Rule assignment from Spec.base_type."""

    @pytest.mark.asyncio
    async def test_string_spec_auto_rule(self):
        """Test string Spec gets StringRule automatically."""
        spec = Spec(str, name="output")
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"output": 42},  # int will be converted to str
            operable=operable,
            auto_fix=True,
        )

        assert validated["output"] == "42"

    @pytest.mark.asyncio
    async def test_number_spec_auto_rule(self):
        """Test number Spec gets NumberRule automatically."""
        spec = Spec(float, name="confidence")
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"confidence": "0.95"},  # str will be converted to float
            operable=operable,
            auto_fix=True,
        )

        assert validated["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_bool_spec_auto_rule(self):
        """Test bool Spec gets BooleanRule automatically."""
        spec = Spec(bool, name="active")
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"active": "yes"},  # str will be converted to bool
            operable=operable,
            auto_fix=True,
        )

        assert validated["active"] is True

    @pytest.mark.asyncio
    async def test_dict_spec_auto_rule(self):
        """Test dict Spec gets MappingRule automatically."""
        spec = Spec(dict, name="config")
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"config": '{"key": "value"}'},  # JSON str will be parsed
            operable=operable,
            auto_fix=True,
        )

        assert validated["config"] == {"key": "value"}


class TestSpecMetadataOverride:
    """Tests for Spec metadata rule override."""

    @pytest.mark.asyncio
    async def test_spec_with_custom_rule(self):
        """Test Spec with explicit rule override in metadata."""
        custom_rule = NumberRule(ge=0.0, le=1.0)
        spec = Spec(float, name="score", rule=custom_rule)
        operable = Operable([spec])

        validator = Validator()

        # Valid value within range
        validated = await validator.validate_operable(
            data={"score": "0.5"},
            operable=operable,
            auto_fix=True,
        )
        assert validated["score"] == 0.5

        # Invalid value outside range should fail
        with pytest.raises(ValidationError):
            await validator.validate_operable(
                data={"score": "1.5"},
                operable=operable,
                auto_fix=True,
            )


class TestSpecNullable:
    """Tests for nullable Spec handling."""

    @pytest.mark.asyncio
    async def test_nullable_spec_accepts_none(self):
        """Test nullable Spec accepts None value."""
        spec = Spec(str, name="optional_field", nullable=True)
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"optional_field": None},
            operable=operable,
        )

        assert validated["optional_field"] is None

    @pytest.mark.asyncio
    async def test_non_nullable_spec_rejects_none(self):
        """Test non-nullable Spec rejects None value."""
        spec = Spec(str, name="required_field")
        operable = Operable([spec])

        validator = Validator()
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_operable(
                data={"required_field": None},
                operable=operable,
            )

        assert "not nullable" in str(exc_info.value).lower()


class TestSpecDefault:
    """Tests for Spec default value handling."""

    @pytest.mark.asyncio
    async def test_spec_with_sync_default(self):
        """Test Spec with sync default value."""
        spec = Spec(str, name="field", default="default_value")
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={},  # Missing field uses default
            operable=operable,
        )

        assert validated["field"] == "default_value"

    @pytest.mark.asyncio
    async def test_spec_with_sync_default_factory(self):
        """Test Spec with sync default factory."""
        spec = Spec(dict, name="config", default_factory=dict)
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={},  # Missing field uses factory
            operable=operable,
        )

        assert validated["config"] == {}

    @pytest.mark.asyncio
    async def test_spec_with_async_default_factory(self):
        """Test Spec with async default factory."""

        async def async_factory():
            return {"async": True}

        spec = Spec(dict, name="config", default_factory=async_factory)
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={},  # Missing field uses async factory
            operable=operable,
        )

        assert validated["config"] == {"async": True}


class TestSpecListable:
    """Tests for listable Spec handling."""

    @pytest.mark.asyncio
    async def test_listable_spec_validates_items(self):
        """Test listable Spec validates each item."""
        spec = Spec(str, name="tags", listable=True)
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"tags": [1, 2, 3]},  # ints will be converted to strs
            operable=operable,
            auto_fix=True,
        )

        assert validated["tags"] == ["1", "2", "3"]

    @pytest.mark.asyncio
    async def test_listable_spec_wraps_single_value(self):
        """Test listable Spec wraps single value in list with auto_fix."""
        spec = Spec(str, name="tags", listable=True)
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"tags": "single"},  # Single value wrapped in list
            operable=operable,
            auto_fix=True,
        )

        assert validated["tags"] == ["single"]

    @pytest.mark.asyncio
    async def test_listable_spec_rejects_non_list_without_autofix(self):
        """Test listable Spec rejects non-list without auto_fix."""
        spec = Spec(str, name="tags", listable=True)
        operable = Operable([spec])

        validator = Validator()
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_operable(
                data={"tags": "single"},
                operable=operable,
                auto_fix=False,
            )

        assert "expected list" in str(exc_info.value).lower()


class TestSpecCustomValidator:
    """Tests for Spec custom validator handling."""

    @pytest.mark.asyncio
    async def test_spec_with_sync_validator(self):
        """Test Spec with sync custom validator."""

        def uppercase_validator(v):
            return v.upper()

        spec = Spec(str, name="code", validator=uppercase_validator)
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"code": "abc"},
            operable=operable,
        )

        assert validated["code"] == "ABC"

    @pytest.mark.asyncio
    async def test_spec_with_async_validator(self):
        """Test Spec with async custom validator."""

        async def async_validator(v):
            return f"validated_{v}"

        spec = Spec(str, name="field", validator=async_validator)
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"field": "test"},
            operable=operable,
        )

        assert validated["field"] == "validated_test"

    @pytest.mark.asyncio
    async def test_spec_with_multiple_validators(self):
        """Test Spec with multiple validators applied in order."""

        def strip_validator(v):
            return v.strip()

        def upper_validator(v):
            return v.upper()

        spec = Spec(str, name="field", validator=[strip_validator, upper_validator])
        operable = Operable([spec])

        validator = Validator()
        validated = await validator.validate_operable(
            data={"field": "  hello  "},
            operable=operable,
        )

        assert validated["field"] == "HELLO"


class TestRuleRegistry:
    """Tests for RuleRegistry functionality."""

    @pytest.mark.asyncio
    async def test_registry_type_inheritance(self):
        """Test registry finds rules for subclasses."""

        class MyStr(str):
            pass

        registry = RuleRegistry()
        registry.register(str, StringRule(min_length=1))

        # MyStr should inherit str's rule
        rule = registry.get_rule(base_type=MyStr)
        assert rule is not None
        assert isinstance(rule, StringRule)

    @pytest.mark.asyncio
    async def test_registry_field_name_priority(self):
        """Test field name takes priority over type."""
        registry = RuleRegistry()
        type_rule = StringRule()
        name_rule = StringRule(min_length=5)

        registry.register(str, type_rule)
        registry.register("special_field", name_rule)

        # Field name should take priority
        rule = registry.get_rule(base_type=str, field_name="special_field")
        assert rule is name_rule


class TestCompleteFlow:
    """Integration tests for the complete IPU validation flow."""

    @pytest.mark.asyncio
    async def test_llm_response_validation_flow(self):
        """Test complete flow: raw LLM response → validated → model."""
        # Define specs for LLM output
        confidence_spec = Spec(float, name="confidence")
        output_spec = Spec(str, name="output")
        reasoning_spec = Spec(str, name="reasoning", nullable=True)

        operable = Operable(
            [confidence_spec, output_spec, reasoning_spec],
            name="LLMResponse",
        )

        # Simulate raw LLM response (strings, missing fields)
        raw_response = {
            "confidence": "0.95",  # String that needs conversion
            "output": 42,  # Int that needs conversion
            # reasoning is missing but nullable
        }

        # Validate
        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["confidence"] == 0.95
        assert validated["output"] == "42"
        assert validated["reasoning"] is None

        # Create output model and validate
        OutputModel = operable.create_model()
        result = OutputModel.model_validate(validated)

        assert result.confidence == 0.95
        assert result.output == "42"
        assert result.reasoning is None

    @pytest.mark.asyncio
    async def test_action_request_response_flow(self):
        """Test flow with action request → action response."""
        from lionpride.rules import ActionRequest, ActionRequestRule

        # Request phase: action_request spec (using ActionRequest type)
        action_request_spec = Spec(ActionRequest, name="action_request")
        request_operable = Operable([action_request_spec], name="ActionRequest")

        # Default registry already has ActionRequest → ActionRequestRule
        validator = Validator()

        # Validate action request (OpenAI format - auto-converts to ActionRequest)
        raw_request = {
            "action_request": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        }

        validated = await validator.validate_operable(
            data=raw_request,
            operable=request_operable,
            auto_fix=True,
        )

        # Result is an ActionRequest instance
        action_req = validated["action_request"]
        assert isinstance(action_req, ActionRequest)
        assert action_req.function == "get_weather"
        assert action_req.arguments == {"city": "NYC"}
