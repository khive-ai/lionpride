# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the complete IPU validation pipeline.

This tests the full flow:
    1. Define Specs with base_types (str, float, ActionRequest, etc.)
    2. Create Operable from Specs
    3. Generate RequestModel for LLM structured output
    4. Simulate LLM response (raw data in various formats)
    5. Validator.validate_operable() applies Rules based on Spec.base_type
    6. Create OutputModel and validate final result

This demonstrates the cohesiveness of: Spec → Rule → Validator → Operable.create_model()
"""

import pytest

from lionpride.rules import (
    ActionRequest,
    ActionRequestRule,
    ActionResponse,
    NumberRule,
    RuleRegistry,
    StringRule,
    ValidationError,
    Validator,
)
from lionpride.types import Operable, Spec


class TestE2EBasicValidationFlow:
    """Test basic validation flow with primitive types."""

    @pytest.mark.asyncio
    async def test_simple_llm_response_flow(self):
        """Test: Raw LLM response → Validated → OutputModel.

        Scenario: LLM returns string numbers and mistyped values.
        The validator should auto-fix them to correct types.
        """
        # Step 1: Define specs
        confidence_spec = Spec(float, name="confidence")
        output_spec = Spec(str, name="output")
        score_spec = Spec(int, name="score")

        # Step 2: Create operable
        operable = Operable(
            [confidence_spec, output_spec, score_spec],
            name="LLMAnalysis",
        )

        # Step 3: Create request model (would be sent to LLM)
        RequestModel = operable.create_model()
        assert RequestModel is not None
        assert "confidence" in RequestModel.model_fields
        assert "output" in RequestModel.model_fields
        assert "score" in RequestModel.model_fields

        # Step 4: Simulate raw LLM response (all strings, typical LLM behavior)
        raw_response = {
            "confidence": "0.95",  # String that needs float conversion
            "output": 42,  # Int that needs string conversion
            "score": "7",  # String that needs int conversion
        }

        # Step 5: Validate with auto-fix
        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        # Step 6: Verify types are correct
        assert validated["confidence"] == 0.95
        assert isinstance(validated["confidence"], float)
        assert validated["output"] == "42"
        assert isinstance(validated["output"], str)
        assert validated["score"] == 7
        assert isinstance(validated["score"], int)

        # Step 7: Create output model and validate final result
        OutputModel = operable.create_model()
        result = OutputModel.model_validate(validated)

        assert result.confidence == 0.95
        assert result.output == "42"
        assert result.score == 7


class TestE2EActionRequestFlow:
    """Test action request/response flow for tool calls."""

    @pytest.mark.asyncio
    async def test_openai_tool_call_flow(self):
        """Test: OpenAI format tool_call → ActionRequest → Execute → ActionResponse.

        Scenario: LLM returns OpenAI-style tool calls with JSON string arguments.
        The validator should normalize to ActionRequest instances.
        """
        # Step 1: Define action request spec using ActionRequest type
        action_spec = Spec(ActionRequest, name="action_request")
        operable = Operable([action_spec], name="ToolCall")

        # Step 2: Simulate OpenAI-style tool call response
        # OpenAI returns: {"name": str, "arguments": str (JSON)}
        openai_response = {
            "action_request": {
                "name": "get_weather",
                "arguments": '{"city": "New York", "units": "celsius"}',
            }
        }

        # Step 3: Validate - auto-converts to ActionRequest
        validator = Validator()
        validated = await validator.validate_operable(
            data=openai_response,
            operable=operable,
            auto_fix=True,
        )

        # Step 4: Result is ActionRequest instance
        action_req = validated["action_request"]
        assert isinstance(action_req, ActionRequest)
        assert action_req.function == "get_weather"
        assert action_req.arguments == {"city": "New York", "units": "celsius"}

    @pytest.mark.asyncio
    async def test_anthropic_tool_use_flow(self):
        """Test: Anthropic format tool_use → ActionRequest.

        Scenario: LLM returns Anthropic-style tool_use with dict input.
        """
        action_spec = Spec(ActionRequest, name="tool_call")
        operable = Operable([action_spec], name="ToolUse")

        # Anthropic returns: {"name": str, "input": dict}
        anthropic_response = {
            "tool_call": {
                "name": "search_documents",
                "input": {"query": "machine learning", "max_results": 10},
            }
        }

        validator = Validator()
        validated = await validator.validate_operable(
            data=anthropic_response,
            operable=operable,
            auto_fix=True,
        )

        action_req = validated["tool_call"]
        assert isinstance(action_req, ActionRequest)
        assert action_req.function == "search_documents"
        assert action_req.arguments == {"query": "machine learning", "max_results": 10}

    @pytest.mark.asyncio
    async def test_multi_field_with_action_request(self):
        """Test: Mixed fields including ActionRequest.

        Scenario: LLM response with confidence, reasoning, AND action request.
        """
        confidence_spec = Spec(float, name="confidence")
        reasoning_spec = Spec(str, name="reasoning", nullable=True)
        action_spec = Spec(ActionRequest, name="action_request")

        operable = Operable(
            [confidence_spec, reasoning_spec, action_spec],
            name="AgentDecision",
        )

        # Simulate complex LLM response
        raw_response = {
            "confidence": "0.87",
            "reasoning": "User asked about weather, using weather tool",
            "action_request": {
                "name": "get_weather",
                "arguments": '{"location": "Seattle"}',
            },
        }

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["confidence"] == 0.87
        assert validated["reasoning"] == "User asked about weather, using weather tool"
        assert isinstance(validated["action_request"], ActionRequest)
        assert validated["action_request"].function == "get_weather"
        assert validated["action_request"].arguments == {"location": "Seattle"}


class TestE2EListableSpecs:
    """Test listable specs for array outputs."""

    @pytest.mark.asyncio
    async def test_listable_strings_flow(self):
        """Test: Listable spec validates each item in list."""
        tags_spec = Spec(str, name="tags", listable=True)
        operable = Operable([tags_spec], name="TagExtraction")

        # LLM might return ints that need string conversion
        raw_response = {"tags": [1, "two", 3.0, "four"]}

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["tags"] == ["1", "two", "3.0", "four"]
        assert all(isinstance(t, str) for t in validated["tags"])

    @pytest.mark.asyncio
    async def test_listable_single_value_wrapping(self):
        """Test: Single value is wrapped in list with auto_fix."""
        items_spec = Spec(str, name="items", listable=True)
        operable = Operable([items_spec], name="ItemList")

        # LLM returns single value instead of list
        raw_response = {"items": "single_item"}

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["items"] == ["single_item"]

    @pytest.mark.asyncio
    async def test_listable_action_requests(self):
        """Test: List of action requests (parallel tool calls)."""
        actions_spec = Spec(ActionRequest, name="actions", listable=True)
        operable = Operable([actions_spec], name="ParallelTools")

        # LLM returns multiple tool calls
        raw_response = {
            "actions": [
                {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                {"name": "get_time", "arguments": '{"timezone": "EST"}'},
            ]
        }

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert len(validated["actions"]) == 2
        assert all(isinstance(a, ActionRequest) for a in validated["actions"])
        assert validated["actions"][0].function == "get_weather"
        assert validated["actions"][1].function == "get_time"


class TestE2ENullableAndDefaults:
    """Test nullable and default value handling."""

    @pytest.mark.asyncio
    async def test_nullable_field_accepts_none(self):
        """Test: Nullable spec accepts None without error."""
        required_spec = Spec(str, name="required")
        optional_spec = Spec(str, name="optional", nullable=True)

        operable = Operable([required_spec, optional_spec], name="WithOptional")

        raw_response = {
            "required": "value",
            "optional": None,  # Explicitly None
        }

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["required"] == "value"
        assert validated["optional"] is None

    @pytest.mark.asyncio
    async def test_default_value_used_when_missing(self):
        """Test: Default value used when field is missing."""
        name_spec = Spec(str, name="name")
        status_spec = Spec(str, name="status", default="pending")

        operable = Operable([name_spec, status_spec], name="Task")

        # status is missing from response
        raw_response = {"name": "My Task"}

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["name"] == "My Task"
        assert validated["status"] == "pending"

    @pytest.mark.asyncio
    async def test_default_factory_creates_fresh_value(self):
        """Test: Default factory creates new instance each time."""
        items_spec = Spec(dict, name="metadata", default_factory=dict)

        operable = Operable([items_spec], name="WithMetadata")

        validator = Validator()

        validated1 = await validator.validate_operable(data={}, operable=operable, auto_fix=True)
        validated2 = await validator.validate_operable(data={}, operable=operable, auto_fix=True)

        # Each should get a fresh dict
        assert validated1["metadata"] == {}
        assert validated2["metadata"] == {}
        assert validated1["metadata"] is not validated2["metadata"]


class TestE2ECustomValidators:
    """Test custom validators in specs."""

    @pytest.mark.asyncio
    async def test_sync_custom_validator(self):
        """Test: Sync validator transforms value after rule validation."""

        def uppercase(v):
            return v.upper()

        code_spec = Spec(str, name="code", validator=uppercase)
        operable = Operable([code_spec], name="Code")

        raw_response = {"code": "abc123"}

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["code"] == "ABC123"

    @pytest.mark.asyncio
    async def test_async_custom_validator(self):
        """Test: Async validator is awaited properly."""

        async def async_transform(v):
            return f"processed_{v}"

        field_spec = Spec(str, name="data", validator=async_transform)
        operable = Operable([field_spec], name="Data")

        raw_response = {"data": "input"}

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["data"] == "processed_input"

    @pytest.mark.asyncio
    async def test_multiple_validators_chain(self):
        """Test: Multiple validators applied in order."""

        def strip(v):
            return v.strip()

        def upper(v):
            return v.upper()

        def prefix(v):
            return f"PREFIX_{v}"

        field_spec = Spec(str, name="text", validator=[strip, upper, prefix])
        operable = Operable([field_spec], name="Text")

        raw_response = {"text": "  hello  "}

        validator = Validator()
        validated = await validator.validate_operable(
            data=raw_response,
            operable=operable,
            auto_fix=True,
        )

        # strip → "hello" → upper → "HELLO" → prefix → "PREFIX_HELLO"
        assert validated["text"] == "PREFIX_HELLO"


class TestE2ECustomRuleOverride:
    """Test spec-level rule override via metadata."""

    @pytest.mark.asyncio
    async def test_spec_rule_override_priority(self):
        """Test: Rule in spec metadata takes priority over registry."""
        # Custom rule with constraints
        custom_rule = NumberRule(ge=0.0, le=1.0)

        # Spec with explicit rule override
        confidence_spec = Spec(float, name="confidence", rule=custom_rule)
        operable = Operable([confidence_spec], name="Confidence")

        validator = Validator()

        # Valid value within range
        validated = await validator.validate_operable(
            data={"confidence": "0.5"},
            operable=operable,
            auto_fix=True,
        )
        assert validated["confidence"] == 0.5

        # Invalid value outside range should fail
        with pytest.raises(ValidationError):
            await validator.validate_operable(
                data={"confidence": "1.5"},  # > 1.0
                operable=operable,
                auto_fix=True,
            )

    @pytest.mark.asyncio
    async def test_allowed_functions_constraint(self):
        """Test: ActionRequestRule with allowed_functions constraint."""
        # Create constrained rule
        tool_rule = ActionRequestRule(allowed_functions={"get_weather", "search"})

        # Spec with constrained rule
        action_spec = Spec(ActionRequest, name="action", rule=tool_rule)
        operable = Operable([action_spec], name="ConstrainedAction")

        validator = Validator()

        # Allowed function works
        validated = await validator.validate_operable(
            data={"action": {"name": "get_weather", "arguments": "{}"}},
            operable=operable,
            auto_fix=True,
        )
        assert validated["action"].function == "get_weather"

        # Disallowed function fails
        with pytest.raises(ValidationError) as exc_info:
            await validator.validate_operable(
                data={"action": {"name": "delete_all", "arguments": "{}"}},
                operable=operable,
                auto_fix=True,
            )
        assert "delete_all" in str(exc_info.value)


class TestE2ECompleteAgentFlow:
    """Test complete agent operation flow."""

    @pytest.mark.asyncio
    async def test_full_agent_decision_cycle(self):
        """Test: Full cycle from request model to validated output.

        This simulates:
        1. Agent receives user request
        2. Creates structured prompt with RequestModel
        3. LLM returns raw response
        4. Validator normalizes response
        5. OutputModel validates final structure
        """
        # Define complete agent response spec
        thinking_spec = Spec(str, name="thinking", nullable=True)
        confidence_spec = Spec(float, name="confidence")
        action_spec = Spec(ActionRequest, name="action_request", nullable=True)
        final_answer_spec = Spec(str, name="final_answer", nullable=True)

        operable = Operable(
            [thinking_spec, confidence_spec, action_spec, final_answer_spec],
            name="AgentResponse",
        )

        # Scenario 1: Agent decides to use a tool
        tool_response = {
            "thinking": "User wants weather info, I'll use the weather tool",
            "confidence": "0.92",
            "action_request": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            "final_answer": None,
        }

        validator = Validator()
        validated = await validator.validate_operable(
            data=tool_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated["thinking"] == "User wants weather info, I'll use the weather tool"
        assert validated["confidence"] == 0.92
        assert isinstance(validated["action_request"], ActionRequest)
        assert validated["action_request"].function == "get_weather"
        assert validated["final_answer"] is None

        # Create model and validate
        ResponseModel = operable.create_model()
        result = ResponseModel.model_validate(validated)
        assert result.confidence == 0.92

        # Scenario 2: Agent provides direct answer
        answer_response = {
            "thinking": "This is a simple question I can answer directly",
            "confidence": "0.99",
            "action_request": None,
            "final_answer": "The answer is 42.",
        }

        validated2 = await validator.validate_operable(
            data=answer_response,
            operable=operable,
            auto_fix=True,
        )

        assert validated2["action_request"] is None
        assert validated2["final_answer"] == "The answer is 42."


class TestE2EValidationLog:
    """Test validation logging for debugging."""

    @pytest.mark.asyncio
    async def test_validation_log_tracks_errors(self):
        """Test: Validation errors are logged for debugging."""
        validator = Validator()

        # Spec with constraint
        score_spec = Spec(float, name="score", rule=NumberRule(ge=0.0, le=1.0))
        operable = Operable([score_spec], name="Score")

        # This should fail and log error
        try:
            await validator.validate_operable(
                data={"score": "2.0"},  # > 1.0
                operable=operable,
                auto_fix=True,
            )
        except ValidationError:
            pass

        # Check validation log
        assert len(validator.validation_log) >= 1
        summary = validator.get_validation_summary()
        assert summary["total_errors"] >= 1
        assert "score" in summary["fields_with_errors"]

    @pytest.mark.asyncio
    async def test_strict_false_allows_unknown_types(self):
        """Test: strict=False allows fields without rules to pass through."""
        # Custom operable with list type (no default rule)
        items_spec = Spec(list, name="items")
        operable = Operable([items_spec], name="Items")

        validator = Validator()

        # strict=True should raise
        with pytest.raises(ValidationError):
            await validator.validate_operable(
                data={"items": [1, 2, 3]},
                operable=operable,
                strict=True,
            )

        # strict=False should pass through
        validated = await validator.validate_operable(
            data={"items": [1, 2, 3]},
            operable=operable,
            strict=False,
        )
        assert validated["items"] == [1, 2, 3]
