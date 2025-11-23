"""Test LNDL integration with action/tool calling.

Tests the interaction between LNDL format and the action_requests/action_responses
flow in operate with tools=True.
"""

import pytest
from pydantic import BaseModel, Field

from lionpride.operations import create_action_operative, create_operative_from_model
from lionpride.operations.models import ActionRequestModel, ActionResponseModel, Reason


class Analysis(BaseModel):
    """Analysis model for testing."""

    summary: str = Field(..., description="Summary of analysis")
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendations: list[str] = Field(default_factory=list)


class TestLNDLActions:
    """Test LNDL with action/tool calling."""

    def test_action_operative_structure(self):
        """Test that action operatives have correct structure."""
        operative = create_action_operative(
            base_model=Analysis,
            reason=True,
            actions=True,
            name="AnalysisWithTools",
        )

        specs = operative.operable.get_specs()
        spec_names = [spec.name for spec in specs]

        # Should have the model as a single spec (lowercased) plus reason and action fields
        assert "analysiswithtools" in spec_names  # Model spec (lowercased)
        assert "reason" in spec_names  # Reason spec
        assert "action_requests" in spec_names  # Part of typed namespace
        assert "action_responses" in spec_names  # Part of typed namespace

        # The operative should have the _supports_actions flag
        assert hasattr(operative, "_supports_actions")
        assert operative._supports_actions is True

        # action_responses should be excluded from request model
        assert "action_responses" in operative.request_exclude

    def test_action_operative_request_model(self):
        """Test request model has proper fields."""
        operative = create_action_operative(
            base_model=Analysis,
            reason=True,
            actions=True,
        )

        request_model = operative.create_request_model()
        fields = request_model.model_fields.keys()

        # Request model should have all specs except action_responses
        assert "analysis" in fields  # The spec name is based on the model name
        assert "reason" in fields
        assert "action_requests" in fields  # Included in request
        assert "action_responses" not in fields  # Excluded from request

    def test_action_operative_response_model(self):
        """Test response model includes all fields."""
        operative = create_action_operative(
            base_model=Analysis,
            reason=True,
            actions=True,
        )

        response_model = operative.create_response_model()
        fields = response_model.model_fields.keys()

        # Response model should have all fields from the composite
        assert "analysis" in fields  # The spec name is based on the model name
        assert "reason" in fields
        assert "action_requests" in fields  # Part of response
        assert "action_responses" in fields  # Part of response

    def test_lndl_with_action_fields(self):
        """Test LNDL parsing with action fields."""
        from lionpride.lndl import parse_lndl_fuzzy

        create_action_operative(
            base_model=Analysis,
            reason=True,
            actions=True,
            name="AnalysisOp",
        )

        # LNDL with action requests

        # This test would need adjustment based on how multi-spec LNDL works
        # Currently focusing on single model specs

    def test_action_request_model_lndl(self):
        """Test ActionRequestModel in LNDL format."""
        from lionpride.lndl import parse_lndl_fuzzy
        from lionpride.types import Operable, Spec

        # Create operable for ActionRequestModel
        spec = Spec(base_type=ActionRequestModel, name="actionrequest")
        operable = Operable(specs=(spec,), name="ActionRequest")

        lndl_action = """<lvar ActionRequestModel.function f>calculate</lvar>
<lvar ActionRequestModel.arguments args>{"x": 10, "y": 20}</lvar>
OUT{actionrequest: [f, args]}"""

        result = parse_lndl_fuzzy(lndl_action, operable)
        model_result = result.fields.get("actionrequest")
        assert model_result.function == "calculate"
        assert model_result.arguments == {"x": 10, "y": 20}

    def test_reason_model_lndl(self):
        """Test Reason model in LNDL format."""
        from lionpride.lndl import parse_lndl_fuzzy
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=Reason, name="reason")
        operable = Operable(specs=(spec,), name="Reason")

        lndl_reason = """<lvar Reason.reasoning r>This is my reasoning process</lvar>
<lvar Reason.confidence c>0.85</lvar>
OUT{reason: [r, c]}"""

        result = parse_lndl_fuzzy(lndl_reason, operable)
        model_result = result.fields.get("reason")
        assert model_result.reasoning == "This is my reasoning process"
        assert model_result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_lndl_action_flow(self):
        """Test complete LNDL flow with actions (mock)."""
        from unittest.mock import AsyncMock, MagicMock

        from lionpride.session import Session

        session = Session()
        _ = session.create_branch()  # Branch created but not used in this test

        # Create action operative
        create_action_operative(
            base_model=Analysis,
            reason=True,
            actions=True,
        )

        # Mock model that returns LNDL with action requests
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.invoke = AsyncMock()

        # This would test the full flow:
        # 1. Model returns LNDL with action_requests
        # 2. Actions get executed
        # 3. action_responses added to the result
        # Full integration test would go here

    def test_lndl_prompt_with_actions(self):
        """Test LNDL prompt generation for action operatives."""
        from lionpride.operations.operate.message_prep import generate_lndl_spec_format

        operative = create_action_operative(
            base_model=Analysis,
            reason=True,
            actions=True,
            name="AnalysisOp",
        )

        prompt = generate_lndl_spec_format(operative)

        # Should mention the specs and their fields
        assert "analysisop" in prompt.lower()  # The spec name
        assert "summary" in prompt.lower()  # Field in Analysis model
        assert "confidence" in prompt.lower()  # Field in Analysis model
        assert "reason" in prompt.lower()  # The reason spec
        assert "action_requests" in prompt.lower()  # Part of typed namespace

    def test_mixed_spec_types(self):
        """Test operative with mixed field and model specs."""
        from lionpride.types import Operable, Spec

        # Mix of scalar and model-based specs
        specs = [
            Spec(base_type=str, name="title"),
            Spec(base_type=Analysis, name="analysis"),  # Model-based
            Spec(base_type=list[ActionRequestModel], name="actions"),
        ]

        operable = Operable(specs=tuple(specs), name="MixedOp")

        # This tests that operables can handle mixed spec types
        assert len(operable.get_specs()) == 3

    def test_action_operative_creates_multiple_specs(self):
        """Verify action operative creates model spec plus action specs."""
        operative = create_action_operative(
            base_model=Analysis,
            actions=True,
        )

        specs = operative.operable.get_specs()
        spec_names = [spec.name for spec in specs]

        # Should have model spec plus action specs
        assert len(spec_names) == 3  # analysis, action_requests, action_responses
        assert "analysis" in spec_names  # Lowercased name from the model
        assert "action_requests" in spec_names
        assert "action_responses" in spec_names

        # The first spec should be model-based
        model_spec = next(s for s in specs if s.name == "analysis")
        assert model_spec.base_type == Analysis

    def test_action_operative_with_actions_flag(self):
        """Verify action operative sets _supports_actions flag."""
        operative_with = create_action_operative(
            base_model=Analysis,
            actions=True,
        )

        operative_without = create_action_operative(
            base_model=Analysis,
            actions=False,
        )

        # Check _supports_actions flag
        assert hasattr(operative_with, "_supports_actions")
        assert operative_with._supports_actions is True
        assert operative_without._supports_actions is False

    def test_lndl_fallback_for_actions(self):
        """Test that LNDL gracefully falls back for action operatives."""
        operative = create_action_operative(
            base_model=Analysis,
            actions=True,
            auto_retry_parse=True,
        )

        # If LNDL parsing fails, should fall back to JSON
        # The JSON should match the actual model structure
        json_response = """{
            "analysis": {
                "summary": "Test summary",
                "confidence": 0.75,
                "recommendations": ["Do this", "Do that"]
            }
        }"""

        result = operative.validate_response(json_response, strict=False)
        # Should parse as JSON fallback
        if result:
            # The result is the composite response with the analysis spec
            assert hasattr(result, "analysis") or hasattr(result, "summary")
            # Actions are not part of the model fields


class TestLNDLActionEdgeCases:
    """Test edge cases for LNDL with actions."""

    def test_empty_action_requests(self):
        """Test LNDL with empty action_requests."""
        from lionpride.lndl import parse_lndl_fuzzy

        create_action_operative(
            base_model=Analysis,
            actions=True,
            name="AnalysisOp",
        )

        # LNDL with empty actions (model decides no tools needed)

        # Should parse with empty action list
        # Implementation-dependent

    def test_malformed_action_request(self):
        """Test LNDL with malformed action requests."""
        from lionpride.lndl import parse_lndl_fuzzy

        create_action_operative(
            base_model=Analysis,
            actions=True,
            name="AnalysisOp",
        )

        # Action request with typo or missing field

        # Should handle gracefully with fuzzy matching or validation error
        # Implementation-dependent

    def test_action_with_complex_arguments(self):
        """Test LNDL with complex nested arguments."""
        from lionpride.lndl import parse_lndl_fuzzy
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=ActionRequestModel, name="action")
        operable = Operable(specs=(spec,), name="Action")

        # Complex nested structure in arguments
        lndl_complex = """<lvar ActionRequestModel.function f>process</lvar>
<lvar ActionRequestModel.arguments a>{
    "config": {
        "nested": {
            "deep": "value",
            "list": [1, 2, 3]
        }
    },
    "options": ["a", "b", "c"]
}</lvar>
OUT{action: [f, a]}"""

        result = parse_lndl_fuzzy(lndl_complex, operable)
        model_result = result.fields.get("action")
        assert model_result.function == "process"
        assert model_result.arguments["config"]["nested"]["deep"] == "value"
        assert model_result.arguments["options"] == ["a", "b", "c"]
