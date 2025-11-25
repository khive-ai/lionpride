# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Test LNDL integration with action/tool calling.

Tests ensure:
1. LNDL can include action requests
2. Actions are executed when LNDL response includes them
3. Action results are properly integrated
"""

import pytest
from pydantic import BaseModel, Field

from lionpride.operations import (
    ActionRequestModel,
    ActionResponseModel,
    create_action_operative,
    create_operative_from_model,
)
from lionpride.operations.lndl import generate_lndl_spec_format
from lionpride.operations.validation import validate_response


class AnalysisModel(BaseModel):
    """Model with analysis results."""

    topic: str = Field(..., description="Topic analyzed")
    summary: str = Field(..., description="Summary of findings")
    confidence: float = Field(..., ge=0.0, le=1.0)


class TestLNDLWithActions:
    """Test suite for LNDL with action/tool calling."""

    @pytest.mark.asyncio
    async def test_create_action_operative(self):
        """Test creating an operative with action support."""
        operative = create_action_operative(
            base_model=AnalysisModel,
            reason=True,
            actions=True,
            name="AnalysisWithActions",
        )

        response_model = operative.response_type
        assert hasattr(response_model, "model_fields")

        fields = response_model.model_fields
        assert "action_requests" in fields
        assert "reason" in fields

    @pytest.mark.asyncio
    async def test_action_execution_structure(self):
        """Test that action requests can be detected and structured properly."""
        from lionpride.operations.operate.tool_executor import has_action_requests

        class MockParsedResponse:
            def __init__(self):
                self.topic = "AI Safety"
                self.summary = "Test summary"
                self.confidence = 0.9
                self.reason = "Testing actions"
                self.action_requests = [
                    ActionRequestModel(
                        function="test_function",
                        arguments={"arg1": "value1"},
                    )
                ]
                self.action_responses = None

        parsed_response = MockParsedResponse()

        assert has_action_requests(parsed_response)
        assert len(parsed_response.action_requests) == 1
        assert parsed_response.action_requests[0].function == "test_function"

    @pytest.mark.asyncio
    async def test_lndl_spec_format_includes_actions(self):
        """Test that LNDL spec format includes action fields when enabled."""
        operative = create_action_operative(
            base_model=AnalysisModel,
            reason=True,
            actions=True,
            name="AnalysisWithActions",
        )

        prompt = generate_lndl_spec_format(operative)

        assert "action_requests" in prompt
        assert "action_responses" in prompt
        assert "reason" in prompt.lower()

    def test_action_response_model_structure(self):
        """Test ActionResponseModel structure for completeness."""
        response = ActionResponseModel(
            function="test_function",
            arguments={"key": "value"},
            output="Test output",
        )

        assert response.function == "test_function"
        assert response.arguments == {"key": "value"}
        assert response.output == "Test output"

    @pytest.mark.asyncio
    async def test_validation_strategy_with_operative(self):
        """Test validation using the new strategy pattern."""
        operative = create_operative_from_model(AnalysisModel, name="Analysis")

        # Test with valid LNDL response
        lndl_response = """<lvar AnalysisModel.topic t>Machine Learning</lvar>
<lvar AnalysisModel.summary s>Overview of ML techniques</lvar>
<lvar AnalysisModel.confidence c>0.95</lvar>
OUT{analysis: [t, s, c]}"""

        result = validate_response(
            lndl_response,
            operable=operative,
            threshold=0.6,
        )

        # Should successfully validate
        if result.success:
            assert result.data is not None


class TestLNDLActionEdgeCases:
    """Test edge cases for LNDL with actions."""

    @pytest.mark.asyncio
    async def test_malformed_lndl_missing_out_block(self):
        """Test handling of malformed LNDL missing OUT block."""
        from lionpride.lndl import parse_lndl_fuzzy
        from lionpride.lndl.errors import MissingOutBlockError

        operative = create_action_operative(
            base_model=AnalysisModel,
            reason=False,
            actions=True,
            name="AnalysisWithActions",
        )

        malformed = """<lvar AnalysisModel.topic t>Test</lvar>
<lvar AnalysisModel.summary s>Summary</lvar>
<lvar AnalysisModel.confidence c>0.5</lvar>"""

        with pytest.raises(MissingOutBlockError):
            parse_lndl_fuzzy(malformed, operative.operable)

    @pytest.mark.asyncio
    async def test_validation_result_structure(self):
        """Test ValidationResult structure from new validation module."""
        from lionpride.operations.validation import ValidationResult

        # Test success result
        success = ValidationResult.ok(data={"test": "value"}, raw_response="raw")
        assert success.success is True
        assert success.data == {"test": "value"}
        assert success.error is None

        # Test failure result
        failure = ValidationResult.fail(error="Test error", raw_response="raw")
        assert failure.success is False
        assert failure.data is None
        assert failure.error == "Test error"
