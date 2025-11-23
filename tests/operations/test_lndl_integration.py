"""Test LNDL (Language Network Directive Language) integration with operate.

Tests ensure:
1. LNDL format is correctly passed to model
2. Model responses in LNDL format are parsed correctly
3. Fuzzy matching tolerates field name variations
4. Fallback to JSON works when LNDL fails
5. Raw LNDL is stored in assistant messages
"""

import pytest
from pydantic import BaseModel, Field

from lionpride.operations import create_operative_from_model
from lionpride.session import Session
from lionpride.session.messages import AssistantResponseContent, InstructionContent, Message


class SampleModel(BaseModel):
    """Sample model for LNDL validation testing."""

    title: str = Field(..., description="Title of the item")
    score: float = Field(..., ge=0.0, le=100.0)
    tags: list[str] = Field(default_factory=list)
    is_valid: bool = Field(default=True)


class TestLNDLIntegration:
    """Test suite for LNDL integration."""

    def test_operative_creates_single_model_spec(self):
        """Test that create_operative_from_model creates a single model-based spec."""
        operative = create_operative_from_model(SampleModel, name="TestOp")

        # Should have single spec
        specs = operative.operable.get_specs()
        assert len(specs) == 1

        # Spec should have the model as base_type
        spec = specs[0]
        assert spec.base_type == SampleModel
        assert spec.name == "testop"  # lowercase for LNDL matching

    def test_operative_response_model_is_original(self):
        """Test that response_model returns original Pydantic model for single spec."""
        operative = create_operative_from_model(SampleModel, name="TestOp")

        response_model = operative.create_response_model()
        assert response_model is SampleModel  # Should be exact same class

    def test_instruction_content_with_lndl(self):
        """Test InstructionContent generates LNDL format instructions."""
        content = InstructionContent(
            instruction="Test instruction",
            response_model=SampleModel,
            use_lndl=True,
        )

        rendered = content.rendered
        assert "LNDL FORMAT" in rendered or "ResponseFormat" in rendered
        assert "SampleModel" in rendered
        assert "lvar" in rendered.lower()

    def test_instruction_content_without_lndl(self):
        """Test InstructionContent generates JSON format without use_lndl."""
        content = InstructionContent(
            instruction="Test instruction",
            response_model=SampleModel,
            use_lndl=False,
        )

        rendered = content.rendered
        assert "JSON" in rendered
        assert "lvar" not in rendered.lower()

    @pytest.mark.asyncio
    async def test_lndl_response_stored_raw(self):
        """Test that LNDL responses are stored raw in assistant messages."""
        # This would require mocking the model call, but demonstrates the test structure
        from unittest.mock import AsyncMock, MagicMock

        session = Session()
        session.create_branch()

        # Mock model
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.invoke = AsyncMock()

        # Mock LNDL response
        lndl_response = """<lvar SampleModel.title t>Test Title</lvar>
<lvar SampleModel.score s>85.5</lvar>
<lvar SampleModel.tags tg>["tag1", "tag2"]</lvar>
<lvar SampleModel.is_valid v>true</lvar>
OUT{testmodel: [t, s, tg, v]}"""

        mock_execution = MagicMock()
        mock_execution.status.value = "completed"
        mock_execution.response.data = lndl_response

        mock_calling = MagicMock()
        mock_calling.execution = mock_execution
        mock_model.invoke.return_value = mock_calling

        # Would need to test through operate_factory
        # This is a placeholder for the actual integration test

    def test_lndl_fuzzy_matching(self):
        """Test that LNDL fuzzy matching handles variations."""
        from lionpride.lndl import parse_lndl_fuzzy

        operative = create_operative_from_model(SampleModel, name="TestOp")

        # Test with correct field names
        lndl_correct = """<lvar SampleModel.title t>Title</lvar>
<lvar SampleModel.score s>75</lvar>
<lvar SampleModel.tags tg>["a", "b"]</lvar>
<lvar SampleModel.is_valid v>false</lvar>
OUT{testop: [t, s, tg, v]}"""

        result = parse_lndl_fuzzy(lndl_correct, operative.operable)
        # Result is LNDLOutput with fields['testop'] containing the model
        model_result = result.fields.get("testop")
        assert model_result.title == "Title"
        assert model_result.score == 75
        assert model_result.tags == ["a", "b"]
        assert model_result.is_valid is False

        # Test with typos in field names (fuzzy matching)
        lndl_typo = """<lvar SampleModel.titel t>Title</lvar>
<lvar SampleModel.scor s>75</lvar>
<lvar SampleModel.tag tg>["a", "b"]</lvar>
<lvar SampleModel.is_valide v>false</lvar>
OUT{testop: [t, s, tg, v]}"""

        result = parse_lndl_fuzzy(lndl_typo, operative.operable, threshold=0.8)
        model_result = result.fields.get("testop")
        assert model_result.title == "Title"
        assert model_result.score == 75

        # Test with Response suffix (common LLM pattern)
        lndl_response = """<lvar SampleModelResponse.title t>Title</lvar>
<lvar SampleModelResponse.score s>75</lvar>
<lvar SampleModelResponse.tags tg>["a", "b"]</lvar>
<lvar SampleModelResponse.is_valid v>false</lvar>
OUT{testmodelresponse: [t, s, tg, v]}"""

        result = parse_lndl_fuzzy(lndl_response, operative.operable)
        # Note: Response suffix might map to 'testmodelresponse' spec
        model_result = result.fields.get("testmodelresponse") or result.fields.get("testop")
        assert model_result.title == "Title"

    def test_operative_validate_response_fallback(self):
        """Test Operative validation fallback when LNDL fails."""
        operative = create_operative_from_model(SampleModel, name="TestOp")

        # Test with valid JSON (fallback)
        json_response = '{"title": "Test", "score": 50.0, "tags": [], "is_valid": true}'
        result = operative.validate_response(json_response, strict=False)
        assert result.title == "Test"
        assert result.score == 50.0

        # Test with invalid response
        invalid_response = "Not valid LNDL or JSON"
        result = operative.validate_response(invalid_response, strict=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_lndl_with_missing_out_block(self):
        """Test LNDL parsing when OUT{} block is missing."""
        from lionpride.lndl import parse_lndl_fuzzy
        from lionpride.lndl.errors import MissingOutBlockError

        operative = create_operative_from_model(SampleModel, name="TestOp")

        # LNDL without OUT{} block (some models might omit it)
        lndl_no_out = """<lvar SampleModel.title t>Title</lvar>
<lvar SampleModel.score s>75</lvar>
<lvar SampleModel.tags tg>["a", "b"]</lvar>
<lvar SampleModel.is_valid v>false</lvar>"""

        # Should raise MissingOutBlockError
        with pytest.raises(MissingOutBlockError):
            parse_lndl_fuzzy(lndl_no_out, operative.operable)

    def test_lndl_prompt_generation(self):
        """Test LNDL prompt generation for different specs."""
        from lionpride.operations.operate.message_prep import generate_lndl_spec_format

        operative = create_operative_from_model(SampleModel, name="TestOp")

        prompt = generate_lndl_spec_format(operative)
        assert "YOUR TASK REQUIRES LNDL FORMAT" in prompt
        assert "SampleModel" in prompt
        assert "title(str)" in prompt or "title" in prompt
        assert "score(float)" in prompt or "score" in prompt
        assert "OUT" in prompt
        assert "fuzzy matching" in prompt.lower()

    def test_message_content_preservation(self):
        """Test that assistant messages preserve LNDL format."""
        lndl_text = """<lvar SampleModel.title t>Title</lvar>
<lvar SampleModel.score s>75</lvar>
OUT{testmodel: [t, s]}"""

        # Create assistant message with LNDL content
        content = AssistantResponseContent(assistant_response=lndl_text)
        msg = Message(content=content, sender="assistant", recipient="user")

        # Verify LNDL is preserved
        assert msg.content.rendered == lndl_text
        assert "<lvar" in msg.content.rendered


class TestLNDLEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_lndl(self):
        """Test handling of malformed LNDL."""
        from lionpride.lndl import parse_lndl_fuzzy
        from lionpride.lndl.parser import ParseError

        operative = create_operative_from_model(SampleModel, name="TestOp")

        # Missing closing tag should raise ParseError
        malformed = "<lvar SampleModel.title t>Title"
        with pytest.raises(ParseError):
            parse_lndl_fuzzy(malformed, operative.operable)

        # Invalid OUT block should raise MissingFieldError
        from lionpride.lndl.errors import MissingFieldError

        invalid_out = """<lvar SampleModel.title t>Title</lvar>
<lvar SampleModel.score s>75</lvar>
OUT{wrong_spec: [t, s]}"""
        with pytest.raises(MissingFieldError):
            parse_lndl_fuzzy(invalid_out, operative.operable)

    def test_mixed_json_lndl(self):
        """Test when model returns mixed JSON and LNDL."""
        operative = create_operative_from_model(SampleModel, name="TestOp")

        # Some models might mix formats but still need OUT block
        mixed = """Here's the result:
<lvar SampleModel.title t>Title</lvar>
<lvar SampleModel.score s>75</lvar>
And also: {"tags": ["test"]}
OUT{testop: [t, s]}"""

        # Should extract LNDL parts
        from lionpride.lndl import parse_lndl_fuzzy

        result = parse_lndl_fuzzy(mixed, operative.operable)
        model_result = result.fields.get("testop")
        assert model_result.title == "Title"
        assert model_result.score == 75
        # JSON parts are ignored in LNDL parsing

    def test_unicode_in_lndl(self):
        """Test LNDL with Unicode characters."""
        from lionpride.lndl import parse_lndl_fuzzy

        operative = create_operative_from_model(SampleModel, name="TestOp")

        lndl_unicode = """<lvar SampleModel.title t>测试标题 🦁</lvar>
<lvar SampleModel.score s>95.5</lvar>
<lvar SampleModel.tags tg>["中文", "emoji 🎯"]</lvar>
<lvar SampleModel.is_valid v>true</lvar>
OUT{testop: [t, s, tg, v]}"""

        result = parse_lndl_fuzzy(lndl_unicode, operative.operable)
        model_result = result.fields.get("testop")
        assert "测试标题" in model_result.title
        assert "🦁" in model_result.title
        assert "emoji 🎯" in model_result.tags
