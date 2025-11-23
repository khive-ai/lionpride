# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests to achieve 95%+ coverage for MessageContent.

Target missing lines (87% → 95%+):
    - Lines 47-48: Exception handling in _validate_image_url
    - Lines 92-95: Exception handling in chat_msg property
    - Line 188: Empty params in dict tool schema
    - Lines 206-209: Nested types ($defs) in response_model
    - Line 221: bytes-to-utf8 decode in example_json
    - Lines 230-232: Exception fallback in response format
    - Lines 249-263: _create_example_from_schema for all field types
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from lionpride.session.messages.content import (
    InstructionContent,
    MessageContent,
    _validate_image_url,
)

# =============================================================================
# Edge Case Tests - URL Validation Exception Handling (Lines 47-48)
# =============================================================================


class TestValidateImageUrlExceptionHandling:
    """Test exception handling in _validate_image_url (lines 47-48)."""

    def test_validate_when_urlparse_raises_then_wraps_exception(self):
        """Test _validate_image_url wraps urlparse exceptions (line 47-48)."""
        # urlparse is very tolerant, but we can trigger exception with mock
        with patch("lionpride.session.messages.content.urlparse") as mock_parse:
            mock_parse.side_effect = ValueError("Malformed URL")

            with pytest.raises(ValueError, match=r"Malformed image URL.*Malformed URL"):
                _validate_image_url("http://example.com")


# =============================================================================
# Edge Case Tests - MessageContent chat_msg Exception (Lines 92-95)
# =============================================================================


class TestMessageContentChatMsgException:
    """Test chat_msg exception handling (lines 92-95)."""

    def test_chat_msg_when_rendered_raises_then_returns_none(self):
        """Test chat_msg returns None when rendered raises exception (lines 92-95)."""

        class BrokenContent(MessageContent):
            """Content with broken rendered property."""

            @property
            def rendered(self):
                raise RuntimeError("Intentional error")

            @classmethod
            def from_dict(cls, data):
                return cls()

        content = BrokenContent()
        assert content.chat_msg is None


# =============================================================================
# Edge Case Tests - InstructionContent Tool Schema Empty Params (Line 188)
# =============================================================================


class TestInstructionContentEmptyToolParams:
    """Test tool schema with empty/missing params (line 188)."""

    def test_rendered_when_tool_dict_no_params_then_uses_description(self):
        """Test tool schema dict with no params uses description only (line 188)."""
        tool_no_params = {
            "name": "simple_tool",
            "description": "A simple tool with no parameters",
            # No "parameters" key
        }

        content = InstructionContent.create(instruction="Use tool", tool_schemas=[tool_no_params])

        rendered = content.rendered
        assert "Tools:" in rendered
        assert "simple_tool" in rendered
        assert "A simple tool with no parameters" in rendered

    def test_rendered_when_tool_dict_empty_params_then_uses_description(self):
        """Test tool schema dict with empty params uses description only (line 188)."""
        tool_empty_params = {
            "name": "empty_params_tool",
            "description": "Tool with empty params dict",
            "parameters": {},  # Empty dict
        }

        content = InstructionContent.create(
            instruction="Use tool", tool_schemas=[tool_empty_params]
        )

        rendered = content.rendered
        assert "Tools:" in rendered
        assert "empty_params_tool" in rendered

    def test_rendered_when_tool_dict_params_no_properties_then_uses_description(self):
        """Test tool schema dict with params but no properties (line 188)."""
        tool_no_properties = {
            "name": "no_props_tool",
            "description": "Tool with params but no properties",
            "parameters": {
                "type": "object"
                # No "properties" key
            },
        }

        content = InstructionContent.create(
            instruction="Use tool", tool_schemas=[tool_no_properties]
        )

        rendered = content.rendered
        assert "Tools:" in rendered
        assert "no_props_tool" in rendered


# =============================================================================
# Edge Case Tests - Response Model Nested Types (Lines 206-209)
# =============================================================================


class NestedType(BaseModel):
    """Nested type for testing $defs."""

    inner_field: str = Field(..., description="Inner field")


class ResponseWithNested(BaseModel):
    """Response model with $defs (nested types)."""

    outer_field: str = Field(..., description="Outer field")
    nested_data: NestedType = Field(..., description="Nested data")


class TestInstructionContentNestedTypes:
    """Test response_model with nested types ($defs) (lines 206-209)."""

    def test_rendered_when_response_model_has_defs_then_includes_nested_types(self):
        """Test response_model with $defs includes nested types section (lines 206-209)."""
        content = InstructionContent.create(
            instruction="Generate nested", response_model=ResponseWithNested
        )

        rendered = content.rendered
        assert "Output Types:" in rendered
        assert "Nested Types:" in rendered
        assert "NestedType" in rendered


# =============================================================================
# Edge Case Tests - Example JSON Bytes Decode (Line 221)
# =============================================================================


class TestInstructionContentBytesDecoding:
    """Test example_json bytes decoding (line 221)."""

    def test_rendered_when_json_dumps_returns_bytes_then_decodes(self):
        """Test example_json decoding when ln.json_dumps returns bytes (line 221)."""

        class SimpleResponse(BaseModel):
            answer: str = Field(..., description="The answer")

        # Need to patch where json_dumps is used, not where it's imported from
        with patch("lionpride.ln.json_dumps") as mock_dumps:
            # Mock json_dumps to return bytes instead of str
            mock_dumps.return_value = b'{"answer":"..."}'

            content = InstructionContent.create(
                instruction="Generate", response_model=SimpleResponse
            )

            rendered = content.rendered
            assert "ResponseFormat:" in rendered
            assert '"answer":"..."' in rendered or '"answer": "..."' in rendered


# =============================================================================
# Edge Case Tests - Response Format Exception Fallback (Lines 230-232)
# =============================================================================


class TestInstructionContentResponseFormatFallback:
    """Test response format exception fallback (lines 230-232)."""

    def test_rendered_when_example_generation_fails_then_uses_fallback(self):
        """Test fallback response format when example generation fails (lines 230-232)."""

        class SimpleResponse(BaseModel):
            answer: str = Field(..., description="The answer")

        # Mock ln.json_dumps to raise exception
        with patch("lionpride.ln.json_dumps") as mock_dumps:
            mock_dumps.side_effect = RuntimeError("JSON encoding failed")

            content = InstructionContent.create(
                instruction="Generate", response_model=SimpleResponse
            )

            rendered = content.rendered
            assert "ResponseFormat:" in rendered
            assert "MUST RETURN VALID JSON matching the Output Types above" in rendered
            # Should not contain example structure
            assert "Example structure" not in rendered


# =============================================================================
# Edge Case Tests - _create_example_from_schema All Field Types (Lines 249-263)
# =============================================================================


class ComplexResponseModel(BaseModel):
    """Complex model covering all field types in _create_example_from_schema."""

    str_field: str = Field(..., description="String field")
    int_field: int = Field(..., description="Integer field")
    num_field: float = Field(..., description="Number field")
    bool_field: bool = Field(..., description="Boolean field")
    str_array: list[str] = Field(..., description="String array")
    obj_array: list[dict] = Field(..., description="Object array")
    simple_array: list = Field(..., description="Generic array")
    nested_obj: dict = Field(..., description="Nested object")
    unknown_type: str = Field(..., description="Unknown type field")


class TestCreateExampleFromSchema:
    """Test _create_example_from_schema for all field types (lines 249-263)."""

    def test_rendered_when_response_model_has_string_then_example_has_ellipsis(self):
        """Test string field generates '...' in example (line 246)."""

        class StringModel(BaseModel):
            name: str = Field(..., description="Name")

        content = InstructionContent.create(instruction="Generate", response_model=StringModel)

        rendered = content.rendered
        # JSON may be compact without spaces
        assert '"name":"..."' in rendered or '"name": "..."' in rendered

    def test_rendered_when_response_model_has_integer_then_example_has_zero(self):
        """Test integer field generates 0 in example (line 248)."""

        class IntModel(BaseModel):
            count: int = Field(..., description="Count")

        content = InstructionContent.create(instruction="Generate", response_model=IntModel)

        rendered = content.rendered
        # JSON may be compact without spaces
        assert '"count":0' in rendered or '"count": 0' in rendered

    def test_rendered_when_response_model_has_number_then_example_has_zero(self):
        """Test number field generates 0 in example (line 248)."""

        class NumberModel(BaseModel):
            score: float = Field(..., description="Score")

        content = InstructionContent.create(instruction="Generate", response_model=NumberModel)

        rendered = content.rendered
        # JSON may be compact without spaces
        assert '"score":0' in rendered or '"score": 0' in rendered

    def test_rendered_when_response_model_has_boolean_then_example_has_false(self):
        """Test boolean field generates false in example (line 250)."""

        class BoolModel(BaseModel):
            active: bool = Field(..., description="Active")

        content = InstructionContent.create(instruction="Generate", response_model=BoolModel)

        rendered = content.rendered
        # JSON may be compact without spaces, and Python may use False or false
        assert (
            '"active":false' in rendered
            or '"active": false' in rendered
            or '"active":False' in rendered
            or '"active": False' in rendered
        )

    def test_rendered_when_response_model_has_string_array_then_example_has_array(self):
        """Test string array field generates ["..."] in example (lines 252-255)."""

        class StringArrayModel(BaseModel):
            tags: list[str] = Field(..., description="Tags")

        content = InstructionContent.create(instruction="Generate", response_model=StringArrayModel)

        rendered = content.rendered
        # JSON may be compact without spaces
        assert (
            '"tags":["..."]' in rendered
            or '"tags": ["..."]' in rendered
            or '"tags":\n    - "..."' in rendered
        )

    def test_rendered_when_response_model_has_object_array_then_example_has_nested(self):
        """Test object array field generates nested example (lines 256-257)."""

        class NestedItem(BaseModel):
            item_name: str = Field(..., description="Item name")

        class ObjectArrayModel(BaseModel):
            items: list[NestedItem] = Field(..., description="Items")

        content = InstructionContent.create(instruction="Generate", response_model=ObjectArrayModel)

        rendered = content.rendered
        # Should have recursive call to _create_example_from_schema
        assert "items" in rendered

    def test_rendered_when_response_model_has_generic_array_then_example_has_empty(self):
        """Test generic array field generates [] in example (lines 258-259)."""

        # This is harder to test directly via Pydantic, but we can mock the schema
        content = InstructionContent.create(instruction="Generate")

        # Test _create_example_from_schema directly
        schema = {
            "properties": {
                "data": {
                    "type": "array",
                    "items": {},  # No type specified
                }
            }
        }

        example = content._create_example_from_schema(schema)
        assert example == {"data": []}

    def test_rendered_when_response_model_has_object_then_example_has_nested_dict(self):
        """Test object field generates nested dict in example (lines 260-261)."""

        class NestedObject(BaseModel):
            nested_field: str = Field(..., description="Nested")

        class ObjectModel(BaseModel):
            metadata: dict = Field(..., description="Metadata object")

        # Test via direct schema manipulation
        content = InstructionContent.create(instruction="Generate")

        schema = {
            "properties": {
                "config": {"type": "object", "properties": {"setting": {"type": "string"}}}
            }
        }

        example = content._create_example_from_schema(schema)
        assert "config" in example
        assert isinstance(example["config"], dict)

    def test_rendered_when_response_model_has_unknown_type_then_example_has_null(self):
        """Test unknown type field generates None in example (lines 262-263)."""

        content = InstructionContent.create(instruction="Generate")

        schema = {
            "properties": {
                "unknown": {
                    "type": "unknown_type"  # Not a recognized type
                }
            }
        }

        example = content._create_example_from_schema(schema)
        assert example == {"unknown": None}

    def test_rendered_when_complex_model_then_all_types_covered(self):
        """Test complex model with all field types (comprehensive coverage of lines 249-263)."""

        content = InstructionContent.create(
            instruction="Generate complex", response_model=ComplexResponseModel
        )

        rendered = content.rendered
        # Verify rendering succeeds with complex schema
        assert "ResponseFormat:" in rendered
        assert "Example structure:" in rendered
