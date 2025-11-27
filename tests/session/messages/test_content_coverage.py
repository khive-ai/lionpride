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

from lionpride.session.messages._utils import _validate_image_url
from lionpride.session.messages.content import (
    InstructionContent,
    MessageContent,
)

# =============================================================================
# Edge Case Tests - URL Validation Exception Handling (Lines 47-48)
# =============================================================================


class TestValidateImageUrlExceptionHandling:
    """Test exception handling in _validate_image_url (lines 47-48)."""

    def test_validate_when_urlparse_raises_then_wraps_exception(self):
        """Test _validate_image_url wraps urlparse exceptions (line 47-48)."""
        # urlparse is very tolerant, but we can trigger exception with mock
        with patch("lionpride.session.messages._utils.urlparse") as mock_parse:
            mock_parse.side_effect = ValueError("Malformed URL")

            with pytest.raises(ValueError, match=r"Malformed image URL.*Malformed URL"):
                _validate_image_url("http://example.com")


# =============================================================================
# Edge Case Tests - MessageContent chat_msg Exception (Lines 92-95)
# =============================================================================


class TestMessageContentToChatException:
    """Test to_chat exception handling (lines 92-95)."""

    def test_to_chat_when_render_raises_then_returns_none(self):
        """Test to_chat returns None when render raises exception (lines 92-95)."""

        class BrokenContent(MessageContent):
            """Content with broken render method."""

            def render(self, *args, **kwargs):
                raise RuntimeError("Intentional error")

            @classmethod
            def from_dict(cls, data):
                return cls()

        content = BrokenContent()
        assert content.to_chat() is None


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

        rendered = content.render()
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

        rendered = content.render()
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

        rendered = content.render()
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
            instruction="Generate nested", request_model=ResponseWithNested
        )

        rendered = content.render()
        assert "NestedType" in rendered
        assert "## ResponseFormat" in rendered


# =============================================================================
# Edge Case Tests - Response Format Generation
# =============================================================================


class TestResponseFormatGeneration:
    """Test response format generation with request_model."""

    def test_rendered_when_request_model_has_string_field(self):
        """Test string field is included in response format."""

        class StringModel(BaseModel):
            name: str = Field(..., description="Name")

        content = InstructionContent.create(instruction="Generate", request_model=StringModel)

        rendered = content.render()
        assert "## ResponseFormat" in rendered
        assert "name" in rendered

    def test_rendered_when_request_model_has_list_field(self):
        """Test list field is included in response format."""

        class ListModel(BaseModel):
            tags: list[str] = Field(..., description="Tags")

        content = InstructionContent.create(instruction="Generate", request_model=ListModel)

        rendered = content.render()
        assert "## ResponseFormat" in rendered
        assert "tags" in rendered

    def test_rendered_when_request_model_has_nested_model(self):
        """Test nested model is included in response format."""

        class NestedItem(BaseModel):
            item_name: str = Field(..., description="Item name")

        class ObjectArrayModel(BaseModel):
            items: list[NestedItem] = Field(..., description="Items")

        content = InstructionContent.create(instruction="Generate", request_model=ObjectArrayModel)

        rendered = content.render()
        assert "## ResponseFormat" in rendered
        assert "items" in rendered
