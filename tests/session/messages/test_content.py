# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Message Content types - 85%+ coverage target.

Test Surface:
    - SystemContent (system_message, system_datetime, datetime_factory, rendered)
    - InstructionContent (instruction, context, tool_schemas, response_model, images, rendered)
    - AssistantResponseContent (assistant_response, rendered)
    - ActionRequestContent (function, arguments, rendered)
    - ActionResponseContent (request_id, result, error, success property, rendered)
    - All .create() factories
    - All .from_dict() classmethods
    - Edge cases (empty strings, None values, Unset sentinels)
"""

import pytest
from pydantic import BaseModel, Field

from lionpride.session.messages.content import (
    ActionRequestContent,
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    SystemContent,
)

# =============================================================================
# Test Helpers & Fixtures
# =============================================================================


class SampleToolModel(BaseModel):
    """Sample Pydantic model for tool schema testing."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Max results")


class SampleResponseModel(BaseModel):
    """Sample Pydantic model for response model testing."""

    answer: str = Field(..., description="The answer")
    confidence: float = Field(..., description="Confidence score")


@pytest.fixture
def sample_tool_dict():
    """Sample tool schema as dict."""
    return {
        "name": "search",
        "description": "Search the database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        },
    }


# =============================================================================
# SystemContent Tests
# =============================================================================


class TestSystemContent:
    """Tests for SystemContent."""

    def test_create_when_minimal_then_unset_fields(self):
        """Test create with no arguments results in all Unset fields."""
        content = SystemContent.create()
        assert SystemContent._is_sentinel(content.system_message)
        assert SystemContent._is_sentinel(content.system_datetime)
        assert SystemContent._is_sentinel(content.datetime_factory)

    def test_create_when_system_message_then_set(self):
        """Test create with system_message sets the field."""
        content = SystemContent.create(system_message="Test system message")
        assert content.system_message == "Test system message"

    def test_create_when_system_datetime_true_then_set(self):
        """Test create with system_datetime=True sets the field."""
        content = SystemContent.create(system_datetime=True)
        assert content.system_datetime is True

    def test_create_when_system_datetime_string_then_set(self):
        """Test create with system_datetime as string sets the field."""
        timestamp = "2025-11-20T10:00:00Z"
        content = SystemContent.create(system_datetime=timestamp)
        assert content.system_datetime == timestamp

    def test_create_when_datetime_factory_then_set(self):
        """Test create with datetime_factory sets the field."""

        def factory():
            return "2025-11-20T10:00:00Z"

        content = SystemContent.create(datetime_factory=factory)
        assert content.datetime_factory is factory

    def test_create_when_both_datetime_then_error(self):
        """Test create with both system_datetime and datetime_factory raises error."""
        with pytest.raises(ValueError, match="Cannot set both"):
            SystemContent.create(system_datetime=True, datetime_factory=lambda: "test")

    def test_rendered_when_minimal_then_empty(self):
        """Test rendered with no fields returns empty string."""
        content = SystemContent.create()
        assert content.render() == ""

    def test_rendered_when_system_message_then_message(self):
        """Test rendered with only system_message returns message."""
        content = SystemContent.create(system_message="Test message")
        assert content.render() == "Test message"

    def test_rendered_when_system_datetime_true_then_timestamp(self):
        """Test rendered with system_datetime=True includes ISO timestamp."""
        content = SystemContent.create(system_datetime=True)
        rendered = content.render()
        assert "System Time:" in rendered
        assert "T" in rendered  # ISO format has 'T'

    def test_rendered_when_system_datetime_string_then_custom_timestamp(self):
        """Test rendered with system_datetime as string uses custom timestamp."""
        timestamp = "2025-11-20T10:00:00Z"
        content = SystemContent.create(system_datetime=timestamp)
        assert content.render() == f"System Time: {timestamp}"

    def test_rendered_when_datetime_factory_then_calls_factory(self):
        """Test rendered with datetime_factory calls the factory."""

        def factory():
            return "custom-timestamp"

        content = SystemContent.create(datetime_factory=factory)
        assert "System Time: custom-timestamp" in content.render()

    def test_rendered_when_message_and_datetime_then_both(self):
        """Test rendered with both message and datetime includes both."""
        content = SystemContent.create(
            system_message="Test message", system_datetime="2025-11-20T10:00:00Z"
        )
        rendered = content.render()
        assert "System Time: 2025-11-20T10:00:00Z" in rendered
        assert "Test message" in rendered

    def test_from_dict_when_valid_then_creates(self):
        """Test from_dict with valid data creates instance."""
        data = {"system_message": "Test", "system_datetime": "2025-11-20T10:00:00Z"}
        content = SystemContent.from_dict(data)
        assert content.system_message == "Test"
        assert content.system_datetime == "2025-11-20T10:00:00Z"


# =============================================================================
# InstructionContent Tests
# =============================================================================


class TestInstructionContent:
    """Tests for InstructionContent."""

    def test_create_when_minimal_then_unset_fields(self):
        """Test create with no arguments results in all Unset fields."""
        content = InstructionContent.create()
        assert InstructionContent._is_sentinel(content.instruction)
        assert InstructionContent._is_sentinel(content.context)

    def test_create_when_instruction_then_set(self):
        """Test create with instruction sets the field."""
        content = InstructionContent.create(instruction="Do something")
        assert content.instruction == "Do something"

    def test_create_when_context_then_set(self):
        """Test create with context sets the field."""
        context = [{"key": "value"}]
        content = InstructionContent.create(context=context)
        assert content.context == context

    def test_create_when_tool_schemas_pydantic_then_set(self):
        """Test create with Pydantic tool schemas sets the field."""
        content = InstructionContent.create(tool_schemas=[SampleToolModel])
        assert content.tool_schemas == [SampleToolModel]

    def test_create_when_tool_schemas_dict_then_set(self, sample_tool_dict):
        """Test create with dict tool schemas sets the field."""
        content = InstructionContent.create(tool_schemas=[sample_tool_dict])
        assert content.tool_schemas == [sample_tool_dict]

    def test_create_when_response_model_then_set(self):
        """Test create with response_model sets the field."""
        content = InstructionContent.create(request_model=SampleResponseModel)
        assert content.request_model is SampleResponseModel

    def test_create_when_images_then_set(self):
        """Test create with images sets the field."""
        images = ["https://example.com/image.png"]
        content = InstructionContent.create(images=images)
        assert content.images == images

    def test_create_when_image_detail_then_set(self):
        """Test create with image_detail sets the field."""
        content = InstructionContent.create(image_detail="high")
        assert content.image_detail == "high"

    def test_rendered_when_instruction_only_then_yaml(self):
        """Test rendered with only instruction returns YAML format."""
        content = InstructionContent.create(instruction="Test instruction")
        rendered = content.render()
        assert isinstance(rendered, str)
        assert "Instruction:" in rendered

    def test_rendered_when_context_then_yaml_includes_context(self):
        """Test rendered with context includes context in YAML."""
        content = InstructionContent.create(instruction="Test", context=[{"key": "value"}])
        rendered = content.render()
        assert "Context:" in rendered

    def test_rendered_when_tool_schemas_string_then_included(self):
        """Test rendered with string tool schemas includes them."""
        tool_schema_str = "interface SampleToolModel { query: string; limit: number; }"
        content = InstructionContent.create(instruction="Use tools", tool_schemas=[tool_schema_str])
        rendered = content.render()
        assert "Tools:" in rendered
        assert "SampleToolModel" in rendered

    def test_rendered_when_tool_schemas_dict_then_typescript(self, sample_tool_dict):
        """Test rendered with dict tool schemas includes TypeScript schema."""
        content = InstructionContent.create(
            instruction="Use tools", tool_schemas=[sample_tool_dict]
        )
        rendered = content.render()
        assert "Tools:" in rendered
        assert "search" in rendered

    def test_rendered_when_request_model_then_includes_schema(self):
        """Test rendered with request_model includes response format."""
        content = InstructionContent.create(
            instruction="Generate", request_model=SampleResponseModel
        )
        rendered = content.render()
        assert "## ResponseFormat" in rendered
        assert "MUST RETURN VALID JSON" in rendered

    def test_rendered_when_images_then_returns_list(self):
        """Test rendered with images returns list of content blocks."""
        content = InstructionContent.create(
            instruction="Analyze", images=["https://example.com/image.png"]
        )
        rendered = content.render()
        assert isinstance(rendered, list)
        assert any(block.get("type") == "text" for block in rendered)
        assert any(block.get("type") == "image_url" for block in rendered)

    def test_rendered_when_images_with_detail_then_includes_detail(self):
        """Test rendered with images and detail includes detail in image blocks."""
        content = InstructionContent.create(
            instruction="Analyze",
            images=["https://example.com/image.png"],
            image_detail="high",
        )
        rendered = content.render()
        assert isinstance(rendered, list)
        image_block = next(block for block in rendered if block.get("type") == "image_url")
        assert image_block["image_url"]["detail"] == "high"

    def test_from_dict_when_valid_then_creates(self):
        """Test from_dict with valid data creates instance."""
        data = {"instruction": "Test", "context": [{"key": "value"}]}
        content = InstructionContent.from_dict(data)
        assert content.instruction == "Test"
        assert content.context == [{"key": "value"}]


# =============================================================================
# InstructionContent Security Tests (BS-1: URL Validation)
# =============================================================================


class TestInstructionContentSecurity:
    """Security tests for InstructionContent image URL validation (BS-1)."""

    def test_create_when_http_url_then_accepts(self):
        """Test create accepts valid http:// URLs."""
        content = InstructionContent.create(
            instruction="Analyze", images=["http://example.com/image.png"]
        )
        assert content.images == ["http://example.com/image.png"]

    def test_create_when_https_url_then_accepts(self):
        """Test create accepts valid https:// URLs."""
        content = InstructionContent.create(
            instruction="Analyze", images=["https://example.com/image.png"]
        )
        assert content.images == ["https://example.com/image.png"]

    def test_create_when_multiple_valid_urls_then_accepts(self):
        """Test create accepts multiple valid URLs."""
        images = [
            "https://example.com/image1.png",
            "http://example.com/image2.jpg",
            "https://cdn.example.com/path/to/image.gif",
        ]
        content = InstructionContent.create(instruction="Analyze", images=images)
        assert content.images == images

    def test_create_when_file_url_then_rejects(self):
        """Test create rejects file:// URLs (local file access vulnerability)."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            InstructionContent.create(instruction="Analyze", images=["file:///etc/passwd"])

    def test_create_when_javascript_url_then_rejects(self):
        """Test create rejects javascript: URLs (XSS vulnerability)."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            InstructionContent.create(
                instruction="Analyze",
                images=["javascript:alert('XSS')"],
            )

    def test_create_when_data_url_then_rejects(self):
        """Test create rejects data:// URLs (DoS vulnerability via large images)."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            InstructionContent.create(
                instruction="Analyze",
                images=[
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                ],
            )

    def test_create_when_ftp_url_then_rejects(self):
        """Test create rejects ftp:// URLs (unauthorized scheme)."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            InstructionContent.create(instruction="Analyze", images=["ftp://example.com/image.png"])

    def test_create_when_malformed_url_missing_scheme_then_rejects(self):
        """Test create rejects malformed URLs without scheme."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            InstructionContent.create(instruction="Analyze", images=["example.com/image.png"])

    def test_create_when_malformed_url_missing_domain_then_rejects(self):
        """Test create rejects URLs missing domain (netloc)."""
        with pytest.raises(ValueError, match="missing domain"):
            InstructionContent.create(instruction="Analyze", images=["http://"])

    def test_create_when_empty_string_url_then_rejects(self):
        """Test create rejects empty string URLs."""
        with pytest.raises(ValueError, match="must be non-empty string"):
            InstructionContent.create(instruction="Analyze", images=[""])

    def test_create_when_non_string_url_then_rejects(self):
        """Test create rejects non-string URLs."""
        with pytest.raises(ValueError, match="must be non-empty string"):
            InstructionContent.create(instruction="Analyze", images=[123])  # type: ignore

    def test_create_when_mixed_valid_invalid_then_rejects(self):
        """Test create rejects if any URL in list is invalid."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            InstructionContent.create(
                instruction="Analyze",
                images=[
                    "https://example.com/valid.png",
                    "file:///etc/passwd",  # Invalid
                ],
            )

    def test_create_when_url_with_path_then_accepts(self):
        """Test create accepts URLs with complex paths."""
        content = InstructionContent.create(
            instruction="Analyze",
            images=["https://example.com/path/to/image.png?v=1&size=large"],
        )
        assert content.images == ["https://example.com/path/to/image.png?v=1&size=large"]

    def test_create_when_url_with_auth_then_accepts(self):
        """Test create accepts URLs with authentication."""
        content = InstructionContent.create(
            instruction="Analyze",
            images=["https://user:pass@example.com/image.png"],
        )
        assert content.images == ["https://user:pass@example.com/image.png"]

    def test_from_dict_when_invalid_url_then_rejects(self):
        """Test from_dict rejects invalid URLs (validates via create)."""
        data = {"instruction": "Test", "images": ["file:///etc/passwd"]}
        with pytest.raises(ValueError, match="must use http:// or https://"):
            InstructionContent.from_dict(data)


# =============================================================================
# AssistantResponseContent Tests
# =============================================================================


class TestAssistantResponseContent:
    """Tests for AssistantResponseContent."""

    def test_create_when_minimal_then_unset_field(self):
        """Test create with no arguments results in Unset field."""
        content = AssistantResponseContent.create()
        assert AssistantResponseContent._is_sentinel(content.assistant_response)

    def test_create_when_assistant_response_then_set(self):
        """Test create with assistant_response sets the field."""
        content = AssistantResponseContent.create(assistant_response="Test response")
        assert content.assistant_response == "Test response"

    def test_rendered_when_unset_then_empty_string(self):
        """Test rendered with Unset assistant_response returns empty string."""
        content = AssistantResponseContent.create()
        assert content.render() == ""

    def test_rendered_when_set_then_returns_response(self):
        """Test rendered with set assistant_response returns the response."""
        content = AssistantResponseContent.create(assistant_response="Test response")
        assert content.render() == "Test response"

    def test_from_dict_when_valid_then_creates(self):
        """Test from_dict with valid data creates instance."""
        data = {"assistant_response": "Test"}
        content = AssistantResponseContent.from_dict(data)
        assert content.assistant_response == "Test"

    def test_from_dict_when_none_then_creates_with_unset(self):
        """Test from_dict with None creates instance with Unset."""
        data = {"assistant_response": None}
        content = AssistantResponseContent.from_dict(data)
        # Should handle None gracefully (sentinel behavior)
        assert content.render() == ""


# =============================================================================
# ActionRequestContent Tests
# =============================================================================


class TestActionRequestContent:
    """Tests for ActionRequestContent."""

    def test_create_when_minimal_then_unset_fields(self):
        """Test create with no arguments results in Unset fields."""
        content = ActionRequestContent.create()
        assert ActionRequestContent._is_sentinel(content.function)
        assert ActionRequestContent._is_sentinel(content.arguments)

    def test_create_when_function_then_set(self):
        """Test create with function sets the field."""
        content = ActionRequestContent.create(function="search")
        assert content.function == "search"

    def test_create_when_arguments_then_set(self):
        """Test create with arguments sets the field."""
        args = {"query": "test", "limit": 10}
        content = ActionRequestContent.create(arguments=args)
        assert content.arguments == args

    def test_create_when_both_then_set(self):
        """Test create with both function and arguments sets both."""
        args = {"query": "test"}
        content = ActionRequestContent.create(function="search", arguments=args)
        assert content.function == "search"
        assert content.arguments == args

    def test_rendered_when_minimal_then_yaml_with_empty_args(self):
        """Test rendered with no function includes empty arguments."""
        content = ActionRequestContent.create()
        rendered = content.render()
        # minimal_yaml of {"arguments": {}} produces "{}\n"
        assert rendered == "{}\n"

    def test_rendered_when_function_then_yaml_includes_function(self):
        """Test rendered with function includes function in YAML."""
        content = ActionRequestContent.create(function="search")
        rendered = content.render()
        assert "function:" in rendered or "function: search" in rendered

    def test_rendered_when_arguments_then_yaml_includes_arguments(self):
        """Test rendered with arguments includes arguments in YAML."""
        args = {"query": "test", "limit": 10}
        content = ActionRequestContent.create(function="search", arguments=args)
        rendered = content.render()
        assert "arguments:" in rendered
        assert "query" in rendered or "test" in rendered

    def test_from_dict_when_valid_then_creates(self):
        """Test from_dict with valid data creates instance."""
        data = {"function": "search", "arguments": {"query": "test"}}
        content = ActionRequestContent.from_dict(data)
        assert content.function == "search"
        assert content.arguments == {"query": "test"}


# =============================================================================
# ActionResponseContent Tests
# =============================================================================


class TestActionResponseContent:
    """Tests for ActionResponseContent."""

    def test_create_when_minimal_then_unset_fields(self):
        """Test create with no arguments results in Unset fields."""
        content = ActionResponseContent.create()
        assert ActionResponseContent._is_sentinel(content.request_id)
        assert ActionResponseContent._is_sentinel(content.result)
        assert ActionResponseContent._is_sentinel(content.error)

    def test_create_when_request_id_then_set(self):
        """Test create with request_id sets the field."""
        content = ActionResponseContent.create(request_id="req-123")
        assert content.request_id == "req-123"

    def test_create_when_result_then_set(self):
        """Test create with result sets the field."""
        result = {"data": "test"}
        content = ActionResponseContent.create(result=result)
        assert content.result == result

    def test_create_when_error_then_set(self):
        """Test create with error sets the field."""
        content = ActionResponseContent.create(error="Something failed")
        assert content.error == "Something failed"

    def test_success_when_no_error_then_true(self):
        """Test success property is True when error is Unset."""
        content = ActionResponseContent.create(result={"data": "test"})
        assert content.success is True

    def test_success_when_error_then_false(self):
        """Test success property is False when error is set."""
        content = ActionResponseContent.create(error="Failed")
        assert content.success is False

    def test_rendered_when_success_then_includes_result(self):
        """Test rendered with success includes result in YAML."""
        content = ActionResponseContent.create(request_id="req-123", result={"data": "test"})
        rendered = content.render()
        assert "success: true" in rendered or "success: True" in rendered
        assert "result:" in rendered or "data" in rendered

    def test_rendered_when_error_then_includes_error(self):
        """Test rendered with error includes error in YAML."""
        content = ActionResponseContent.create(request_id="req-123", error="Something failed")
        rendered = content.render()
        assert "success: false" in rendered or "success: False" in rendered
        assert "error:" in rendered or "Something failed" in rendered

    def test_rendered_when_request_id_then_includes_id(self):
        """Test rendered includes request_id when set."""
        content = ActionResponseContent.create(request_id="req-123", result="ok")
        rendered = content.render()
        assert "request_id:" in rendered or "req-123" in rendered

    def test_from_dict_when_success_then_creates(self):
        """Test from_dict with success data creates instance."""
        data = {"request_id": "req-123", "result": {"data": "test"}}
        content = ActionResponseContent.from_dict(data)
        assert content.request_id == "req-123"
        assert content.result == {"data": "test"}
        assert content.success is True

    def test_from_dict_when_error_then_creates(self):
        """Test from_dict with error data creates instance."""
        data = {"request_id": "req-123", "error": "Failed"}
        content = ActionResponseContent.from_dict(data)
        assert content.request_id == "req-123"
        assert content.error == "Failed"
        assert content.success is False
