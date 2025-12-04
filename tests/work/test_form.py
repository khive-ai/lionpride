# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Form and parse_assignment."""

import pytest
from pydantic import BaseModel

from lionpride.work import Form, FormResources, parse_assignment


class TestParseAssignment:
    """Tests for parse_assignment function."""

    def test_simple_assignment(self):
        """Test basic assignment without branch."""
        parsed = parse_assignment("a, b -> c")
        assert parsed.branch_name is None
        assert parsed.input_fields == ["a", "b"]
        assert parsed.output_fields == ["c"]

    def test_single_input_output(self):
        """Test single input to single output."""
        parsed = parse_assignment("input -> output")
        assert parsed.branch_name is None
        assert parsed.input_fields == ["input"]
        assert parsed.output_fields == ["output"]

    def test_multiple_outputs(self):
        """Test multiple outputs."""
        parsed = parse_assignment("a -> b, c, d")
        assert parsed.branch_name is None
        assert parsed.input_fields == ["a"]
        assert parsed.output_fields == ["b", "c", "d"]

    def test_no_inputs(self):
        """Test assignment with no inputs."""
        parsed = parse_assignment(" -> output")
        assert parsed.branch_name is None
        assert parsed.input_fields == []
        assert parsed.output_fields == ["output"]

    def test_with_branch_prefix(self):
        """Test assignment with branch prefix."""
        parsed = parse_assignment("orchestrator: a, b -> c")
        assert parsed.branch_name == "orchestrator"
        assert parsed.input_fields == ["a", "b"]
        assert parsed.output_fields == ["c"]

    def test_branch_with_spaces(self):
        """Test branch prefix with surrounding spaces."""
        parsed = parse_assignment("  planner  :  x, y  ->  z  ")
        assert parsed.branch_name == "planner"
        assert parsed.input_fields == ["x", "y"]
        assert parsed.output_fields == ["z"]

    def test_complex_branch_assignment(self):
        """Test complex assignment with branch and multiple fields."""
        parsed = parse_assignment("implementer: context, instruction, plan -> result, score")
        assert parsed.branch_name == "implementer"
        assert parsed.input_fields == ["context", "instruction", "plan"]
        assert parsed.output_fields == ["result", "score"]

    def test_invalid_no_arrow(self):
        """Test that missing arrow raises ValueError."""
        with pytest.raises(ValueError, match="Must contain '->'"):
            parse_assignment("a, b, c")

    def test_invalid_empty(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_assignment("")

    def test_invalid_no_outputs(self):
        """Test that missing outputs raises ValueError."""
        with pytest.raises(ValueError, match="at least one output"):
            parse_assignment("a, b -> ")

    def test_colon_after_arrow_not_branch(self):
        """Test that colon after arrow is not treated as branch."""
        parsed = parse_assignment("a -> b:c")
        assert parsed.branch_name is None
        assert parsed.input_fields == ["a"]
        assert parsed.output_fields == ["b:c"]


class TestParseAssignmentWithResources:
    """Tests for parse_assignment with resource declarations."""

    def test_single_api(self):
        """Test single api declaration."""
        parsed = parse_assignment("a -> b | api:gpt4")
        assert parsed.resources.api == "gpt4"
        assert parsed.resources.api_gen is None
        assert parsed.resources.tools is None

    def test_api_gen(self):
        """Test api_gen declaration."""
        parsed = parse_assignment("a -> b | api_gen:gpt5")
        assert parsed.resources.api_gen == "gpt5"
        assert parsed.resources.api is None

    def test_api_parse(self):
        """Test api_parse declaration."""
        parsed = parse_assignment("a -> b | api_parse:gpt4mini")
        assert parsed.resources.api_parse == "gpt4mini"

    def test_multiple_api_roles(self):
        """Test multiple api role declarations."""
        parsed = parse_assignment("a -> b | api_gen:gpt5, api_parse:gpt4mini")
        assert parsed.resources.api_gen == "gpt5"
        assert parsed.resources.api_parse == "gpt4mini"

    def test_single_tool(self):
        """Test single tool declaration."""
        parsed = parse_assignment("a -> b | tool:search")
        assert parsed.resources.tools == frozenset({"search"})

    def test_multiple_tools(self):
        """Test multiple tool declarations."""
        parsed = parse_assignment("a -> b | tool:search, tool:calendar")
        assert parsed.resources.tools == frozenset({"search", "calendar"})

    def test_tool_wildcard(self):
        """Test tool:* wildcard."""
        parsed = parse_assignment("a -> b | tool:*")
        assert parsed.resources.tools == "*"

    def test_api_and_tools(self):
        """Test api and tools together."""
        parsed = parse_assignment("a -> b | api:gpt4, tool:search, tool:calendar")
        assert parsed.resources.api == "gpt4"
        assert parsed.resources.tools == frozenset({"search", "calendar"})

    def test_full_resource_declaration(self):
        """Test full resource declaration with all components."""
        parsed = parse_assignment(
            "orchestrator: context -> plan | api_gen:gpt5, api_parse:gpt4mini, tool:*"
        )
        assert parsed.branch_name == "orchestrator"
        assert parsed.input_fields == ["context"]
        assert parsed.output_fields == ["plan"]
        assert parsed.resources.api_gen == "gpt5"
        assert parsed.resources.api_parse == "gpt4mini"
        assert parsed.resources.tools == "*"

    def test_invalid_resource_format(self):
        """Test invalid resource format raises error."""
        with pytest.raises(ValueError, match="Invalid resource format"):
            parse_assignment("a -> b | invalid")

    def test_invalid_resource_type(self):
        """Test invalid resource type raises error."""
        with pytest.raises(ValueError, match="Invalid resource type"):
            parse_assignment("a -> b | unknown:value")

    def test_empty_resource_name(self):
        """Test empty resource name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_assignment("a -> b | api:")

    def test_duplicate_api_raises_error(self):
        """Test duplicate api declaration raises error."""
        with pytest.raises(ValueError, match="Duplicate 'api'"):
            parse_assignment("a -> b | api:gpt4, api:gpt5")

    def test_duplicate_api_gen_raises_error(self):
        """Test duplicate api_gen declaration raises error."""
        with pytest.raises(ValueError, match="Duplicate 'api_gen'"):
            parse_assignment("a -> b | api_gen:gpt4, api_gen:gpt5")

    def test_duplicate_api_parse_raises_error(self):
        """Test duplicate api_parse declaration raises error."""
        with pytest.raises(ValueError, match="Duplicate 'api_parse'"):
            parse_assignment("a -> b | api_parse:gpt4, api_parse:gpt5")

    def test_duplicate_tool_raises_error(self):
        """Test duplicate tool declaration raises error."""
        with pytest.raises(ValueError, match="Duplicate tool"):
            parse_assignment("a -> b | tool:search, tool:search")

    def test_tool_wildcard_with_specific_raises_error(self):
        """Test mixing tool:* with specific tools raises error."""
        with pytest.raises(ValueError, match="Cannot mix"):
            parse_assignment("a -> b | tool:search, tool:*")

    def test_specific_tool_after_wildcard_raises_error(self):
        """Test adding specific tool after tool:* raises error."""
        with pytest.raises(ValueError, match="Cannot mix"):
            parse_assignment("a -> b | tool:*, tool:search")


class TestFormResources:
    """Tests for FormResources class."""

    def test_default_resources(self):
        """Test default FormResources."""
        res = FormResources()
        assert res.api is None
        assert res.api_gen is None
        assert res.api_parse is None
        assert res.tools is None

    def test_frozen(self):
        """Test FormResources is immutable."""
        res = FormResources(api="gpt4")
        with pytest.raises(AttributeError):
            res.api = "gpt5"


class TestForm:
    """Tests for Form class."""

    def test_basic_creation(self):
        """Test basic form creation."""
        form = Form(assignment="a, b -> c")
        assert form.assignment == "a, b -> c"
        assert form.branch_name is None
        assert form.input_fields == ["a", "b"]
        assert form.output_fields == ["c"]
        assert form.filled is False
        assert form.output is None

    def test_creation_with_branch(self):
        """Test form creation with branch prefix."""
        form = Form(assignment="worker: x -> y, z")
        assert form.branch_name == "worker"
        assert form.input_fields == ["x"]
        assert form.output_fields == ["y", "z"]

    def test_creation_with_resources(self):
        """Test form creation with resources."""
        form = Form(assignment="a -> b | api:gpt4, tool:search")
        assert form.resources.api == "gpt4"
        assert form.resources.tools == frozenset({"search"})

    def test_creation_full(self):
        """Test form creation with all components."""
        form = Form(assignment="orchestrator: a, b -> c | api:gpt4, tool:*")
        assert form.branch_name == "orchestrator"
        assert form.input_fields == ["a", "b"]
        assert form.output_fields == ["c"]
        assert form.resources.api == "gpt4"
        assert form.resources.tools == "*"

    def test_is_workable_all_inputs_available(self):
        """Test is_workable returns True when all inputs available."""
        form = Form(assignment="a, b -> c")
        available = {"a": 1, "b": 2, "extra": 3}
        assert form.is_workable(available) is True

    def test_is_workable_missing_input(self):
        """Test is_workable returns False when input missing."""
        form = Form(assignment="a, b -> c")
        available = {"a": 1}
        assert form.is_workable(available) is False

    def test_is_workable_none_value(self):
        """Test is_workable returns False when input is None."""
        form = Form(assignment="a, b -> c")
        available = {"a": 1, "b": None}
        assert form.is_workable(available) is False

    def test_is_workable_already_filled(self):
        """Test is_workable returns False when already filled."""
        form = Form(assignment="a -> b")
        form.filled = True
        available = {"a": 1}
        assert form.is_workable(available) is False

    def test_is_workable_no_inputs(self):
        """Test is_workable with no inputs required."""
        form = Form(assignment=" -> output")
        assert form.is_workable({}) is True

    def test_get_inputs(self):
        """Test get_inputs extracts correct fields."""
        form = Form(assignment="a, b -> c")
        available = {"a": 1, "b": 2, "c": 3, "d": 4}
        inputs = form.get_inputs(available)
        assert inputs == {"a": 1, "b": 2}

    def test_get_inputs_partial(self):
        """Test get_inputs with partial availability."""
        form = Form(assignment="a, b, c -> d")
        available = {"a": 1, "c": 3}
        inputs = form.get_inputs(available)
        assert inputs == {"a": 1, "c": 3}

    def test_fill(self):
        """Test fill marks form as filled."""
        form = Form(assignment="a -> b")
        assert form.filled is False
        assert form.output is None

        form.fill(output={"result": "value"})
        assert form.filled is True
        assert form.output == {"result": "value"}

    def test_get_output_data_dict(self):
        """Test get_output_data with dict output."""
        form = Form(assignment="a -> b, c")
        form.fill(output={"b": 1, "c": 2, "extra": 3})
        # Dict outputs don't have attribute access
        output_data = form.get_output_data()
        assert output_data == {}

    def test_get_output_data_model(self):
        """Test get_output_data with Pydantic model output."""

        class Output(BaseModel):
            b: int
            c: str

        form = Form(assignment="a -> b, c")
        form.fill(output=Output(b=42, c="test"))
        output_data = form.get_output_data()
        assert output_data == {"b": 42, "c": "test"}

    def test_get_output_data_partial_model(self):
        """Test get_output_data when model has subset of output fields."""

        class PartialOutput(BaseModel):
            b: int

        form = Form(assignment="a -> b, c")
        form.fill(output=PartialOutput(b=42))
        output_data = form.get_output_data()
        assert output_data == {"b": 42}

    def test_get_output_data_none(self):
        """Test get_output_data when output is None."""
        form = Form(assignment="a -> b")
        output_data = form.get_output_data()
        assert output_data == {}

    def test_repr_pending(self):
        """Test repr for pending form."""
        form = Form(assignment="a -> b")
        assert "pending" in repr(form)

    def test_repr_filled(self):
        """Test repr for filled form."""
        form = Form(assignment="a -> b")
        form.fill(output="result")
        assert "filled" in repr(form)

    def test_repr_with_resources(self):
        """Test repr shows resources."""
        form = Form(assignment="a -> b | api:gpt4, tool:search")
        r = repr(form)
        assert "api:gpt4" in r
        assert "tools:" in r

    def test_form_has_uuid(self):
        """Test that form has UUID id."""
        form = Form(assignment="a -> b")
        assert form.id is not None
        assert len(str(form.id)) == 36  # UUID format

    def test_form_has_created_at(self):
        """Test that form has created_at timestamp."""
        form = Form(assignment="a -> b")
        assert form.created_at is not None

    def test_get_output_data_model_dump_fallback(self):
        """Test get_output_data uses model_dump when field not direct attribute.

        This covers line 163: result[field] = data[field] via model_dump path.
        """

        class CustomOutput:
            """Custom class with model_dump but no direct field attributes."""

            def __init__(self, data: dict):
                self._data = data

            def model_dump(self) -> dict:
                return self._data

        form = Form(assignment="a -> result, extra")
        # CustomOutput has model_dump but no 'result' attribute
        output = CustomOutput({"result": 99, "extra": "value"})
        form.fill(output=output)
        output_data = form.get_output_data()
        assert output_data == {"result": 99, "extra": "value"}
