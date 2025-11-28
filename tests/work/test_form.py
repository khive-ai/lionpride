# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Form and parse_assignment."""

import pytest
from pydantic import BaseModel

from lionpride.work import Form, parse_assignment


class TestParseAssignment:
    """Tests for parse_assignment function."""

    def test_simple_assignment(self):
        """Test basic assignment without branch."""
        branch, inputs, outputs = parse_assignment("a, b -> c")
        assert branch is None
        assert inputs == ["a", "b"]
        assert outputs == ["c"]

    def test_single_input_output(self):
        """Test single input to single output."""
        branch, inputs, outputs = parse_assignment("input -> output")
        assert branch is None
        assert inputs == ["input"]
        assert outputs == ["output"]

    def test_multiple_outputs(self):
        """Test multiple outputs."""
        branch, inputs, outputs = parse_assignment("a -> b, c, d")
        assert branch is None
        assert inputs == ["a"]
        assert outputs == ["b", "c", "d"]

    def test_no_inputs(self):
        """Test assignment with no inputs."""
        branch, inputs, outputs = parse_assignment(" -> output")
        assert branch is None
        assert inputs == []
        assert outputs == ["output"]

    def test_with_branch_prefix(self):
        """Test assignment with branch prefix."""
        branch, inputs, outputs = parse_assignment("orchestrator: a, b -> c")
        assert branch == "orchestrator"
        assert inputs == ["a", "b"]
        assert outputs == ["c"]

    def test_branch_with_spaces(self):
        """Test branch prefix with surrounding spaces."""
        branch, inputs, outputs = parse_assignment("  planner  :  x, y  ->  z  ")
        assert branch == "planner"
        assert inputs == ["x", "y"]
        assert outputs == ["z"]

    def test_complex_branch_assignment(self):
        """Test complex assignment with branch and multiple fields."""
        branch, inputs, outputs = parse_assignment(
            "implementer: context, instruction, plan -> result, score"
        )
        assert branch == "implementer"
        assert inputs == ["context", "instruction", "plan"]
        assert outputs == ["result", "score"]

    def test_invalid_no_arrow(self):
        """Test that missing arrow raises ValueError."""
        with pytest.raises(ValueError, match="Must contain '->'"):
            parse_assignment("a, b, c")

    def test_invalid_empty(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Must contain '->'"):
            parse_assignment("")

    def test_invalid_no_outputs(self):
        """Test that missing outputs raises ValueError."""
        with pytest.raises(ValueError, match="at least one output"):
            parse_assignment("a, b -> ")

    def test_colon_after_arrow_not_branch(self):
        """Test that colon after arrow is not treated as branch."""
        branch, inputs, outputs = parse_assignment("a -> b:c")
        assert branch is None
        assert inputs == ["a"]
        assert outputs == ["b:c"]


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
        assert repr(form) == "Form('a -> b', pending)"

    def test_repr_filled(self):
        """Test repr for filled form."""
        form = Form(assignment="a -> b")
        form.fill(output="result")
        assert repr(form) == "Form('a -> b', filled)"

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
