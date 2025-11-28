# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Report class."""

import pytest
from pydantic import BaseModel, Field

from lionpride.work import Form, Report


class AnalysisModel(BaseModel):
    """Test model for analysis output."""

    summary: str = Field(description="Summary text")
    score: float = Field(description="Score value")


class InsightsModel(BaseModel):
    """Test model for insights output."""

    patterns: list[str] = Field(default_factory=list)


class TestReportBasic:
    """Basic Report tests."""

    def test_create_empty_report(self):
        """Test creating a report without assignment."""
        report = Report()
        assert report.assignment == ""
        assert report.form_assignments == []
        assert len(report.forms) == 0

    def test_create_report_with_assignment(self):
        """Test creating a report with assignment."""
        report = Report(
            assignment="input -> output",
            form_assignments=["input -> middle", "middle -> output"],
        )
        assert report.input_fields == ["input"]
        assert report.output_fields == ["output"]
        assert len(report.forms) == 2

    def test_forms_created_from_assignments(self):
        """Test that forms are created from form_assignments."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        forms = list(report.forms)
        assert len(forms) == 2
        assert forms[0].assignment == "a -> b"
        assert forms[1].assignment == "b -> c"

    def test_initialize_with_inputs(self):
        """Test initialize sets available_data."""
        report = Report(
            assignment="x, y -> z",
            form_assignments=["x, y -> z"],
        )
        report.initialize(x=1, y=2)
        assert report.available_data == {"x": 1, "y": 2}

    def test_initialize_missing_input_raises(self):
        """Test initialize raises for missing required input."""
        report = Report(
            assignment="x, y -> z",
            form_assignments=["x, y -> z"],
        )
        with pytest.raises(ValueError, match="Missing required input: 'y'"):
            report.initialize(x=1)


class TestReportSubclass:
    """Tests for Report subclass pattern."""

    def test_subclass_with_schema_attributes(self):
        """Test Report subclass with schema class attributes."""

        class MyReport(Report):
            analysis: AnalysisModel | None = None
            insights: InsightsModel | None = None

            assignment: str = "topic -> insights"
            form_assignments: list[str] = [
                "topic -> analysis",
                "analysis -> insights",
            ]

        report = MyReport()
        assert report.input_fields == ["topic"]
        assert report.output_fields == ["insights"]
        assert len(report.forms) == 2

    def test_get_field_type_pydantic_model(self):
        """Test get_field_type returns Pydantic model type."""

        class MyReport(Report):
            analysis: AnalysisModel | None = None

            assignment: str = "a -> analysis"
            form_assignments: list[str] = ["a -> analysis"]

        report = MyReport()
        field_type = report.get_field_type("analysis")
        assert field_type is AnalysisModel

    def test_get_field_type_primitive(self):
        """Test get_field_type returns primitive type."""

        class MyReport(Report):
            score: float | None = None

            assignment: str = "a -> score"
            form_assignments: list[str] = ["a -> score"]

        report = MyReport()
        field_type = report.get_field_type("score")
        assert field_type is float

    def test_get_field_type_unwraps_optional(self):
        """Test get_field_type unwraps Optional/Union with None."""

        class MyReport(Report):
            analysis: AnalysisModel | None = None

            assignment: str = "a -> analysis"
            form_assignments: list[str] = ["a -> analysis"]

        report = MyReport()
        # Should unwrap AnalysisModel | None to AnalysisModel
        field_type = report.get_field_type("analysis")
        assert field_type is AnalysisModel

    def test_get_field_type_unknown_field(self):
        """Test get_field_type returns None for unknown field."""

        class MyReport(Report):
            assignment: str = "a -> b"
            form_assignments: list[str] = ["a -> b"]

        report = MyReport()
        assert report.get_field_type("unknown") is None

    def test_get_field_type_excludes_infrastructure_fields(self):
        """Test get_field_type excludes Report infrastructure fields."""

        class MyReport(Report):
            assignment: str = "a -> b"
            form_assignments: list[str] = ["a -> b"]

        report = MyReport()
        # These are infrastructure fields, not output schemas
        assert report.get_field_type("assignment") is None
        assert report.get_field_type("forms") is None
        assert report.get_field_type("available_data") is None

    def test_get_request_model_pydantic(self):
        """Test get_request_model returns Pydantic model."""

        class MyReport(Report):
            analysis: AnalysisModel | None = None

            assignment: str = "a -> analysis"
            form_assignments: list[str] = ["a -> analysis"]

        report = MyReport()
        model = report.get_request_model("analysis")
        assert model is AnalysisModel

    def test_get_request_model_primitive_returns_none(self):
        """Test get_request_model returns None for primitive types."""

        class MyReport(Report):
            score: float | None = None

            assignment: str = "a -> score"
            form_assignments: list[str] = ["a -> score"]

        report = MyReport()
        model = report.get_request_model("score")
        assert model is None


class TestReportWorkflow:
    """Tests for Report workflow execution."""

    def test_next_forms_initial(self):
        """Test next_forms returns forms with available inputs."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        next_forms = report.next_forms()
        assert len(next_forms) == 1
        assert next_forms[0].assignment == "a -> b"

    def test_next_forms_after_completion(self):
        """Test next_forms after completing a form."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        # Complete first form
        forms = list(report.forms)
        forms[0].fill(output={"b": 2})
        report.complete_form(forms[0])

        # Now second form should be workable
        next_forms = report.next_forms()
        assert len(next_forms) == 1
        assert next_forms[0].assignment == "b -> c"

    def test_complete_form_updates_available_data(self):
        """Test complete_form updates available_data."""

        class OutputModel(BaseModel):
            b: int

        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)

        form = next(iter(report.forms))
        form.fill(output=OutputModel(b=42))
        report.complete_form(form)

        assert "b" in report.available_data
        assert report.available_data["b"] == 42

    def test_complete_form_not_filled_raises(self):
        """Test complete_form raises if form not filled."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)

        form = next(iter(report.forms))
        with pytest.raises(ValueError, match="Form is not filled"):
            report.complete_form(form)

    def test_is_complete_false_initially(self):
        """Test is_complete is False initially."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)
        assert report.is_complete() is False

    def test_is_complete_true_when_outputs_available(self):
        """Test is_complete is True when all outputs available."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)
        report.available_data["b"] = "result"
        assert report.is_complete() is True

    def test_get_deliverable(self):
        """Test get_deliverable returns output fields."""
        report = Report(
            assignment="a -> b, c",
            form_assignments=["a -> b, c"],
        )
        report.initialize(a=1)
        report.available_data["b"] = "value_b"
        report.available_data["c"] = "value_c"
        report.available_data["extra"] = "ignored"

        deliverable = report.get_deliverable()
        assert deliverable == {"b": "value_b", "c": "value_c"}

    def test_get_all_data(self):
        """Test get_all_data returns copy of available_data."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)
        report.available_data["b"] = 2

        all_data = report.get_all_data()
        assert all_data == {"a": 1, "b": 2}

        # Verify it's a copy
        all_data["c"] = 3
        assert "c" not in report.available_data

    def test_progress(self):
        """Test progress property."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        assert report.progress == (0, 2)

        # Complete first form
        form = next(iter(report.forms))
        form.fill(output={"b": 2})
        report.complete_form(form)

        assert report.progress == (1, 2)

    def test_repr(self):
        """Test Report repr."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        assert repr(report) == "Report('a -> c', 0/2 forms)"


class TestReportParallel:
    """Tests for parallel form execution patterns."""

    def test_parallel_forms_all_workable(self):
        """Test multiple forms are workable when inputs available."""
        report = Report(
            assignment="a -> b, c, d",
            form_assignments=[
                "a -> b",  # Parallel
                "a -> c",  # Parallel
                "a -> d",  # Parallel
            ],
        )
        report.initialize(a=1)

        next_forms = report.next_forms()
        assert len(next_forms) == 3

    def test_diamond_dependency(self):
        """Test diamond dependency pattern."""
        report = Report(
            assignment="a -> d",
            form_assignments=[
                "a -> b",
                "a -> c",
                "b, c -> d",
            ],
        )
        report.initialize(a=1)

        # Initially only b and c are workable
        next_forms = report.next_forms()
        assert len(next_forms) == 2
        assignments = {f.assignment for f in next_forms}
        assert assignments == {"a -> b", "a -> c"}

        # Complete b
        forms = list(report.forms)
        forms[0].fill(output={"b": 2})
        report.complete_form(forms[0])

        # Still waiting for c
        next_forms = report.next_forms()
        assert len(next_forms) == 1
        assert next_forms[0].assignment == "a -> c"

        # Complete c
        forms[1].fill(output={"c": 3})
        report.complete_form(forms[1])

        # Now d is workable
        next_forms = report.next_forms()
        assert len(next_forms) == 1
        assert next_forms[0].assignment == "b, c -> d"


class TestReportEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_field_type_non_optional_type(self):
        """Test get_field_type returns type directly when not Optional."""

        class MyReport(Report):
            # Non-optional field (required)
            data: dict = Field(default_factory=dict)

            assignment: str = "a -> data"
            form_assignments: list[str] = ["a -> data"]

        report = MyReport()
        field_type = report.get_field_type("data")
        assert field_type is dict

    def test_get_field_type_handles_exception(self):
        """Test get_field_type returns None on exception.

        Covers lines 135-136: except Exception: pass
        """
        from unittest.mock import patch

        class MyReport(Report):
            data: str | None = None

            assignment: str = "a -> data"
            form_assignments: list[str] = ["a -> data"]

        report = MyReport()

        # Mock get_type_hints to raise an exception
        with patch("lionpride.work.report.get_type_hints") as mock_hints:
            mock_hints.side_effect = NameError("Unresolved forward reference")
            result = report.get_field_type("data")
            assert result is None

    def test_get_request_model_handles_typeerror(self):
        """Test get_request_model returns None on TypeError.

        Covers lines 149-150: except TypeError: pass
        """
        from unittest.mock import patch

        class MyReport(Report):
            data: AnalysisModel | None = None

            assignment: str = "a -> data"
            form_assignments: list[str] = ["a -> data"]

        report = MyReport()

        # Mock issubclass to raise TypeError
        original_issubclass = issubclass

        def mock_issubclass(cls, classinfo):
            if classinfo is BaseModel:
                raise TypeError("issubclass() arg 1 must be a class")
            return original_issubclass(cls, classinfo)

        with patch("builtins.issubclass", side_effect=mock_issubclass):
            model = report.get_request_model("data")
            assert model is None

    def test_complete_form_stores_raw_output(self):
        """Test complete_form stores raw output under primary field."""

        class Result(BaseModel):
            value: int

        report = Report(
            assignment="a -> result",
            form_assignments=["a -> result"],
        )
        report.initialize(a=1)

        form = next(iter(report.forms))
        result_obj = Result(value=42)
        form.fill(output=result_obj)
        report.complete_form(form)

        # Raw output should be stored
        assert report.available_data["result"] == result_obj
