# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for breakdown_pydantic_annotation module.

Covers edge cases including:
- clean_types=False path
- Non-Pydantic model error handling
- Maximum recursion depth handling
- is_pydantic_model with non-class inputs
"""

import pytest
from pydantic import BaseModel

from lionpride.libs.schema_handlers._breakdown_pydantic_annotation import (
    breakdown_pydantic_annotation,
    is_pydantic_model,
)


class SimpleModel(BaseModel):
    name: str
    age: int


class NestedModel(BaseModel):
    title: str
    details: SimpleModel


class ListModel(BaseModel):
    items: list[SimpleModel]
    tags: list[str]


class TestBreakdownPydanticAnnotation:
    """Test breakdown_pydantic_annotation function."""

    def test_simple_model_clean_types_true(self):
        """Test simple model with clean_types=True (default)."""
        result = breakdown_pydantic_annotation(SimpleModel)
        assert result == {"name": "str", "age": "int"}

    def test_simple_model_clean_types_false(self):
        """Test simple model with clean_types=False (covers line 59)."""
        result = breakdown_pydantic_annotation(SimpleModel, clean_types=False)
        # With clean_types=False, we get raw type objects
        assert "name" in result
        assert "age" in result
        # Values should be raw type objects, not strings
        assert result["name"] is str
        assert result["age"] is int

    def test_nested_model_clean_types_false(self):
        """Test nested model with clean_types=False."""
        result = breakdown_pydantic_annotation(NestedModel, clean_types=False)
        assert "title" in result
        assert "details" in result
        assert isinstance(result["details"], dict)

    def test_list_model_clean_types_false(self):
        """Test list model with clean_types=False."""
        result = breakdown_pydantic_annotation(ListModel, clean_types=False)
        assert "items" in result
        assert "tags" in result
        assert isinstance(result["items"], list)
        assert isinstance(result["tags"], list)

    def test_non_pydantic_model_raises_type_error(self):
        """Test that non-Pydantic model raises TypeError (covers line 84)."""

        class NotAModel:
            name: str

        with pytest.raises(TypeError, match="must be a Pydantic model"):
            breakdown_pydantic_annotation(NotAModel)

    def test_max_depth_exceeded_raises_recursion_error(self):
        """Test that exceeding max_depth raises RecursionError (covers line 87)."""
        with pytest.raises(RecursionError, match="Maximum recursion depth reached"):
            breakdown_pydantic_annotation(NestedModel, max_depth=0)

    def test_max_depth_one_stops_at_first_level(self):
        """Test max_depth=1 processes only first level."""
        # max_depth=1 allows only the outer model, nested SimpleModel triggers error
        with pytest.raises(RecursionError, match="Maximum recursion depth reached"):
            breakdown_pydantic_annotation(NestedModel, max_depth=1)


class TestIsPydanticModel:
    """Test is_pydantic_model function."""

    def test_valid_pydantic_model(self):
        """Test returns True for Pydantic BaseModel subclass."""
        assert is_pydantic_model(SimpleModel) is True

    def test_base_model_itself(self):
        """Test returns True for BaseModel itself."""
        assert is_pydantic_model(BaseModel) is True

    def test_non_model_class(self):
        """Test returns False for non-Pydantic class."""

        class RegularClass:
            pass

        assert is_pydantic_model(RegularClass) is False

    def test_instance_not_class(self):
        """Test returns False for instance instead of class."""
        instance = SimpleModel(name="test", age=42)
        assert is_pydantic_model(instance) is False

    def test_string_input(self):
        """Test returns False for string input (covers lines 109-110)."""
        assert is_pydantic_model("SimpleModel") is False

    def test_none_input(self):
        """Test returns False for None input (covers lines 109-110)."""
        assert is_pydantic_model(None) is False

    def test_int_input(self):
        """Test returns False for int input (covers lines 109-110)."""
        assert is_pydantic_model(42) is False

    def test_callable_input(self):
        """Test returns False for callable that isn't a class."""

        def my_func():
            pass

        assert is_pydantic_model(my_func) is False

    def test_builtin_type(self):
        """Test returns False for builtin types like str, int."""
        assert is_pydantic_model(str) is False
        assert is_pydantic_model(int) is False
        assert is_pydantic_model(list) is False
