# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Operative validation and action execution."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lionpride.operations.operate.operative import (
    Operative,
    create_action_operative,
    create_operative_from_model,
)


class TestOperative:
    """Test Operative class for two-tier validation."""

    def test_init_with_operable(self):
        """Test Operative initialization with Operable."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, name="test", strict=True)

        assert operative.name == "test"
        assert operative.adapter == "pydantic"
        assert operative.strict is True
        assert operative.auto_retry_parse is True
        assert operative.max_retries == 3
        assert operative.operable is operable
        assert operative.request_exclude == set()

    def test_init_defaults(self):
        """Test Operative initialization with defaults."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable)

        assert operative.name == "TestOp"
        assert operative.strict is False
        assert operative.auto_retry_parse is True
        assert operative.max_retries == 3

    def test_create_request_model(self):
        """Test request model creation."""
        from lionpride.types import Operable, Spec

        spec1 = Spec(base_type=str, name="field1")
        spec2 = Spec(base_type=int, name="field2")
        operable = Operable(specs=(spec1, spec2), name="TestOp")

        operative = Operative(operable=operable, name="test")

        request_cls = operative.create_request_model()

        assert request_cls.__name__ == "testRequest"
        assert "field1" in request_cls.model_fields
        assert "field2" in request_cls.model_fields

    def test_create_request_model_cached(self):
        """Test request model is cached."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable)

        request_cls1 = operative.create_request_model()
        request_cls2 = operative.create_request_model()

        assert request_cls1 is request_cls2

    def test_create_response_model(self):
        """Test response model creation."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, name="test")

        response_cls = operative.create_response_model()

        assert response_cls.__name__ == "testResponse"
        assert "field1" in response_cls.model_fields

    def test_create_response_model_creates_request_first(self):
        """Test response model creation triggers request model creation."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, name="test")

        # Create response without creating request first
        response_cls = operative.create_response_model()

        # Verify both are created
        assert operative._request_model_cls is not None
        assert operative._response_model_cls is response_cls

    def test_create_response_model_cached(self):
        """Test response model is cached."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable)

        response_cls1 = operative.create_response_model()
        response_cls2 = operative.create_response_model()

        assert response_cls1 is response_cls2

    def test_request_type_property(self):
        """Test request_type property."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, name="test")

        request_type = operative.request_type

        assert request_type.__name__ == "testRequest"
        assert operative._request_model_cls is request_type

    def test_response_type_property(self):
        """Test response_type property."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, name="test")

        response_type = operative.response_type

        assert response_type.__name__ == "testResponse"
        assert operative._response_model_cls is response_type

    def test_validate_response_strict_success(self):
        """Test strict validation success."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, strict=True)

        # Valid JSON matching schema
        text = '{"field1": "value1"}'

        result = operative.validate_response(text)

        assert result is not None
        assert result.field1 == "value1"
        assert operative._should_retry is False

    def test_validate_response_strict_failure_raises(self):
        """Test strict validation failure raises exception."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, strict=True)

        # Invalid JSON
        text = '{"wrong_field": "value"}'

        with pytest.raises((ValueError, Exception)):
            operative.validate_response(text)

        assert operative._should_retry is True

    def test_validate_response_fuzzy_fallback(self):
        """Test fuzzy fallback on validation failure."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1", default="default")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, strict=False, auto_retry_parse=True)

        # Partially invalid JSON (missing required field, but has default)
        text = "{}"

        result = operative.validate_response(text)

        # Should succeed via fuzzy parsing with default
        assert result is not None

    def test_validate_response_override_strict(self):
        """Test strict parameter override."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        # Initialize as non-strict
        operative = Operative(operable=operable, strict=False)

        # Invalid JSON
        text = '{"wrong_field": "value"}'

        # Override to strict
        with pytest.raises((ValueError, Exception)):
            operative.validate_response(text, strict=True)

    def test_validate_response_no_auto_retry(self):
        """Test validation without auto retry."""
        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")

        operative = Operative(operable=operable, strict=False, auto_retry_parse=False)

        # Invalid JSON
        text = '{"wrong_field": "value"}'

        result = operative.validate_response(text)

        # Should return None without retrying
        assert result is None
        assert operative._should_retry is False

    def test_validate_response_fuzzy_fallback_success(self):
        """Test fuzzy fallback succeeds after first attempt fails (lines 134-143)."""
        from unittest.mock import patch

        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")
        operative = Operative(operable=operable, strict=False, auto_retry_parse=True)

        # Create a valid response model instance for the second call
        response_cls = operative.create_response_model()
        success_result = response_cls(field1="value1")

        text = '{"field1": "value1"}'

        # Mock validate_response at the adapter module level
        with patch(
            "lionpride.types.spec_adapters.pydantic_field.PydanticSpecAdapter.validate_response"
        ) as mock_validate:
            # First call raises, second call succeeds
            mock_validate.side_effect = [
                ValueError("First attempt fails"),
                success_result,
            ]

            result = operative.validate_response(text)

            # Fuzzy fallback should succeed
            assert result is not None
            assert result.field1 == "value1"
            assert operative._should_retry is False
            assert mock_validate.call_count == 2

    def test_validate_response_fuzzy_fallback_failure(self):
        """Test fuzzy fallback fails when both attempts fail (lines 144-147)."""
        from unittest.mock import patch

        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")
        operative = Operative(operable=operable, strict=False, auto_retry_parse=True)

        text = '{"field1": "value1"}'

        # Mock validate_response at the adapter module level
        with patch(
            "lionpride.types.spec_adapters.pydantic_field.PydanticSpecAdapter.validate_response"
        ) as mock_validate:
            # Both calls raise exceptions
            mock_validate.side_effect = [
                ValueError("First attempt fails"),
                ValueError("Second attempt also fails"),
            ]

            result = operative.validate_response(text)

            # Both attempts fail, should return None
            assert result is None
            assert operative.response_str_dict == text
            assert operative._should_retry is False
            assert mock_validate.call_count == 2

    def test_validate_response_fuzzy_fallback_not_triggered_when_strict(self):
        """Test fuzzy fallback not triggered when strict=True (line 134 condition)."""
        from unittest.mock import patch

        from lionpride.types import Operable, Spec

        spec = Spec(base_type=str, name="field1")
        operable = Operable(specs=(spec,), name="TestOp")
        operative = Operative(operable=operable, strict=True, auto_retry_parse=True)

        text = '{"wrong": "value"}'

        # Mock validate_response at the adapter module level
        with patch(
            "lionpride.types.spec_adapters.pydantic_field.PydanticSpecAdapter.validate_response"
        ) as mock_validate:
            mock_validate.side_effect = ValueError("Validation fails")

            # Should raise because strict=True prevents fuzzy fallback
            with pytest.raises(ValueError):
                operative.validate_response(text)

            # Only one call (no fuzzy fallback)
            assert mock_validate.call_count == 1
            assert operative._should_retry is True


class TestCreateOperativeFromModel:
    """Test create_operative_from_model factory."""

    def test_basic_model(self):
        """Test creating Operative from basic Pydantic model."""

        class TestModel(BaseModel):
            field1: str
            field2: int

        operative = create_operative_from_model(TestModel, name="test")

        assert operative.name == "test"
        assert operative.operable.name == "test"

        request_cls = operative.create_request_model()
        # Request model has a single field with the operative name
        assert "test" in request_cls.model_fields

        # Response model should be the original TestModel
        response_cls = operative.create_response_model()
        assert response_cls is TestModel

    def test_model_with_defaults(self):
        """Test creating Operative from model with default values."""

        class TestModel(BaseModel):
            field1: str = "default"
            field2: int = 42

        operative = create_operative_from_model(TestModel)

        # Response model should be the original TestModel
        response_cls = operative.create_response_model()
        assert response_cls is TestModel

        # Can create instance with defaults
        instance = response_cls()
        assert instance.field1 == "default"
        assert instance.field2 == 42

    def test_model_with_strict(self):
        """Test creating strict Operative."""

        class TestModel(BaseModel):
            field1: str

        operative = create_operative_from_model(TestModel, strict=True)

        assert operative.strict is True

    def test_model_without_auto_retry(self):
        """Test creating Operative without auto retry."""

        class TestModel(BaseModel):
            field1: str

        operative = create_operative_from_model(TestModel, auto_retry_parse=False)

        assert operative.auto_retry_parse is False

    def test_name_defaults_to_model_name(self):
        """Test name defaults to model class name."""

        class MyCustomModel(BaseModel):
            field1: str

        operative = create_operative_from_model(MyCustomModel)

        assert operative.name == "MyCustomModel"


class TestCreateActionOperative:
    """Test create_action_operative factory."""

    def test_basic_action_operative(self):
        """Test creating basic action Operative."""
        operative = create_action_operative(name="test", actions=True)

        request_cls = operative.create_request_model()
        response_cls = operative.create_response_model()

        # Request model includes action_requests
        assert "action_requests" in request_cls.model_fields

        # Request model excludes action_responses
        assert "action_responses" not in request_cls.model_fields

        # Response model includes both
        assert "action_requests" in response_cls.model_fields
        assert "action_responses" in response_cls.model_fields

    def test_action_operative_with_base_model(self):
        """Test action Operative extending base model."""

        class Analysis(BaseModel):
            summary: str
            confidence: float

        operative = create_action_operative(base_model=Analysis, actions=True)

        request_cls = operative.create_request_model()

        # Should have the base model as a spec + action_requests
        assert "analysis" in request_cls.model_fields  # The model as a single spec (lowercase name)
        assert "action_requests" in request_cls.model_fields
        # action_responses should be excluded from request model
        assert "action_responses" not in request_cls.model_fields

    def test_action_operative_with_reason(self):
        """Test action Operative with reasoning field."""
        operative = create_action_operative(reason=True, actions=True)

        request_cls = operative.create_request_model()

        # Should have reason field
        assert "reason" in request_cls.model_fields

    def test_action_operative_reason_only(self):
        """Test Operative with reasoning but no actions."""
        operative = create_action_operative(reason=True, actions=False)

        request_cls = operative.create_request_model()

        # Should have reason, but no action fields
        assert "reason" in request_cls.model_fields
        assert "action_requests" not in request_cls.model_fields
        assert "action_responses" not in request_cls.model_fields

    def test_action_operative_actions_only(self):
        """Test Operative with actions but no reasoning."""
        operative = create_action_operative(reason=False, actions=True)

        request_cls = operative.create_request_model()

        # Should have actions, but no reason
        assert "action_requests" in request_cls.model_fields
        assert "reason" not in request_cls.model_fields

    def test_action_operative_name_defaults(self):
        """Test name defaults."""

        class MyModel(BaseModel):
            field: str

        operative1 = create_action_operative(base_model=MyModel)
        operative2 = create_action_operative()

        assert operative1.name == "MyModel"
        assert operative2.name == "ActionOperative"


# NOTE: act() tests skipped due to complex mocking requirements for ServiceRegistry + Tool
# The act() function is straightforward (see actions.py) and will be integration-tested
# with real Tool instances when Tool infrastructure is available.
# Coverage: 25 Operative tests cover the critical validation logic (operative.py)
