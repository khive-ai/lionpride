# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for schema_handlers __init__.py module.

Coverage targets:
- __getattr__ lazy import behavior
- AttributeError for unknown attributes (lines 37-38)
"""

import pytest


class TestLazyImport:
    """Test lazy import behavior in __getattr__."""

    def test_load_pydantic_model_from_schema_lazy_import(self):
        """Test load_pydantic_model_from_schema is lazily imported."""
        from lionpride.libs import schema_handlers

        # Access the lazy-loaded function
        fn = schema_handlers.load_pydantic_model_from_schema

        # Verify it's callable and imported correctly
        assert callable(fn)
        assert fn.__name__ == "load_pydantic_model_from_schema"

    def test_unknown_attribute_raises_attribute_error(self):
        """Test accessing unknown attribute raises AttributeError.

        Coverage: schema_handlers/__init__.py lines 37-38
        """
        from lionpride.libs import schema_handlers

        with pytest.raises(AttributeError) as exc_info:
            _ = schema_handlers.nonexistent_function

        # Verify error message format
        assert "has no attribute" in str(exc_info.value)
        assert "nonexistent_function" in str(exc_info.value)

    def test_regular_exports_available(self):
        """Test regular exports are available without lazy import."""
        from lionpride.libs.schema_handlers import (
            breakdown_pydantic_annotation,
            is_pydantic_model,
            minimal_yaml,
            typescript_schema,
        )

        # Verify all regular exports are callable
        assert callable(breakdown_pydantic_annotation)
        assert callable(is_pydantic_model)
        assert callable(minimal_yaml)
        assert callable(typescript_schema)
