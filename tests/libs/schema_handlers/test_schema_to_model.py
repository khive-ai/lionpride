# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamic Pydantic model generation from JSON schemas.

This module tests the _schema_to_model module with comprehensive coverage
including helper functions (which don't need datamodel_code_generator) and
mocked tests for the main generation function.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from pydantic import BaseModel, PydanticUserError, ValidationError

from lionpride import ln

# Check if optional dependency is available
_HAS_SCHEMA_GEN = ln.is_import_installed("datamodel_code_generator")


# ===========================================================================
# Helper Function Tests (no dependency required)
# ===========================================================================


class TestSanitizeModelName:
    """Test _sanitize_model_name function - no external dependencies."""

    def test_valid_simple_name(self):
        """Test sanitization of simple valid name."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        assert _sanitize_model_name("UserModel") == "UserModel"

    def test_name_with_spaces(self):
        """Test sanitization removes spaces."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        assert _sanitize_model_name("User Model") == "UserModel"

    def test_name_with_special_chars(self):
        """Test sanitization removes special characters."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        assert _sanitize_model_name("User@Profile#123") == "UserProfile123"

    def test_name_with_underscores(self):
        """Test underscores are preserved."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        assert _sanitize_model_name("user_profile_model") == "user_profile_model"

    def test_name_starting_with_digit_raises(self):
        """Test that names starting with digit raise ValueError."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        with pytest.raises(ValueError, match="Cannot extract valid Python identifier"):
            _sanitize_model_name("123Invalid")

    def test_only_special_chars_raises(self):
        """Test that names with only special chars raise ValueError."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        with pytest.raises(ValueError, match="Cannot extract valid Python identifier"):
            _sanitize_model_name("@@@###")

    def test_whitespace_only_raises(self):
        """Test that whitespace-only names raise ValueError."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        with pytest.raises(ValueError, match="Cannot extract valid Python identifier"):
            _sanitize_model_name("   ")

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        with pytest.raises(ValueError, match="Cannot extract valid Python identifier"):
            _sanitize_model_name("")

    def test_digit_after_chars_ok(self):
        """Test that names with digits after letters are valid."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        assert _sanitize_model_name("Model2023") == "Model2023"

    def test_mixed_special_and_valid(self):
        """Test mixed special characters and valid chars."""
        from lionpride.libs.schema_handlers._schema_to_model import _sanitize_model_name

        assert _sanitize_model_name("My-Custom.Model!Test") == "MyCustomModelTest"


class TestExtractModelNameFromSchema:
    """Test _extract_model_name_from_schema function."""

    def test_valid_title(self):
        """Test extraction with valid title."""
        from lionpride.libs.schema_handlers._schema_to_model import (
            _extract_model_name_from_schema,
        )

        schema = {"title": "UserProfile"}
        assert _extract_model_name_from_schema(schema, "Default") == "UserProfile"

    def test_title_with_spaces(self):
        """Test extraction sanitizes title with spaces."""
        from lionpride.libs.schema_handlers._schema_to_model import (
            _extract_model_name_from_schema,
        )

        schema = {"title": "User Profile Model"}
        assert _extract_model_name_from_schema(schema, "Default") == "UserProfileModel"

    def test_no_title_uses_default(self):
        """Test fallback to default when no title."""
        from lionpride.libs.schema_handlers._schema_to_model import (
            _extract_model_name_from_schema,
        )

        schema = {}
        assert _extract_model_name_from_schema(schema, "DefaultModel") == "DefaultModel"

    def test_invalid_title_uses_default(self):
        """Test fallback when title can't be sanitized."""
        from lionpride.libs.schema_handlers._schema_to_model import (
            _extract_model_name_from_schema,
        )

        # Digit start
        schema = {"title": "123Invalid"}
        assert _extract_model_name_from_schema(schema, "DefaultModel") == "DefaultModel"

        # Only special chars
        schema = {"title": "@@@###"}
        assert _extract_model_name_from_schema(schema, "DefaultModel") == "DefaultModel"

        # Whitespace only
        schema = {"title": "   "}
        assert _extract_model_name_from_schema(schema, "DefaultModel") == "DefaultModel"

    def test_non_string_title_uses_default(self):
        """Test fallback when title is not a string."""
        from lionpride.libs.schema_handlers._schema_to_model import (
            _extract_model_name_from_schema,
        )

        schema = {"title": 123}
        assert _extract_model_name_from_schema(schema, "DefaultModel") == "DefaultModel"

        schema = {"title": ["list", "title"]}
        assert _extract_model_name_from_schema(schema, "DefaultModel") == "DefaultModel"

        schema = {"title": None}
        assert _extract_model_name_from_schema(schema, "DefaultModel") == "DefaultModel"


class TestGetPythonVersionEnum:
    """Test _get_python_version_enum function."""

    def test_current_version_detected(self):
        """Test current Python version is properly mapped."""
        from lionpride.libs.schema_handlers._schema_to_model import _get_python_version_enum

        # Create mock module with all versions
        class MockPythonVersion:
            PY_311 = "3.11"
            PY_312 = "3.12"
            PY_313 = "3.13"
            PY_314 = "3.14"

        result = _get_python_version_enum(MockPythonVersion)
        # Should return one of the valid versions based on sys.version_info
        assert result in ["3.11", "3.12", "3.13", "3.14"]

    def test_unsupported_version_fallback(self, monkeypatch):
        """Test fallback to PY_312 for unsupported versions."""
        from lionpride.libs.schema_handlers._schema_to_model import _get_python_version_enum

        class MockVersionInfo:
            major = 3
            minor = 10

        monkeypatch.setattr(sys, "version_info", MockVersionInfo())

        class MockPythonVersion:
            PY_312 = "3.12"

        result = _get_python_version_enum(MockPythonVersion)
        assert result == "3.12"

    def test_python_311(self, monkeypatch):
        """Test Python 3.11 detection."""
        from lionpride.libs.schema_handlers._schema_to_model import _get_python_version_enum

        class MockVersionInfo:
            major = 3
            minor = 11

        monkeypatch.setattr(sys, "version_info", MockVersionInfo())

        class MockPythonVersion:
            PY_311 = "3.11"
            PY_312 = "3.12"

        result = _get_python_version_enum(MockPythonVersion)
        assert result == "3.11"

    def test_python_312(self, monkeypatch):
        """Test Python 3.12 detection."""
        from lionpride.libs.schema_handlers._schema_to_model import _get_python_version_enum

        class MockVersionInfo:
            major = 3
            minor = 12

        monkeypatch.setattr(sys, "version_info", MockVersionInfo())

        class MockPythonVersion:
            PY_312 = "3.12"

        result = _get_python_version_enum(MockPythonVersion)
        assert result == "3.12"

    def test_python_313(self, monkeypatch):
        """Test Python 3.13 detection."""
        from lionpride.libs.schema_handlers._schema_to_model import _get_python_version_enum

        class MockVersionInfo:
            major = 3
            minor = 13

        monkeypatch.setattr(sys, "version_info", MockVersionInfo())

        class MockPythonVersion:
            PY_313 = "3.13"
            PY_312 = "3.12"

        result = _get_python_version_enum(MockPythonVersion)
        assert result == "3.13"

    def test_python_314(self, monkeypatch):
        """Test Python 3.14 detection."""
        from lionpride.libs.schema_handlers._schema_to_model import _get_python_version_enum

        class MockVersionInfo:
            major = 3
            minor = 14

        monkeypatch.setattr(sys, "version_info", MockVersionInfo())

        class MockPythonVersion:
            PY_314 = "3.14"
            PY_312 = "3.12"

        result = _get_python_version_enum(MockPythonVersion)
        assert result == "3.14"

    def test_fallback_when_enum_missing(self, monkeypatch):
        """Test fallback when specific enum value is missing."""
        from lionpride.libs.schema_handlers._schema_to_model import _get_python_version_enum

        class MockVersionInfo:
            major = 3
            minor = 15  # Future version not in map

        monkeypatch.setattr(sys, "version_info", MockVersionInfo())

        class MockPythonVersion:
            PY_312 = "3.12"

        result = _get_python_version_enum(MockPythonVersion)
        assert result == "3.12"


class TestPrepareSchemaInput:
    """Test _prepare_schema_input function."""

    def test_dict_schema(self):
        """Test preparation of dict schema."""
        from lionpride.libs.schema_handlers._schema_to_model import _prepare_schema_input

        schema = {"title": "User", "type": "object", "properties": {"name": {"type": "string"}}}

        schema_json, schema_dict, name = _prepare_schema_input(schema, "Default")

        assert schema_dict == schema
        assert '"title"' in schema_json or "'title'" in schema_json.replace('"', "'")
        assert name == "User"

    def test_string_schema(self):
        """Test preparation of JSON string schema."""
        from lionpride.libs.schema_handlers._schema_to_model import _prepare_schema_input

        schema_str = '{"title": "Product", "type": "object"}'

        schema_json, schema_dict, name = _prepare_schema_input(schema_str, "Default")

        assert schema_json == schema_str
        assert schema_dict["title"] == "Product"
        assert name == "Product"

    def test_dict_without_title_uses_default(self):
        """Test dict without title uses default name."""
        from lionpride.libs.schema_handlers._schema_to_model import _prepare_schema_input

        schema = {"type": "object", "properties": {}}

        _, _, name = _prepare_schema_input(schema, "MyDefault")

        assert name == "MyDefault"

    def test_invalid_dict_raises(self):
        """Test invalid dict (non-serializable) raises ValueError."""
        from lionpride.libs.schema_handlers._schema_to_model import _prepare_schema_input

        # Non-serializable dict
        schema = {"func": lambda x: x}

        with pytest.raises(ValueError, match="Invalid dictionary"):
            _prepare_schema_input(schema, "Default")

    def test_invalid_json_string_raises(self):
        """Test invalid JSON string raises ValueError."""
        from lionpride.libs.schema_handlers._schema_to_model import _prepare_schema_input

        invalid_json = "{not valid json"

        with pytest.raises(ValueError, match="Invalid JSON schema"):
            _prepare_schema_input(invalid_json, "Default")

    def test_invalid_type_raises(self):
        """Test invalid schema type raises TypeError."""
        from lionpride.libs.schema_handlers._schema_to_model import _prepare_schema_input

        with pytest.raises(TypeError, match="Schema must be"):
            _prepare_schema_input(123, "Default")

        with pytest.raises(TypeError, match="Schema must be"):
            _prepare_schema_input(["list", "schema"], "Default")

        with pytest.raises(TypeError, match="Schema must be"):
            _prepare_schema_input(None, "Default")


# ===========================================================================
# Module Loading Tests (no dependency required)
# ===========================================================================


class TestLoadGeneratedModule:
    """Test _load_generated_module function."""

    def test_file_not_exists_raises(self, tmp_path):
        """Test FileNotFoundError when file doesn't exist."""
        from lionpride.libs.schema_handlers._schema_to_model import _load_generated_module

        non_existent = tmp_path / "nonexistent.py"
        with pytest.raises(FileNotFoundError, match="Generated model file not created"):
            _load_generated_module(non_existent, "test_module")

    def test_spec_creation_failure(self, tmp_path, monkeypatch):
        """Test ImportError when spec creation fails."""
        from lionpride.libs.schema_handlers._schema_to_model import _load_generated_module

        test_file = tmp_path / "test.py"
        test_file.write_text("# empty")

        # Mock spec_from_file_location to return None
        monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)

        with pytest.raises(ImportError, match="Could not create module spec"):
            _load_generated_module(test_file, "test_module")

    def test_spec_loader_none(self, tmp_path, monkeypatch):
        """Test ImportError when spec.loader is None."""
        from lionpride.libs.schema_handlers._schema_to_model import _load_generated_module

        test_file = tmp_path / "test.py"
        test_file.write_text("# empty")

        # Create a spec with None loader
        class MockSpec:
            loader = None

        monkeypatch.setattr(
            importlib.util, "spec_from_file_location", lambda *args, **kwargs: MockSpec()
        )

        with pytest.raises(ImportError, match="Could not create module spec"):
            _load_generated_module(test_file, "test_module")

    def test_exec_module_failure(self, tmp_path):
        """Test RuntimeError when module execution fails."""
        from lionpride.libs.schema_handlers._schema_to_model import _load_generated_module

        # Create a file with invalid syntax
        test_file = tmp_path / "bad_syntax.py"
        test_file.write_text("def broken(\n")  # Intentionally invalid

        with pytest.raises(RuntimeError, match="Failed to load generated module"):
            _load_generated_module(test_file, "test_module")

    def test_successful_load(self, tmp_path):
        """Test successful module loading."""
        from lionpride.libs.schema_handlers._schema_to_model import _load_generated_module

        test_file = tmp_path / "valid_module.py"
        test_file.write_text("x = 42\ndef hello(): return 'world'\n")

        module = _load_generated_module(test_file, "valid_module")

        assert module.x == 42
        assert module.hello() == "world"

    def test_load_module_with_basemodel(self, tmp_path):
        """Test loading module containing BaseModel."""
        from lionpride.libs.schema_handlers._schema_to_model import _load_generated_module

        test_file = tmp_path / "model_module.py"
        test_file.write_text(
            """
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
"""
        )

        module = _load_generated_module(test_file, "model_module")

        assert hasattr(module, "User")
        assert issubclass(module.User, BaseModel)


class TestExtractModelClass:
    """Test _extract_model_class function."""

    def test_find_by_name(self, tmp_path):
        """Test finding model class by exact name."""
        from lionpride.libs.schema_handlers._schema_to_model import _extract_model_class

        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str
"""
        )

        spec = importlib.util.spec_from_file_location("test_module", str(test_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model_class = _extract_model_class(module, "UserModel", test_file)

        assert model_class.__name__ == "UserModel"
        assert issubclass(model_class, BaseModel)

    def test_fallback_to_model(self, tmp_path):
        """Test fallback to 'Model' class when name not found."""
        from lionpride.libs.schema_handlers._schema_to_model import _extract_model_class

        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
from pydantic import BaseModel

class Model(BaseModel):
    value: str
"""
        )

        spec = importlib.util.spec_from_file_location("test_module", str(test_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model_class = _extract_model_class(module, "NotFound", test_file)

        assert model_class.__name__ == "Model"
        assert issubclass(model_class, BaseModel)

    def test_name_not_basemodel_raises(self, tmp_path):
        """Test TypeError when named class is not BaseModel."""
        from lionpride.libs.schema_handlers._schema_to_model import _extract_model_class

        test_file = tmp_path / "test.py"
        test_file.write_text("class TestModel:\n    pass\n")

        spec = importlib.util.spec_from_file_location("test_module", str(test_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(TypeError, match="is not a Pydantic BaseModel class"):
            _extract_model_class(module, "TestModel", test_file)

    def test_fallback_model_not_basemodel_raises(self, tmp_path):
        """Test TypeError when fallback Model is not BaseModel."""
        from lionpride.libs.schema_handlers._schema_to_model import _extract_model_class

        test_file = tmp_path / "test.py"
        test_file.write_text("class Model:\n    pass\n")

        spec = importlib.util.spec_from_file_location("test_module", str(test_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(TypeError, match="Fallback 'Model' is not a Pydantic BaseModel class"):
            _extract_model_class(module, "NotFound", test_file)

    def test_no_model_found_raises(self, tmp_path):
        """Test AttributeError when no BaseModel classes exist."""
        from lionpride.libs.schema_handlers._schema_to_model import _extract_model_class

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\nclass NotAModel:\n    pass\n")

        spec = importlib.util.spec_from_file_location("test_module", str(test_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(AttributeError, match=r"Could not find.*Available BaseModel classes"):
            _extract_model_class(module, "NotFound", test_file)

    def test_lists_available_models(self, tmp_path):
        """Test error message lists available BaseModel classes."""
        from lionpride.libs.schema_handlers._schema_to_model import _extract_model_class

        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
from pydantic import BaseModel

class OtherModel(BaseModel):
    value: str

class AnotherModel(BaseModel):
    name: str
"""
        )

        spec = importlib.util.spec_from_file_location("test_module", str(test_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(AttributeError) as exc_info:
            _extract_model_class(module, "NotFound", test_file)

        error_msg = str(exc_info.value)
        assert "OtherModel" in error_msg or "AnotherModel" in error_msg


class TestRebuildModel:
    """Test _rebuild_model function."""

    def test_successful_rebuild(self):
        """Test successful model rebuild."""
        from lionpride.libs.schema_handlers._schema_to_model import _rebuild_model

        class TestModel(BaseModel):
            name: str
            value: int

        mock_module = SimpleNamespace(__dict__={"str": str, "int": int})

        # Should not raise
        _rebuild_model(TestModel, mock_module, "TestModel")

    def test_type_resolution_failure_nameerror(self, monkeypatch):
        """Test RuntimeError on NameError during rebuild."""
        from lionpride.libs.schema_handlers._schema_to_model import _rebuild_model

        class TestModel(BaseModel):
            pass

        mock_module = SimpleNamespace(__dict__={})

        def mock_rebuild(*args, **kwargs):
            raise NameError("Type resolution error")

        monkeypatch.setattr(TestModel, "model_rebuild", mock_rebuild)

        with pytest.raises(RuntimeError, match="Type resolution failed during rebuild"):
            _rebuild_model(TestModel, mock_module, "TestModel")

    def test_type_resolution_failure_pydantic_error(self, monkeypatch):
        """Test RuntimeError on PydanticUserError during rebuild."""
        from lionpride.libs.schema_handlers._schema_to_model import _rebuild_model

        class TestModel(BaseModel):
            pass

        mock_module = SimpleNamespace(__dict__={})

        def mock_rebuild(*args, **kwargs):
            raise PydanticUserError("Pydantic error", code="model-field-missing-annotation")

        monkeypatch.setattr(TestModel, "model_rebuild", mock_rebuild)

        with pytest.raises(RuntimeError, match="Type resolution failed during rebuild"):
            _rebuild_model(TestModel, mock_module, "TestModel")

    def test_unexpected_error(self, monkeypatch):
        """Test RuntimeError on unexpected exception."""
        from lionpride.libs.schema_handlers._schema_to_model import _rebuild_model

        class TestModel(BaseModel):
            pass

        mock_module = SimpleNamespace(__dict__={})

        def mock_rebuild(*args, **kwargs):
            raise ValueError("Unexpected error")

        monkeypatch.setattr(TestModel, "model_rebuild", mock_rebuild)

        with pytest.raises(RuntimeError, match="Unexpected error during model_rebuild"):
            _rebuild_model(TestModel, mock_module, "TestModel")


# ===========================================================================
# Code Generation Tests (no dependency required - mocked)
# ===========================================================================


class TestGenerateModelCode:
    """Test _generate_model_code function."""

    def test_generate_model_code_calls_generate(self):
        """Test that _generate_model_code calls the generate function correctly."""
        from lionpride.libs.schema_handlers._schema_to_model import _generate_model_code

        mock_generate = mock.Mock()
        mock_pydantic_version = mock.Mock()
        mock_python_version = mock.Mock()
        mock_input_file_type = mock.Mock()
        mock_input_file_type.JsonSchema = "json_schema"

        output_file = Path("/tmp/test.py")
        schema_json = '{"type": "object"}'

        _generate_model_code(
            schema_json,
            output_file,
            mock_pydantic_version,
            mock_python_version,
            mock_generate,
            mock_input_file_type,
        )

        mock_generate.assert_called_once_with(
            schema_json,
            input_file_type="json_schema",
            input_filename="schema.json",
            output=output_file,
            output_model_type=mock_pydantic_version,
            target_python_version=mock_python_version,
            base_class="pydantic.BaseModel",
        )


# ===========================================================================
# Import Error Handling Tests
# ===========================================================================


class TestImportErrorHandling:
    """Test behavior when optional dependency is missing."""

    def test_import_error_message(self, monkeypatch):
        """Verify helpful error when datamodel-code-generator not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "datamodel_code_generator":
                raise ImportError("No module named 'datamodel_code_generator'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {"title": "Test", "type": "object"}

        with pytest.raises(ImportError, match="datamodel-code-generator not installed"):
            load_pydantic_model_from_schema(schema)


class TestLoadPydanticModelFromSchemaMocked:
    """Test load_pydantic_model_from_schema with mocked imports."""

    def test_full_workflow_mocked(self, tmp_path, monkeypatch):
        """Test full workflow with mocked datamodel_code_generator."""
        import builtins

        # Create a mock model file that will be "generated"
        model_code = """
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
"""
        # Track what file was written
        generated_files = []

        def mock_generate(
            schema_json,
            input_file_type,
            input_filename,
            output,
            output_model_type,
            target_python_version,
            base_class,
        ):
            generated_files.append(output)
            output.write_text(model_code)

        # Create mock module
        mock_datamodel = mock.MagicMock()
        mock_datamodel.generate = mock_generate
        mock_datamodel.DataModelType = mock.MagicMock()
        mock_datamodel.DataModelType.PydanticV2BaseModel = "PydanticV2BaseModel"
        mock_datamodel.InputFileType = mock.MagicMock()
        mock_datamodel.InputFileType.JsonSchema = "JsonSchema"
        mock_datamodel.PythonVersion = mock.MagicMock()
        mock_datamodel.PythonVersion.PY_312 = "3.12"

        original_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datamodel_code_generator":
                return mock_datamodel
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {"title": "User", "type": "object", "properties": {"name": {"type": "string"}}}

        # The function uses tempfile, so we need to ensure our mock generate writes there
        # Patch tempfile to use our tmp_path
        with mock.patch("tempfile.TemporaryDirectory") as mock_tmpdir:
            mock_tmpdir.return_value.__enter__.return_value = str(tmp_path)
            mock_tmpdir.return_value.__exit__.return_value = False

            # Now the generated file should be in tmp_path
            result = load_pydantic_model_from_schema(schema)

            assert issubclass(result, BaseModel)

    def test_uses_custom_pydantic_version(self, monkeypatch):
        """Test custom pydantic_version parameter is used."""
        import builtins

        captured_args = {}

        def mock_generate(
            schema_json,
            input_file_type,
            input_filename,
            output,
            output_model_type,
            target_python_version,
            base_class,
        ):
            captured_args.update(
                {
                    "schema_json": schema_json,
                    "input_file_type": input_file_type,
                    "input_filename": input_filename,
                    "output": output,
                    "output_model_type": output_model_type,
                    "target_python_version": target_python_version,
                    "base_class": base_class,
                }
            )
            output.write_text("from pydantic import BaseModel\nclass Model(BaseModel):\n    pass\n")

        mock_datamodel = mock.MagicMock()
        mock_datamodel.generate = mock_generate
        mock_datamodel.DataModelType = mock.MagicMock()
        mock_datamodel.DataModelType.PydanticV2BaseModel = "PydanticV2BaseModel"
        mock_datamodel.InputFileType = mock.MagicMock()
        mock_datamodel.InputFileType.JsonSchema = "JsonSchema"
        mock_datamodel.PythonVersion = mock.MagicMock()
        mock_datamodel.PythonVersion.PY_312 = "3.12"

        original_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datamodel_code_generator":
                return mock_datamodel
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        custom_version = "CustomPydanticVersion"

        load_pydantic_model_from_schema(
            {"type": "object"}, pydantic_version=custom_version, python_version="3.12"
        )

        assert captured_args["output_model_type"] == custom_version
        assert captured_args["target_python_version"] == "3.12"

    def test_uses_default_versions(self, monkeypatch):
        """Test default pydantic and python versions are used when not specified."""
        import builtins

        captured_args = {}

        def mock_generate(
            schema_json,
            input_file_type,
            input_filename,
            output,
            output_model_type,
            target_python_version,
            base_class,
        ):
            captured_args.update(
                {
                    "schema_json": schema_json,
                    "input_file_type": input_file_type,
                    "input_filename": input_filename,
                    "output": output,
                    "output_model_type": output_model_type,
                    "target_python_version": target_python_version,
                    "base_class": base_class,
                }
            )
            output.write_text("from pydantic import BaseModel\nclass Model(BaseModel):\n    pass\n")

        mock_datamodel = mock.MagicMock()
        mock_datamodel.generate = mock_generate
        mock_datamodel.DataModelType = mock.MagicMock()
        mock_datamodel.DataModelType.PydanticV2BaseModel = "DefaultPydanticV2"
        mock_datamodel.InputFileType = mock.MagicMock()
        mock_datamodel.InputFileType.JsonSchema = "JsonSchema"
        mock_datamodel.PythonVersion = mock.MagicMock()
        mock_datamodel.PythonVersion.PY_312 = "DefaultPython312"

        original_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datamodel_code_generator":
                return mock_datamodel
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        load_pydantic_model_from_schema({"type": "object"})

        # Should use DataModelType.PydanticV2BaseModel
        assert captured_args["output_model_type"] == "DefaultPydanticV2"


# ===========================================================================
# Integration Tests (require datamodel-code-generator)
# ===========================================================================


@pytest.mark.skipif(
    not _HAS_SCHEMA_GEN,
    reason="datamodel-code-generator not installed (optional dependency)",
)
class TestLoadPydanticModelFromSchemaIntegration:
    """Integration tests for load_pydantic_model_from_schema."""

    def test_simple_schema_dict(self):
        """Load model from simple dict schema."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "User",
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

        UserModel = load_pydantic_model_from_schema(schema)

        assert issubclass(UserModel, BaseModel)
        assert UserModel.__name__ == "User"

        user = UserModel(name="Alice", age=30)
        assert user.name == "Alice"
        assert user.age == 30

        with pytest.raises(ValidationError):
            UserModel(age=25)  # missing required name

    def test_simple_schema_string(self):
        """Load model from JSON string schema."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = """{
            "title": "Product",
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "price": {"type": "number"}
            },
            "required": ["id", "name"]
        }"""

        ProductModel = load_pydantic_model_from_schema(schema)

        assert ProductModel.__name__ == "Product"

        product = ProductModel(id=1, name="Widget", price=9.99)
        assert product.id == 1
        assert product.name == "Widget"
        assert product.price == 9.99

    def test_nested_schema(self):
        """Load model with nested objects."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Company",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "string"},
                    },
                },
            },
        }

        CompanyModel = load_pydantic_model_from_schema(schema)

        company = CompanyModel(
            name="Acme Corp", address={"street": "123 Main St", "city": "Springfield"}
        )
        assert company.name == "Acme Corp"
        assert company.address.street == "123 Main St"
        assert company.address.city == "Springfield"

    def test_schema_with_array(self):
        """Load model with array fields."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Team",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "members": {"type": "array", "items": {"type": "string"}},
            },
        }

        TeamModel = load_pydantic_model_from_schema(schema)

        team = TeamModel(name="Engineering", members=["Alice", "Bob", "Charlie"])
        assert team.name == "Engineering"
        assert len(team.members) == 3
        assert team.members[0] == "Alice"

    def test_schema_with_enum(self):
        """Load model with enum constraints."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Task",
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            },
        }

        TaskModel = load_pydantic_model_from_schema(schema)

        task = TaskModel(title="Write tests", status="in_progress")
        # Generator creates Enum, check value
        assert task.status.value == "in_progress"  # type: ignore[attr-defined]

        with pytest.raises(ValidationError):
            TaskModel(title="Invalid", status="invalid_status")

    def test_custom_model_name(self):
        """Generator falls back to 'Model' when schema has no title."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {"type": "object", "properties": {"value": {"type": "string"}}}

        Model = load_pydantic_model_from_schema(schema, "CustomModel")

        # Generator ignores model_name param, uses 'Model' fallback
        assert Model.__name__ == "Model"
        instance = Model(value="test")
        assert instance.value == "test"

    def test_schema_title_with_spaces(self):
        """Handle schema titles with spaces."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "My Custom Model",
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }

        Model = load_pydantic_model_from_schema(schema)

        assert Model.__name__ == "MyCustomModel"

    def test_schema_title_with_special_chars(self):
        """Sanitize schema titles with special characters."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "User@Profile#123",
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        Model = load_pydantic_model_from_schema(schema)

        # Special chars should be stripped
        assert Model.__name__ == "UserProfile123"

    def test_model_rebuild_with_forward_refs(self):
        """Ensure model_rebuild resolves forward references."""
        from lionpride.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Node",
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
                "children": {"type": "array", "items": {"$ref": "#"}},
            },
        }

        NodeModel = load_pydantic_model_from_schema(schema)

        # Should handle recursive structure
        node = NodeModel(value=1, children=[{"value": 2, "children": []}])
        assert node.value == 1
        assert len(node.children) == 1
        assert node.children[0].value == 2


# ===========================================================================
# Lazy Import Tests
# ===========================================================================


class TestLazyImport:
    """Test lazy import via __getattr__."""

    def test_lazy_import_works(self):
        """Verify function is accessible via lazy import."""
        from lionpride.libs import schema_handlers

        # Should not raise even if accessed via __getattr__
        func = schema_handlers.load_pydantic_model_from_schema
        assert callable(func)

    def test_invalid_attribute_raises(self):
        """Verify __getattr__ raises for invalid attributes."""
        from lionpride.libs import schema_handlers

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = schema_handlers.nonexistent_function  # type: ignore[attr-defined]
