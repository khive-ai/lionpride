# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import string
import sys
import tempfile
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, PydanticUserError

from lionpride.ln import json_dumps, to_dict

B = TypeVar("B", bound=BaseModel)

# Security limits to prevent DoS
MAX_SCHEMA_SIZE = 1024 * 1024  # 1 MB default limit
MAX_SCHEMA_DEPTH = 50  # Maximum nesting depth


def _check_schema_depth(
    obj: Any,
    current_depth: int = 0,
    *,
    max_depth: int = MAX_SCHEMA_DEPTH,
) -> int:
    """Check maximum nesting depth of a schema structure.

    Args:
        obj: The object to check (dict, list, or primitive)
        current_depth: Current recursion depth
        max_depth: Maximum allowed depth

    Returns:
        Maximum depth found

    Raises:
        ValueError: If depth exceeds max_depth
    """
    if current_depth > max_depth:
        msg = f"Schema nesting depth exceeds maximum allowed ({max_depth})"
        raise ValueError(msg)

    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(
            _check_schema_depth(v, current_depth + 1, max_depth=max_depth) for v in obj.values()
        )
    if isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_check_schema_depth(v, current_depth + 1, max_depth=max_depth) for v in obj)
    return current_depth


def _get_python_version_enum(python_version_module: Any) -> Any:
    """Auto-detect Python version from environment."""
    version_info = sys.version_info
    version_map = {
        (3, 11): "PY_311",
        (3, 12): "PY_312",
        (3, 13): "PY_313",
        (3, 14): "PY_314",
    }
    version_key = (version_info.major, version_info.minor)
    enum_name = version_map.get(version_key, "PY_312")
    return getattr(python_version_module, enum_name, python_version_module.PY_312)


def _sanitize_model_name(name: str) -> str:
    """Extract valid Python identifier from string.

    Raises:
        ValueError: If name cannot be converted to valid Python identifier.
    """
    valid_chars = string.ascii_letters + string.digits + "_"
    sanitized = "".join(c for c in name.replace(" ", "") if c in valid_chars)

    if not sanitized or not sanitized[0].isalpha():
        msg = f"Cannot extract valid Python identifier from: {name!r}"
        raise ValueError(msg)

    return sanitized


def _extract_model_name_from_schema(schema_dict: dict[str, Any], default: str) -> str:
    """Extract model name from schema title or use default."""
    title = schema_dict.get("title")
    if title and isinstance(title, str):
        try:
            return _sanitize_model_name(title)
        except ValueError:
            pass  # Fall back to default if title cannot be sanitized
    return default


def _prepare_schema_input(
    schema: str | dict[str, Any],
    model_name: str,
    *,
    max_size: int = MAX_SCHEMA_SIZE,
    max_depth: int = MAX_SCHEMA_DEPTH,
) -> tuple[str, dict[str, Any], str]:
    """Convert schema to JSON string and extract model name.

    Args:
        schema: JSON schema as string or dict
        model_name: Default model name if schema has no title
        max_size: Maximum schema size in bytes (default: 1MB)
        max_depth: Maximum nesting depth (default: 50)

    Raises:
        ValueError: If schema exceeds size/depth limits or is invalid
        TypeError: If schema is not a string or dict
    """
    schema_dict: dict[str, Any]
    schema_json: str
    if isinstance(schema, dict):
        try:
            schema_dict = schema
            schema_json = cast(str, json_dumps(schema_dict))
        except TypeError as e:
            msg = "Invalid dictionary provided for schema"
            raise ValueError(msg) from e
    elif isinstance(schema, str):
        # Check size before parsing
        if len(schema) > max_size:
            msg = f"Schema size ({len(schema)} bytes) exceeds maximum ({max_size} bytes)"
            raise ValueError(msg)
        try:
            schema_dict = cast(dict[str, Any], to_dict(schema))
        except Exception as e:
            msg = "Invalid JSON schema string provided"
            raise ValueError(msg) from e
        schema_json = schema
    else:
        msg = "Schema must be a JSON string or a dictionary"
        raise TypeError(msg)

    # Validate size (for dict input, check serialized size)
    if len(schema_json) > max_size:
        msg = f"Schema size ({len(schema_json)} bytes) exceeds maximum ({max_size} bytes)"
        raise ValueError(msg)

    # Validate depth to prevent stack overflow during processing
    _check_schema_depth(schema_dict, max_depth=max_depth)

    resolved_name = _extract_model_name_from_schema(schema_dict, model_name)
    return schema_json, schema_dict, resolved_name


def _generate_model_code(
    schema_json: str,
    output_file: Path,
    pydantic_version: Any,
    python_version: Any,
    generate_func: Any,
    input_file_type_enum: Any,
) -> None:
    """Generate Pydantic model code from schema."""
    try:
        generate_func(
            schema_json,
            input_file_type=input_file_type_enum.JsonSchema,
            input_filename="schema.json",
            output=output_file,
            output_model_type=pydantic_version,
            target_python_version=python_version,
            base_class="pydantic.BaseModel",
        )
    except Exception as e:  # pragma: no cover
        msg = "Failed to generate model code from schema"
        raise RuntimeError(msg) from e


def _load_generated_module(output_file: Path, module_name: str) -> Any:
    """Dynamically import generated Python module."""
    if not output_file.exists():
        msg = f"Generated model file not created: {output_file}"
        raise FileNotFoundError(msg)

    spec = importlib.util.spec_from_file_location(module_name, str(output_file))
    if spec is None or spec.loader is None:
        msg = f"Could not create module spec for {output_file}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        msg = f"Failed to load generated module from {output_file}"
        raise RuntimeError(msg) from e

    return module


def _extract_model_class(
    module: Any,
    model_name: str,
    output_file: Path,
) -> type[BaseModel]:
    """Find BaseModel class in generated module."""

    def _is_valid_model(obj: Any) -> bool:
        return isinstance(obj, type) and issubclass(obj, BaseModel)

    try:
        model_class = getattr(module, model_name)
        if not _is_valid_model(model_class):
            msg = f"'{model_name}' is not a Pydantic BaseModel class"
            raise TypeError(msg)
        return model_class
    except AttributeError:
        pass

    # Fallback to 'Model'
    try:
        model_class = module.Model
        if not _is_valid_model(model_class):
            msg = "Fallback 'Model' is not a Pydantic BaseModel class"
            raise TypeError(msg)
        return model_class
    except AttributeError:
        pass

    # List available models for debugging
    available = [
        attr
        for attr in dir(module)
        if _is_valid_model(getattr(module, attr, None)) and getattr(module, attr) is not BaseModel
    ]
    msg = (
        f"Could not find '{model_name}' or 'Model' in {output_file}. "
        f"Available BaseModel classes: {available}"
    )
    raise AttributeError(msg)


def _rebuild_model(model_class: type[BaseModel], module: Any, model_name: str) -> None:
    """Rebuild model with proper type resolution."""
    try:
        model_class.model_rebuild(_types_namespace=module.__dict__, force=True)
    except (PydanticUserError, NameError) as e:
        msg = f"Type resolution failed during rebuild for {model_name}"
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Unexpected error during model_rebuild for {model_name}"
        raise RuntimeError(msg) from e


def load_pydantic_model_from_schema(
    schema: str | dict[str, Any],
    model_name: str = "DynamicModel",
    /,
    pydantic_version: Any = None,
    python_version: Any = None,
    *,
    max_size: int = MAX_SCHEMA_SIZE,
    max_depth: int = MAX_SCHEMA_DEPTH,
) -> type[BaseModel]:
    """Generate Pydantic model dynamically from JSON schema.

    This function uses datamodel-code-generator to generate Python code from
    the JSON schema, then dynamically loads and executes that code. This is
    inherently a code execution operation.

    Security Considerations:
        - The schema is converted to Python code by datamodel-code-generator
        - That code is executed in a temporary module
        - Size and depth limits are enforced to prevent DoS attacks
        - The generated code quality depends on datamodel-code-generator

    Trust Boundary:
        This function trusts datamodel-code-generator to produce safe code.
        Do not use with schemas from untrusted sources without careful review.
        The max_size and max_depth parameters help prevent resource exhaustion
        but cannot prevent malicious code if the generator is compromised.

    Args:
        schema: JSON schema (string or dict)
        model_name: Base name for model (schema title takes precedence)
        pydantic_version: Pydantic version for code generation (default: v2)
        python_version: Target Python version (default: auto-detect)
        max_size: Maximum schema size in bytes (default: 1MB)
        max_depth: Maximum schema nesting depth (default: 50)

    Returns:
        Generated Pydantic model class

    Raises:
        ImportError: datamodel-code-generator not installed
        ValueError: Schema exceeds size/depth limits or is invalid
        TypeError: Schema is not a string or dict
        RuntimeError: Code generation or model loading failed
    """
    try:
        from datamodel_code_generator import (
            DataModelType,
            InputFileType,
            PythonVersion,
            generate,
        )
    except ImportError as e:
        msg = (
            "datamodel-code-generator not installed. "
            "Install with: pip install 'lionpride-core[schema-gen]' "
            "or: pip install datamodel-code-generator"
        )
        raise ImportError(msg) from e

    pydantic_version = pydantic_version or DataModelType.PydanticV2BaseModel
    python_version = python_version or _get_python_version_enum(PythonVersion)

    schema_json, _, resolved_name = _prepare_schema_input(
        schema, model_name, max_size=max_size, max_depth=max_depth
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Use cryptographic hash to avoid collisions and predictability
        schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()[:16]
        output_file = tmpdir_path / f"{resolved_name.lower()}_model_{schema_hash}.py"
        module_name = output_file.stem

        _generate_model_code(
            schema_json,
            output_file,
            pydantic_version,
            python_version,
            generate,
            InputFileType,
        )
        module = _load_generated_module(output_file, module_name)
        model_class = _extract_model_class(module, resolved_name, output_file)
        _rebuild_model(model_class, module, resolved_name)

        return model_class
