# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for type system tests.

Provides fixtures for Spec, Params, DataClass, and HashableModel testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from lionpride.types import DataClass, HashableModel, ModelConfig, Params, Spec, Unset

__all__ = (
    "create_spec",
    "get_sample_hashable_models",
    "get_sample_params_classes",
    "get_sample_validators",
)


# =============================================================================
# Spec Factory
# =============================================================================


def create_spec(
    base_type: type,
    name: str,
    *,
    default: Any = None,
    nullable: bool = False,
    listable: bool = False,
    validator: Any = None,
    description: str | None = None,
    **kwargs: Any,
) -> Spec:
    """Factory for Spec creation in tests."""
    spec_kwargs: dict[str, Any] = {"name": name}

    if default is not None:
        spec_kwargs["default"] = default
    if nullable:
        spec_kwargs["nullable"] = nullable
    if listable:
        spec_kwargs["listable"] = listable
    if validator is not None:
        spec_kwargs["validator"] = validator
    if description is not None:
        spec_kwargs["description"] = description
    spec_kwargs.update(kwargs)

    return Spec(base_type, **spec_kwargs)


# =============================================================================
# Sample Validators
# =============================================================================


def get_sample_validators() -> dict[str, Any]:
    """Returns dict of common validator functions for testing."""

    def nonneg(v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    def string_length(v: str) -> str:
        if len(v) == 0:
            raise ValueError("string cannot be empty")
        return v

    def email_format(v: str) -> str:
        if "@" not in v:
            raise ValueError("invalid email format")
        return v

    def range_validator(v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("value must be between 0 and 100")
        return v

    return {
        "nonneg": nonneg,
        "string_length": string_length,
        "email_format": email_format,
        "range_validator": range_validator,
    }


# =============================================================================
# Sample Params Classes
# =============================================================================


def get_sample_params_classes() -> dict[str, type]:
    """Returns dict of Params/DataClass test fixtures."""

    @dataclass(slots=True, frozen=True, init=False)
    class MyParams(Params):
        field1: str = Unset
        field2: int = Unset
        field3: bool = Unset

    @dataclass(slots=True, frozen=True, init=False)
    class MyParamsNoneSentinel(Params):
        _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
        field1: str = Unset

    @dataclass(slots=True, frozen=True, init=False)
    class MyParamsStrict(Params):
        _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
        field1: str = Unset
        field2: int = Unset

    @dataclass(slots=True)
    class MyDataClass(DataClass):
        field1: str = Unset
        field2: int = Unset

    @dataclass(slots=True)
    class MyDataClassStrict(DataClass):
        _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
        field1: str = Unset
        field2: int = Unset

    @dataclass(slots=True)
    class MyDataClassPrefillUnset(DataClass):
        _config: ClassVar[ModelConfig] = ModelConfig(prefill_unset=True)
        field1: str = Unset
        field2: int = Unset

    return {
        "basic_params": MyParams,
        "params_none_sentinel": MyParamsNoneSentinel,
        "params_strict": MyParamsStrict,
        "basic_dataclass": MyDataClass,
        "dataclass_strict": MyDataClassStrict,
        "dataclass_prefill": MyDataClassPrefillUnset,
    }


# =============================================================================
# Sample HashableModel Classes
# =============================================================================


def get_sample_hashable_models() -> dict[str, type]:
    """Returns dict of HashableModel test fixtures."""

    class SimpleConfig(HashableModel):
        name: str
        value: int

    class NestedConfig(HashableModel):
        config: SimpleConfig
        enabled: bool = True

    class ConfigWithOptional(HashableModel):
        required: str
        optional: str | None = None

    return {
        "simple": SimpleConfig,
        "nested": NestedConfig,
        "optional": ConfigWithOptional,
    }


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def sample_validators():
    """Sample validator functions for testing."""
    return get_sample_validators()


@pytest.fixture
def sample_params_classes():
    """Sample Params/DataClass classes for testing."""
    return get_sample_params_classes()


@pytest.fixture
def sample_hashable_models():
    """Sample HashableModel classes for testing."""
    return get_sample_hashable_models()


@pytest.fixture
def spec_factory():
    """Factory fixture for creating Spec instances."""
    return create_spec
