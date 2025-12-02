# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import types
from functools import reduce
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from .._sentinel import Unset, is_sentinel
from ._protocol import SpecAdapter

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    from ..operable import Operable
    from ..spec import Spec


@functools.lru_cache(maxsize=1)
def _get_pydantic_field_params() -> set[str]:
    """Get valid Pydantic Field params (cached)."""
    import inspect

    from pydantic import Field as PydanticField

    params = set(inspect.signature(PydanticField).parameters.keys())
    params.discard("kwargs")
    return params


# --------------------------------------------------------------------------
# Helper functions for extracting Specs from Pydantic models
# --------------------------------------------------------------------------


def _is_nullable_type(annotation: Any) -> tuple[bool, Any]:
    """Check if annotation is Optional/nullable and extract inner type.

    Returns:
        Tuple of (is_nullable, inner_type_or_original)
    """
    origin = get_origin(annotation)

    # Handle Union types (including Optional which is Union[X, None])
    if origin is type(None):
        return True, type(None)

    if origin in (type(int | str), types.UnionType):  # Python 3.10+ union syntax
        args = get_args(annotation)
        non_none_args = [a for a in args if a is not type(None)]
        if len(args) != len(non_none_args):  # None was in the union
            if len(non_none_args) == 1:
                return True, non_none_args[0]
            # Multiple non-None types in union, keep as union minus None
            if non_none_args:
                return True, reduce(lambda a, b: a | b, non_none_args)
        return False, annotation

    # typing.Union
    if origin is Union:
        args = get_args(annotation)
        non_none_args = [a for a in args if a is not type(None)]
        if len(args) != len(non_none_args):
            if len(non_none_args) == 1:
                return True, non_none_args[0]
            if non_none_args:
                return True, reduce(lambda a, b: a | b, non_none_args)
        return False, annotation

    return False, annotation


def _is_list_type(annotation: Any) -> tuple[bool, Any]:
    """Check if annotation is a list type and extract element type.

    Returns:
        Tuple of (is_list, element_type_or_original)
    """
    origin = get_origin(annotation)

    if origin is list:
        args = get_args(annotation)
        if args:
            return True, args[0]
        return True, Any

    return False, annotation


# Direct FieldInfo attributes to preserve
_FIELD_INFO_ATTRS = frozenset(
    {
        # Metadata (stored as direct attributes)
        "alias",
        "validation_alias",
        "serialization_alias",
        "title",
        "description",
        "examples",
        "deprecated",
        "frozen",
        "json_schema_extra",
        "discriminator",  # For discriminated unions
        "exclude",  # Exclude from serialization (important for sensitive fields)
        "repr",  # Control repr output
        "init",  # Control __init__ inclusion
        "init_var",  # Init-only variable
        "kw_only",  # Keyword-only argument
        "validate_default",  # Validate default value
    }
)

# Constraints stored in FieldInfo.metadata as annotated_types objects
_CONSTRAINT_MAPPING = {
    "Gt": "gt",
    "Ge": "ge",
    "Lt": "lt",
    "Le": "le",
    "MultipleOf": "multiple_of",
    "MinLen": "min_length",
    "MaxLen": "max_length",
}


def _field_info_to_spec(field_name: str, field_info: Any) -> Spec:
    """Convert a Pydantic FieldInfo to a Spec.

    Preserves:
    - Type annotation (nullable, listable)
    - Default value or factory
    - Validation constraints (gt, lt, min_length, pattern, etc.)
    - Metadata (description, alias, title, etc.)
    - Requiredness for nullable fields without defaults

    Args:
        field_name: Name of the field
        field_info: Pydantic FieldInfo object

    Returns:
        Spec representing the field
    """
    from pydantic_core import PydanticUndefined

    from ..spec import Spec

    annotation = field_info.annotation

    # Check for nullable (Optional) types
    is_nullable, inner_type = _is_nullable_type(annotation)

    # Check for list types (after unwrapping Optional)
    is_listable, element_type = _is_list_type(inner_type)

    # Determine the base type
    base_type = element_type if is_listable else inner_type

    # Build the spec
    spec = Spec(base_type=base_type, name=field_name)

    if is_listable:
        spec = spec.as_listable()

    if is_nullable:
        spec = spec.as_nullable()

    # Determine if field is required (no default value or factory)
    has_default = field_info.default is not PydanticUndefined
    has_default_factory = (
        field_info.default_factory is not None
        and field_info.default_factory is not PydanticUndefined
    )
    is_required = not has_default and not has_default_factory

    # Handle default value
    if has_default:
        spec = spec.with_default(field_info.default)
    elif has_default_factory:
        spec = spec.with_default(field_info.default_factory)

    # For nullable fields without defaults, mark as required to prevent
    # PydanticSpecAdapter from auto-injecting default=None
    if is_nullable and is_required:
        spec = spec.with_updates(required=True)

    # Preserve FieldInfo direct attributes (metadata)
    updates = {}
    for attr in _FIELD_INFO_ATTRS:
        value = getattr(field_info, attr, None)
        if value is not None:
            updates[attr] = value

    # Extract constraints from FieldInfo.metadata (annotated_types objects)
    # e.g., Gt(gt=0), Lt(lt=150), MinLen(min_length=1)
    if hasattr(field_info, "metadata") and field_info.metadata:
        for constraint in field_info.metadata:
            constraint_type = type(constraint).__name__
            if constraint_type in _CONSTRAINT_MAPPING:
                key = _CONSTRAINT_MAPPING[constraint_type]
                # Get the value from the constraint object
                value = getattr(constraint, key, None)
                if value is not None:
                    updates[key] = value
            # Handle Pydantic's Strict type
            elif constraint_type == "Strict":
                updates["strict"] = getattr(constraint, "strict", True)
            # Handle pattern from _PydanticGeneralMetadata
            elif constraint_type == "_PydanticGeneralMetadata":
                pattern = getattr(constraint, "pattern", None)
                if pattern is not None:
                    updates["pattern"] = pattern
                # Also check for strict in general metadata
                strict = getattr(constraint, "strict", None)
                if strict is not None:
                    updates["strict"] = strict

    if updates:
        spec = spec.with_updates(**updates)

    return spec


class PydanticSpecAdapter(SpecAdapter["BaseModel"]):
    """Pydantic adapter: Spec → FieldInfo → BaseModel."""

    @classmethod
    def create_field(cls, spec: Spec) -> FieldInfo:
        """Create Pydantic FieldInfo from Spec."""
        from pydantic import Field as PydanticField

        # Get valid Pydantic Field parameters (cached)
        pydantic_field_params = _get_pydantic_field_params()

        # Extract metadata for FieldInfo
        field_kwargs = {}

        if not is_sentinel(spec.metadata, none_as_sentinel=True):
            for meta in spec.metadata:
                if meta.key == "default":
                    # Handle callable defaults as default_factory
                    if callable(meta.value):
                        field_kwargs["default_factory"] = meta.value
                    else:
                        field_kwargs["default"] = meta.value
                elif meta.key == "validator":
                    # Validators are handled separately in create_model
                    continue
                elif meta.key in pydantic_field_params:
                    # Pass through standard Pydantic field attributes
                    field_kwargs[meta.key] = meta.value
                elif meta.key in {"nullable", "listable"}:
                    # These are FieldTemplate markers, don't pass to FieldInfo
                    pass
                else:
                    # Filter out unserializable objects from json_schema_extra
                    if isinstance(meta.value, type):
                        # Skip type objects - can't be serialized
                        continue

                    # Any other metadata goes in json_schema_extra
                    if "json_schema_extra" not in field_kwargs:
                        field_kwargs["json_schema_extra"] = {}
                    field_kwargs["json_schema_extra"][meta.key] = meta.value

        # Handle nullable case - ensure default is set if not already
        # BUT respect 'required=True' metadata to preserve requiredness for
        # nullable fields that were originally required (e.g., `foo: str | None`)
        is_required = any(m.key == "required" and m.value for m in spec.metadata)
        if (
            spec.is_nullable
            and "default" not in field_kwargs
            and "default_factory" not in field_kwargs
            and not is_required
        ):
            field_kwargs["default"] = None

        field_info = PydanticField(**field_kwargs)
        field_info.annotation = spec.annotation

        return field_info

    @classmethod
    def create_validator(cls, spec: Spec) -> dict[str, Any] | None:
        """Create Pydantic field_validator from Spec metadata."""
        from .._sentinel import Undefined

        v = spec.get("validator")
        if v is Unset or v is Undefined:
            return None

        from pydantic import field_validator

        field_name = spec.name or "field"
        # check_fields=False allows the validator to be defined in a base class before the field exists
        return {f"{field_name}_validator": field_validator(field_name, check_fields=False)(v)}

    @classmethod
    def create_model(
        cls,
        op: Operable,
        model_name: str,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        base_type: type[BaseModel] | None = None,
        doc: str | None = None,
    ) -> type[BaseModel]:
        """Generate Pydantic BaseModel from Operable using pydantic.create_model()."""
        from pydantic import BaseModel, create_model

        use_specs = op.get_specs(include=include, exclude=exclude)
        use_fields = {i.name: cls.create_field(i) for i in use_specs if i.name}

        # Convert fields to (type, FieldInfo) tuples for create_model
        field_definitions = {
            name: (field_info.annotation, field_info) for name, field_info in use_fields.items()
        }

        # Collect validators
        validators = {}
        for spec in use_specs:
            if spec.name and (validator := cls.create_validator(spec)):
                validators.update(validator)

        # If we have validators, create a base class with them
        # Otherwise use the provided base_type or BaseModel
        if validators:
            # Create a temporary base class with validators as class attributes
            base_with_validators = type(
                f"{model_name}Base",
                (base_type or BaseModel,),
                validators,
            )
            actual_base = base_with_validators
        else:
            actual_base = base_type or BaseModel

        # Create model using pydantic's create_model
        model_cls = create_model(
            model_name,
            __base__=actual_base,
            __doc__=doc,
            **field_definitions,
        )

        model_cls.model_rebuild()
        return model_cls

    @classmethod
    def fuzzy_match_fields(
        cls, data: dict, model_cls: type[BaseModel], strict: bool = False
    ) -> dict[str, Any]:
        """Match data keys to Pydantic fields (fuzzy). Filters sentinels. Args: data, model_cls, strict."""
        from lionpride.ln._fuzzy_match import fuzzy_match_keys

        from .._sentinel import not_sentinel

        # "ignore" mode only includes successfully matched fields (no sentinel injection)
        # "raise" mode raises on unmatched keys for strict validation
        handle_mode = "raise" if strict else "ignore"

        matched = fuzzy_match_keys(data, model_cls.model_fields, handle_unmatched=handle_mode)

        # Filter out sentinel values (Unset, Undefined)
        return {k: v for k, v in matched.items() if not_sentinel(v)}

    @classmethod
    def validate_model(cls, model_cls: type[BaseModel], data: dict) -> BaseModel:
        """Validate dict → Pydantic model via model_validate()."""
        return model_cls.model_validate(data)

    @classmethod
    def dump_model(cls, instance: BaseModel) -> dict[str, Any]:
        """Dump Pydantic model → dict via model_dump()."""
        return instance.model_dump()

    @classmethod
    def specs_from_model(cls, model: type[BaseModel]) -> tuple[Spec, ...]:
        """Extract Specs from a Pydantic model's fields.

        Disassembles a model and returns a tuple of Specs
        representing top-level fields of that model.

        Args:
            model: Pydantic BaseModel class to disassemble

        Returns:
            Tuple of Specs for each top-level field

        Raises:
            TypeError: If model is not a Pydantic BaseModel subclass
        """
        from pydantic import BaseModel

        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(f"model must be a Pydantic BaseModel subclass, got {type(model)}")

        specs: list[Spec] = []

        for field_name, field_info in model.model_fields.items():
            spec = _field_info_to_spec(field_name, field_info)
            specs.append(spec)

        return tuple(specs)
