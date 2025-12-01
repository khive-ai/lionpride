# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, get_args, get_origin

from lionpride.protocols import Allowable, Hashable, implements

from ._sentinel import MaybeUnset, Unset

if TYPE_CHECKING:
    from pydantic import BaseModel

    from .spec import Spec
    from .spec_adapters._protocol import SpecAdapter

__all__ = ("Operable", "get_adapter")

# Supported adapter types
AdapterType = Literal["pydantic"]  # "rust" coming soon


@functools.cache
def get_adapter(adapter_name: str) -> type[SpecAdapter]:
    """Get adapter class by name (cached).

    Factory method for adapter classes. Caches to avoid repeated imports.
    When Rust adapter is ready, add case here.

    Args:
        adapter_name: Adapter identifier ("pydantic", future: "rust")

    Returns:
        Adapter class (not instance)

    Raises:
        ValueError: If adapter not supported
        ImportError: If adapter dependencies not installed
    """
    match adapter_name:
        case "pydantic":
            try:
                from .spec_adapters.pydantic_field import PydanticSpecAdapter

                return PydanticSpecAdapter
            except ImportError as e:
                raise ImportError(
                    "PydanticSpecAdapter requires Pydantic. Install with: pip install pydantic"
                ) from e
        # case "rust":
        #     from .spec_adapters.rust_field import RustSpecAdapter
        #     return RustSpecAdapter
        case _:
            raise ValueError(f"Unsupported adapter: {adapter_name}")


@implements(Hashable, Allowable)
@dataclass(frozen=True, slots=True, init=False)
class Operable:
    """Ordered Spec collection for model generation.

    Validates uniqueness, no duplicates. Provides adapter interface
    for framework-agnostic model creation and validation.

    Usage:
        op = Operable(specs, adapter="pydantic")
        Model = op.adapter.create_model(op)
        instance = op.adapter.validate_model(Model, data)
    """

    __op_fields__: tuple[Spec, ...]
    __adapter_name__: str
    name: str | None

    def __init__(
        self,
        specs: tuple[Spec, ...] | list[Spec] = (),
        *,
        name: str | None = None,
        adapter: AdapterType = "pydantic",
    ):
        """Init with specs and adapter.

        Args:
            specs: Tuple or list of Spec objects
            name: Optional operable name (used as model name default)
            adapter: Adapter type for model generation ("pydantic", future: "rust")

        Raises:
            TypeError: If non-Spec in specs
            ValueError: If duplicate field names
        """
        from .spec import Spec

        # Convert to tuple if list
        if isinstance(specs, list):
            specs = tuple(specs)

        # Validate all items are Spec objects
        for i, item in enumerate(specs):
            if not isinstance(item, Spec):
                raise TypeError(
                    f"All specs must be Spec objects, got {type(item).__name__} at index {i}"
                )

        # Check for duplicate names
        names = [s.name for s in specs if s.name is not None]
        if len(names) != len(set(names)):
            from collections import Counter

            duplicates = [name for name, count in Counter(names).items() if count > 1]
            raise ValueError(
                f"Duplicate field names found: {duplicates}. Each spec must have a unique name."
            )

        object.__setattr__(self, "__op_fields__", specs)
        object.__setattr__(self, "__adapter_name__", adapter)
        object.__setattr__(self, "name", name)

    @property
    def adapter(self) -> type[SpecAdapter]:
        """Get adapter class (cached).

        Returns the adapter class for this operable. All adapter operations
        should go through this interface: op.adapter.create_model(), etc.

        Returns:
            Adapter class (PydanticSpecAdapter, future: RustSpecAdapter)
        """
        return get_adapter(self.__adapter_name__)

    def allowed(self) -> set[str]:
        """Get set of allowed field names from specs."""
        return {i.name for i in self.__op_fields__}  # type: ignore[misc]

    def check_allowed(self, *args, as_boolean: bool = False):
        """Check field names allowed. Args: field names, as_boolean. Raises ValueError if not allowed."""
        if not set(args).issubset(self.allowed()):
            if as_boolean:
                return False
            raise ValueError(
                f"Some specified fields are not allowed: {set(args).difference(self.allowed())}"
            )
        return True

    def get(self, key: str, /, default=Unset) -> MaybeUnset[Spec]:
        """Get Spec by field name. Returns default if not found."""
        if not self.check_allowed(key, as_boolean=True):
            return default
        for i in self.__op_fields__:
            if i.name == key:
                return i

    def get_specs(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[Spec, ...]:
        """Get filtered Specs. Args: include/exclude sets. Raises ValueError if both or invalid names."""
        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude")

        if include:
            if self.check_allowed(*include, as_boolean=True) is False:
                raise ValueError(
                    "Some specified fields are not allowed: "
                    f"{set(include).difference(self.allowed())}"
                )
            return tuple(self.get(i) for i in include if self.get(i) is not Unset)  # type: ignore[misc]

        if exclude:
            _discards = {self.get(i) for i in exclude if self.get(i) is not Unset}
            return tuple(s for s in self.__op_fields__ if s not in _discards)

        return self.__op_fields__

    def create_model(
        self,
        *,
        model_name: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        **kw,
    ):
        """Create framework model from specs via adapter.

        Args:
            model_name: Override model name (default: self.name or "DynamicModel")
            include: Only include these field names
            exclude: Exclude these field names
            **kw: Additional adapter-specific kwargs

        Returns:
            Framework model class (e.g., Pydantic BaseModel subclass)
        """
        # Filter out 'adapter' kwarg - it's for selecting adapter, not passing to it
        kw.pop("adapter", None)
        return self.adapter.create_model(
            self,
            model_name=model_name or self.name or "DynamicModel",
            include=include,
            exclude=exclude,
            **kw,
        )

    @classmethod
    def from_model(
        cls,
        model: type[BaseModel],
        *,
        name: str | None = None,
        adapter: AdapterType = "pydantic",
    ) -> Self:
        """Create Operable from a Pydantic model's fields.

        Disassembles a model and returns an Operable with Specs
        representing top-level fields of that model.

        Args:
            model: Pydantic BaseModel class to disassemble
            name: Optional operable name (defaults to model class name)
            adapter: Adapter type for model generation

        Returns:
            Operable with Specs for each top-level field

        Example:
            >>> class MyModel(BaseModel):
            ...     name: str
            ...     age: int = 0
            ...     tags: list[str] | None = None
            >>> op = Operable.from_model(MyModel)
            >>> op.allowed()  # {'name', 'age', 'tags'}
        """
        from pydantic import BaseModel

        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(f"model must be a Pydantic BaseModel subclass, got {type(model)}")

        specs: list[Spec] = []

        for field_name, field_info in model.model_fields.items():
            spec = _field_info_to_spec(field_name, field_info)
            specs.append(spec)

        return cls(
            specs=tuple(specs),
            name=name or model.__name__,
            adapter=adapter,
        )


def _is_nullable_type(annotation: Any) -> tuple[bool, Any]:
    """Check if annotation is Optional/nullable and extract inner type.

    Returns:
        Tuple of (is_nullable, inner_type_or_original)
    """
    import types

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
                from functools import reduce

                return True, reduce(lambda a, b: a | b, non_none_args)
        return False, annotation

    # typing.Union
    try:
        from typing import Union

        if origin is Union:
            args = get_args(annotation)
            non_none_args = [a for a in args if a is not type(None)]
            if len(args) != len(non_none_args):
                if len(non_none_args) == 1:
                    return True, non_none_args[0]
                if non_none_args:
                    from functools import reduce

                    return True, reduce(lambda a, b: a | b, non_none_args)
            return False, annotation
    except ImportError:
        pass

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

    from .spec import Spec

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
