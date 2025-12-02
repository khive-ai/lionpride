# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic.main import BaseModel

from lionpride.errors import ValidationError

from ..libs.string_handlers._extract_json import extract_json
from ..libs.string_handlers._string_similarity import SIMILARITY_TYPE
from ..types import KeysLike
from ._fuzzy_match import FuzzyMatchKeysParams, fuzzy_match_keys
from ._to_dict import to_dict

if TYPE_CHECKING:
    from pydantic import BaseModel


__all__ = ("fuzzy_validate_pydantic",)


def fuzzy_validate_pydantic(
    data: BaseModel | dict[str, Any] | str,
    /,
    model_type: type[BaseModel],
    fuzzy_parse: bool = True,
    fuzzy_match: bool = False,
    fuzzy_match_params: FuzzyMatchKeysParams | dict[Any, Any] | None = None,
    return_one: bool = True,
    extract_all: bool = False,
) -> BaseModel | list[BaseModel]:
    """Validate and parse text/dict into Pydantic model with fuzzy parsing.

    Args:
        data: Input data (BaseModel instance, dict, or JSON string)
        model_type: Target Pydantic model class
        fuzzy_parse: Enable fuzzy JSON extraction from text
        fuzzy_match: Enable fuzzy key matching for field names
        fuzzy_match_params: Parameters for fuzzy matching (dict or FuzzyMatchKeysParams)
        return_one: Always return a single model (default True)
        extract_all: Extract all valid dicts as models (default False)

    Returns:
        Single model if return_one=True, list of models if extract_all=True

    Raises:
        ValidationError: If JSON extraction or model validation fails
        TypeError: If fuzzy_match_params is invalid type
        ValueError: If return_one and extract_all are both True
    """
    if return_one and extract_all:
        raise ValueError("return_one and extract_all are mutually exclusive")

    # Handle already-valid model instances
    if isinstance(data, model_type):
        return [data] if extract_all else data

    # Collect candidate dicts for validation
    candidates: list[dict[str, Any]] = []

    if isinstance(data, dict):
        candidates = [data]
    elif isinstance(data, BaseModel):
        candidates = [data.model_dump()]
    else:
        # String input - extract JSON
        try:
            extracted = extract_json(data, fuzzy_parse=fuzzy_parse, return_one_if_single=False)
            # extract_json returns list of extracted items
            # Each item can be dict or list (for JSON arrays)
            for item in extracted if isinstance(extracted, list) else [extracted]:
                if isinstance(item, dict):
                    candidates.append(item)
                elif isinstance(item, list):
                    # JSON array - add each dict in the array
                    candidates.extend(x for x in item if isinstance(x, dict))
        except Exception:
            pass

        # Fallback to to_dict if no candidates
        if not candidates:
            fallback = to_dict(
                data,
                fuzzy_parse=fuzzy_parse,
                suppress=True,
                recursive=True,
                recursive_python_only=False,
            )
            if isinstance(fallback, dict) and fallback:
                candidates = [fallback]

        if not candidates:
            raise ValidationError("Failed to extract valid JSON from input")

    # Apply fuzzy matching if enabled
    field_names = list(model_type.model_fields.keys()) if fuzzy_match else None

    def _apply_fuzzy_match(d: dict[str, Any]) -> dict[str, Any]:
        if not fuzzy_match:
            return d
        if fuzzy_match_params is None:
            return fuzzy_match_keys(d, field_names, handle_unmatched="remove")
        elif isinstance(fuzzy_match_params, dict):
            return fuzzy_match_keys(d, field_names, **fuzzy_match_params)
        elif isinstance(fuzzy_match_params, FuzzyMatchKeysParams):
            return fuzzy_match_params(d, field_names)
        else:
            raise TypeError("fuzzy_keys_params must be a dict or FuzzyMatchKeysParams instance")

    # Validate candidates
    validated: list[BaseModel] = []
    errors: list[str] = []

    for candidate in candidates:
        try:
            matched = _apply_fuzzy_match(candidate)
            model = model_type.model_validate(matched)
            validated.append(model)
            if return_one:
                return model  # Return first valid model
        except Exception as e:
            errors.append(str(e))

    if extract_all:
        if not validated:
            raise ValidationError(f"No valid models extracted. Errors: {'; '.join(errors)}")
        return validated

    # return_one=True but no valid models found
    raise ValidationError(f"Validation failed: {'; '.join(errors)}")


def fuzzy_validate_mapping(
    d: Any,
    keys: KeysLike,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
) -> dict[str, Any]:
    """Validate any input into dict with expected keys and fuzzy matching.

    Converts input (dict, JSON string, XML, object) to dict and validates keys.

    Args:
        d: Input to convert and validate
        keys: Expected keys (list or dict-like)
        similarity_algo: String similarity algorithm
        similarity_threshold: Minimum similarity score (0.0-1.0)
        fuzzy_match: Enable fuzzy key matching
        handle_unmatched: How to handle unmatched keys
        fill_value: Default value for missing keys
        fill_mapping: Custom values for specific keys
        strict: Raise if expected keys are missing
        suppress_conversion_errors: Return empty dict on conversion failure

    Returns:
        Validated dictionary with corrected keys

    Raises:
        TypeError: If d is None
        ValueError: If conversion fails and suppress_conversion_errors is False
    """
    if d is None:
        raise TypeError("Input cannot be None")

    # Try converting to dictionary
    try:
        if isinstance(d, str):
            try:
                json_result = extract_json(d, fuzzy_parse=True, return_one_if_single=True)
                dict_input = json_result[0] if isinstance(json_result, list) else json_result
            except Exception:
                dict_input = to_dict(d, fuzzy_parse=True, suppress=True)
        else:
            dict_input = to_dict(d, prioritize_model_dump=True, fuzzy_parse=True, suppress=True)

        if not isinstance(dict_input, dict):
            if suppress_conversion_errors:
                dict_input = {}
            else:
                raise ValueError(f"Failed to convert input to dictionary: {type(dict_input)}")

    except Exception as e:
        if suppress_conversion_errors:
            dict_input = {}
        else:
            raise ValueError(f"Failed to convert input to dictionary: {e}")

    # Validate the dictionary
    return fuzzy_match_keys(
        dict_input,
        keys,
        similarity_algo=similarity_algo,
        similarity_threshold=similarity_threshold,
        fuzzy_match=fuzzy_match,
        handle_unmatched=handle_unmatched,
        fill_value=fill_value,
        fill_mapping=fill_mapping,
        strict=strict,
    )
