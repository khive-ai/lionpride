# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.ln import json_dumps
from lionpride.lndl import parse_lndl_fuzzy
from lionpride.types.spec_adapters.pydantic_field import PydanticSpecAdapter

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

    from .operative import Operative


# =============================================================================
# Common Helpers
# =============================================================================


def _update_model_fields(model: Any, updates: dict[str, Any]) -> Any:
    """Update model fields with given updates (Pydantic v2 compatible)."""
    if hasattr(model, "model_copy"):
        return model.model_copy(update=updates)
    else:
        model_dict = model.model_dump()
        model_dict.update(updates)
        return type(model).model_validate(model_dict)


def to_response_str(data: Any) -> str:
    """Convert response data to string for Message storage."""
    if isinstance(data, str):
        return data
    if isinstance(data, BaseModel):
        return data.model_dump_json()
    if isinstance(data, dict):
        result = json_dumps(data)
        return result if isinstance(result, str) else result.decode("utf-8")
    return str(data)


async def parse_response(
    response_text: str,
    response_data: Any,
    *,
    use_lndl: bool = False,
    operative: Operative | None = None,
    response_model: type[BaseModel] | None = None,
    lndl_threshold: float = 0.85,
    skip_validation: bool = False,
    branch: Branch | None = None,
    session: Session | None = None,
) -> tuple[Any, str]:
    """Parse and validate response with appropriate strategy."""
    if skip_validation:
        return response_data, response_text

    if use_lndl and operative:
        return await _parse_lndl_response(
            response_text,
            operative,
            lndl_threshold,
            branch,
            session,
        )
    else:
        return _parse_json_response(
            response_text,
            response_data,
            operative,
            response_model,
        )


async def _parse_lndl_response(
    response_text: str,
    operative: Operative,
    threshold: float,
    branch: Branch | None,
    session: Session | None,
) -> tuple[Any, str]:
    """Parse LNDL response with fuzzy matching."""
    try:
        # Parse LNDL with fuzzy matching
        lndl_output = parse_lndl_fuzzy(
            response_text,
            operative.operable,
            threshold=threshold,
        )

        # Extract the model from fields
        if lndl_output and lndl_output.fields:
            # Get the first (usually only) field
            spec_name = next(iter(lndl_output.fields.keys()))
            parsed_response = lndl_output.fields[spec_name]

            # Process ActionCall objects within model fields
            if branch is not None and session is not None:
                parsed_response = await _process_embedded_actions(
                    parsed_response,
                    branch,
                    session,
                )

            # Handle action calls if present in actions dict
            if lndl_output.actions and branch and session:
                parsed_response = await _process_lndl_actions(
                    parsed_response,
                    lndl_output.actions,
                    branch,
                    session,
                )

            return parsed_response, to_response_str(parsed_response)

    except Exception as e:
        # Log and fallback
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"LNDL parsing failed: {e}")

        # Fallback to operative validation
        if operative:
            parsed = operative.validate_response(response_text, strict=False)
            if parsed:
                return parsed, to_response_str(parsed)

    # Validation failed
    raw_dict = operative.response_str_dict or response_text
    return (
        {"raw": raw_dict, "validation_failed": True},
        to_response_str(raw_dict),
    )


def _parse_json_response(
    response_text: str,
    response_data: Any,
    operative: Operative | None,
    response_model: type[BaseModel] | None,
) -> tuple[Any, str]:
    """Parse JSON response with validation."""
    if operative:
        # Use operative's two-tier validation
        parsed = operative.validate_response(response_text, strict=False)
        if parsed:
            return parsed, to_response_str(parsed)

        # Operative validation failed
        raw_dict = operative.response_str_dict or response_text
        return (
            {"raw": raw_dict, "validation_failed": True},
            to_response_str(raw_dict),
        )

    elif response_model:
        # Direct Pydantic validation
        parsed = _validate_with_model(response_text, response_data, response_model)
        return parsed, to_response_str(parsed)

    # No validation
    return response_data, response_text


def _validate_with_model(
    response_text: str,
    response_data: Any,
    response_model: type[BaseModel],
) -> Any:
    """Validate response with Pydantic model."""
    # Try PydanticSpecAdapter first (includes fuzzy matching)
    parsed = PydanticSpecAdapter.validate_response(
        response_text,
        response_model,
        strict=False,
        fuzzy_parse=True,
    )

    if parsed is not None:
        return parsed

    # Fallback to direct validation
    if isinstance(response_data, dict):
        return response_model.model_validate(response_data)

    # Try parsing as JSON
    import json

    try:
        data = json.loads(response_text) if isinstance(response_text, str) else response_data
        return response_model.model_validate(data)
    except Exception:
        # Wrap in content field as last resort
        return response_model.model_validate({"content": response_text})


async def _process_embedded_actions(
    parsed_response: Any,
    branch: Branch,
    session: Session,
) -> Any:
    """Process ActionCall objects embedded within model fields."""
    from pydantic import BaseModel

    from lionpride.lndl.types import ActionCall

    from ..actions import act
    from ..models import ActionRequestModel

    if not isinstance(parsed_response, BaseModel):
        return parsed_response

    # Collect ActionCall objects from model fields
    action_fields = {}
    action_requests = []

    # Use model_fields from the class, not instance
    for field_name in type(parsed_response).model_fields:
        field_value = getattr(parsed_response, field_name, None)
        if isinstance(field_value, ActionCall):
            action_fields[field_name] = len(action_requests)
            action_requests.append(
                ActionRequestModel(
                    function=field_value.function,
                    arguments=field_value.arguments,
                )
            )

    # Execute actions if found
    if action_requests:
        action_responses = await act(
            action_requests=action_requests,
            registry=session.services,
            concurrent=True,
        )

        # Replace ActionCall objects with results
        updates = {}
        for field_name, index in action_fields.items():
            if index < len(action_responses):
                updates[field_name] = action_responses[index].output

        if updates:
            if hasattr(parsed_response, "model_copy"):
                parsed_response = parsed_response.model_copy(update=updates)
            else:
                response_dict = parsed_response.model_dump()
                response_dict.update(updates)
                parsed_response = type(parsed_response).model_validate(response_dict)

    return parsed_response


async def _process_lndl_actions(
    parsed_response: Any,
    actions: dict[str, Any],
    branch: Branch,
    session: Session,
) -> Any:
    """Process LNDL action calls."""
    # TODO: Implement LNDL action processing
    # This would convert LNDL actions to ActionRequestModel instances
    # and execute them via act()
    from lionpride.lndl.types import ActionCall

    from ..models import ActionRequestModel

    if not actions:
        return parsed_response

    # Convert LNDL actions to ActionRequestModel instances
    action_requests = []
    for _action_name, action_call in actions.items():
        if isinstance(action_call, ActionCall):
            action_requests.append(
                ActionRequestModel(
                    function=action_call.name,
                    arguments=action_call.kwargs,
                )
            )

    if action_requests:
        # Execute actions
        from ..actions import act

        action_responses = await act(
            action_requests=action_requests,
            registry=session.services,
            concurrent=True,
        )

        # Update response with action results if it has the field
        if hasattr(parsed_response, "action_responses"):
            if hasattr(parsed_response, "model_copy"):
                parsed_response = parsed_response.model_copy(
                    update={"action_responses": action_responses}
                )
            else:
                response_dict = parsed_response.model_dump()
                response_dict["action_responses"] = action_responses
                parsed_response = type(parsed_response).model_validate(response_dict)

    return parsed_response
