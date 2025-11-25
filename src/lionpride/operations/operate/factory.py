# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.rules import ActionRequest, ActionResponse, Reason
from lionpride.types import Operable, Spec

from .message_prep import prepare_tool_schemas
from .tool_executor import execute_tools, has_action_requests

if TYPE_CHECKING:
    from lionpride.session import Branch, Session


async def operate(
    session: Session,
    branch: Branch | str,
    parameters: dict[str, Any],
) -> Any:
    """Structured output with optional actions.

    Args:
        parameters: Must include 'instruction', 'imodel', and 'response_model' or 'operable'.
            Optional: actions, reason, tools, max_retries, return_message.
    """
    # 1. Validate and extract parameters
    params = _validate_parameters(parameters)

    # 2. Build Operable from response_model + action/reason specs
    _operable, validation_model = _build_operable(
        response_model=params["response_model"],
        operable=params["operable"],
        actions=params["actions"],
        reason=params["reason"],
    )

    # 3. Prepare tool schemas
    tool_schemas = params["tool_schemas"] or prepare_tool_schemas(session, params["tools"])

    # 4. Build communicate parameters
    communicate_params = {
        "instruction": params["instruction"],
        "imodel": params["imodel"],
        "context": params["context"],
        "images": params["images"],
        "image_detail": params["image_detail"],
        "max_retries": params["max_retries"],
        "return_as": "model",  # We want validated model back
        **params["model_kwargs"],
    }

    # Add tool schemas to context if present
    if tool_schemas:
        existing_context = communicate_params.get("context") or {}
        if isinstance(existing_context, dict):
            communicate_params["context"] = {**existing_context, "tool_schemas": tool_schemas}
        else:
            communicate_params["context"] = {
                "original": existing_context,
                "tool_schemas": tool_schemas,
            }

    # 5. Set response_model for JSON validation
    # Note: LNDL mode removed for this PR, will be reintroduced later
    if validation_model:
        communicate_params["response_model"] = validation_model
    else:
        raise ValueError("operate requires response_model or operable")

    # Handle skip_validation
    if params["skip_validation"]:
        communicate_params["return_as"] = "text"
        communicate_params.pop("response_model", None)
        communicate_params.pop("operable", None)

    # 6. Call communicate
    from .communicate import communicate

    result = await communicate(session, branch, communicate_params)

    # Handle validation failure
    if isinstance(result, dict) and result.get("validation_failed"):
        if not params["return_message"]:
            raise ValueError(f"Response validation failed: {result.get('error')}")
        return result, None

    # 7. Execute actions if enabled and present
    if params["actions"] and has_action_requests(result):
        # Resolve branch for tool execution
        if isinstance(branch, str):
            branch = session.conversations.get_progression(branch)

        result, _action_responses = await execute_tools(
            result,
            session,
            branch,
            concurrent=params["concurrent_tool_execution"],
        )

    # 8. Return result
    if params["return_message"]:
        # Get the last assistant message from branch
        if isinstance(branch, str):
            branch = session.conversations.get_progression(branch)
        branch_msgs = session.messages[branch]
        assistant_msg = branch_msgs[-1] if branch_msgs else None
        return result, assistant_msg

    return result


def _validate_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize parameters."""
    # Required parameters
    if not params.get("instruction"):
        raise ValueError("operate requires 'instruction' parameter")
    if not params.get("imodel"):
        raise ValueError("operate requires 'imodel' parameter")

    # Either response_model or operable/operative required
    operable = params.get("operable") or params.get("operative")
    if not params.get("response_model") and not operable:
        raise ValueError("operate requires either 'response_model' or 'operable' parameter")

    return {
        "instruction": params["instruction"],
        "imodel": params["imodel"],
        "response_model": params.get("response_model"),
        "operable": operable,
        "context": params.get("context"),
        "images": params.get("images"),
        "image_detail": params.get("image_detail"),
        "tool_schemas": params.get("tool_schemas"),
        "tools": params.get("tools", False),
        "actions": params.get("actions", False),
        "reason": params.get("reason", False),
        "max_retries": params.get("max_retries", 0),
        "skip_validation": params.get("skip_validation", False),
        "return_message": params.get("return_message", False),
        "concurrent_tool_execution": params.get("concurrent_tool_execution", True),
        "model_kwargs": _extract_model_kwargs(params),
    }


def _extract_model_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Extract model kwargs from params, handling nested model_kwargs dict."""
    known_keys = {
        "instruction",
        "imodel",
        "response_model",
        "operable",
        "operative",
        "context",
        "images",
        "image_detail",
        "tool_schemas",
        "tools",
        "actions",
        "reason",
        "max_retries",
        "skip_validation",
        "return_message",
        "concurrent_tool_execution",
        "model_kwargs",  # Handle nested case
    }

    # Start with explicitly passed model_kwargs (from Branch.operate wrapper)
    result = dict(params.get("model_kwargs", {}))

    # Add any flat params that aren't known keys
    for k, v in params.items():
        if k not in known_keys:
            result[k] = v

    return result


def _build_operable(
    response_model: type[BaseModel] | None,
    operable: Operable | None,
    actions: bool,
    reason: bool,
) -> tuple[Operable | None, type[BaseModel] | None]:
    """Build Operable from response_model + action/reason specs.

    Following lionagi v0 pattern: fields are FLAT (not nested).
    response_model fields are flattened into top-level specs.
    """
    # If operable provided, use it directly
    if operable:
        return operable, operable.create_model() if operable else None

    # Validate response_model
    if response_model and (
        not isinstance(response_model, type) or not issubclass(response_model, BaseModel)
    ):
        raise ValueError(
            f"response_model must be a Pydantic BaseModel subclass, got {response_model}"
        )

    # If no actions/reason needed, just use response_model directly
    if not actions and not reason:
        return None, response_model

    # Build Operable with FLAT field specs (lionagi pattern)
    specs = []
    existing_field_names = set()

    # Flatten response_model fields into top-level specs
    if response_model:
        for field_name, field_info in response_model.model_fields.items():
            existing_field_names.add(field_name)
            # Determine if field has a default
            has_default = field_info.default is not None or field_info.default_factory is not None
            default_val = field_info.default if has_default else None

            specs.append(
                Spec(
                    base_type=field_info.annotation,
                    name=field_name,
                    default=default_val,
                )
            )

    # Add reason spec (optional) - skip if already exists
    if reason and "reason" not in existing_field_names:
        specs.append(
            Spec(
                base_type=Reason,
                name="reason",
                default=None,
            )
        )

    # Add action specs (optional) - skip if already exist
    if actions:
        if "action_requests" not in existing_field_names:
            specs.append(
                Spec(
                    base_type=list[ActionRequest],
                    name="action_requests",
                    default=None,
                )
            )
        if "action_responses" not in existing_field_names:
            specs.append(
                Spec(
                    base_type=list[ActionResponse],
                    name="action_responses",
                    default=None,
                )
            )

    # Create Operable with all flat fields
    name = response_model.__name__ if response_model else "OperateResponse"
    operable = Operable(specs=tuple(specs), name=name)

    # Create validation model (excluding action_responses for request)
    # LLM fills request fields; action_responses filled after tool execution
    validation_model = operable.create_model(
        model_name=f"{name}Request",
        exclude={"action_responses"} if actions else None,
    )

    return operable, validation_model
