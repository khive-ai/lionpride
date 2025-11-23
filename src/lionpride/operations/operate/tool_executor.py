from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.operations.models import ActionResponseModel

if TYPE_CHECKING:
    from lionpride.session import Branch, Session


async def execute_tools(
    parsed_response: Any,
    session: Session,
    branch: Branch,
    *,
    concurrent: bool = True,
) -> tuple[Any, list[ActionResponseModel]]:
    """Execute tool calls from action_requests.

    Args:
        parsed_response: Parsed response with potential action_requests
        session: Current session
        branch: Current branch (unused but kept for API compatibility)
        concurrent: Whether to execute tools concurrently

    Returns:
        Tuple of (updated_response, action_responses)
    """
    if not hasattr(parsed_response, "action_requests"):
        return parsed_response, []

    action_requests = getattr(parsed_response, "action_requests", None)
    if not action_requests:
        return parsed_response, []

    # Execute tools
    from ..actions import act

    action_responses = await act(
        action_requests=action_requests,
        registry=session.services,
        concurrent=concurrent,
    )

    # Update parsed response with action_responses
    updated_response = _update_response_with_actions(
        parsed_response,
        action_responses,
    )

    return updated_response, action_responses


def _update_response_with_actions(
    parsed_response: Any,
    action_responses: list[ActionResponseModel],
) -> Any:
    """Update response with action results.

    Args:
        parsed_response: Original parsed response
        action_responses: Executed action results

    Returns:
        Updated response with action_responses field
    """
    if hasattr(parsed_response, "model_copy"):
        # Pydantic v2 way
        return parsed_response.model_copy(update={"action_responses": action_responses})
    else:
        # Fallback: create new instance
        response_dict = parsed_response.model_dump()
        response_dict["action_responses"] = action_responses
        return type(parsed_response).model_validate(response_dict)


def has_action_requests(parsed_response: Any) -> bool:
    """Check if response has action requests.

    Args:
        parsed_response: Parsed response

    Returns:
        True if has non-empty action_requests
    """
    if not hasattr(parsed_response, "action_requests"):
        return False

    action_requests = getattr(parsed_response, "action_requests", None)
    return bool(action_requests)
