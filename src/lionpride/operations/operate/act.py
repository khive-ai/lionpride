# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.errors import AccessError, NotFoundError, ValidationError
from lionpride.ln import alcall
from lionpride.rules import ActionRequest, ActionResponse
from lionpride.session.messages import (
    ActionRequestContent,
    ActionResponseContent,
    Message,
)

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("act", "execute_tools", "has_action_requests")


async def act(
    action_requests: list[ActionRequest],
    session: Session,
    branch: Branch,
    *,
    max_concurrent: int = 10,
    retry_timeout: float | None = None,
    retry_attempts: int = 0,
    throttle_period: float | None = None,
) -> list[ActionResponse]:
    """Execute tool calls from action_requests.

    Args:
        action_requests: Tool calls from LLM structured output.
        session: Session containing registered tools.
        branch: Branch for resource access check and message persistence.
        max_concurrent: Max concurrent executions (default 10).
        retry_timeout: Timeout per tool call.
        retry_attempts: Retry attempts on failure.
        throttle_period: Delay between starting tasks.

    Returns:
        List of ActionResponse objects with execution results.
    """
    if not action_requests:
        return []

    # Validate upfront
    for req in action_requests:
        if not req.function:
            raise ValidationError(
                "Action request missing function name",
                details={"request": req},
            )
        if not session.services.has(req.function):
            raise NotFoundError(
                f"Tool '{req.function}' not found in registry",
                details={"available": session.services.list_names()},
            )
        # Check branch has access to tool
        if req.function not in branch.resources:
            raise AccessError(
                f"Branch '{branch.name}' doesn't have access to tool '{req.function}'",
                details={"requested": req.function, "available": list(branch.resources)},
            )

    async def execute_single(req: ActionRequest) -> ActionResponse:
        try:
            result = await session.request(req.function, **(req.arguments or {}))
            # Extract output from result - handle Event/Calling types
            # Event subclasses (ToolCalling, etc.) have execution.response/error
            if hasattr(result, "execution"):
                if result.execution.error:
                    return ActionResponse(
                        function=req.function,
                        arguments=req.arguments,
                        output=f"{type(result.execution.error).__name__}: {result.execution.error}",
                    )
                response = result.execution.response
                output = response.data if hasattr(response, "data") else response
            elif hasattr(result, "response") and hasattr(result.response, "data"):
                output = result.response.data
            elif hasattr(result, "data"):
                output = result.data
            else:
                output = result
            return ActionResponse(
                function=req.function,
                arguments=req.arguments,
                output=output,
            )
        except Exception as e:
            return ActionResponse(
                function=req.function,
                arguments=req.arguments,
                output=f"{type(e).__name__}: {e}",
            )

    responses = await alcall(
        action_requests,
        execute_single,
        max_concurrent=max_concurrent,
        retry_timeout=retry_timeout,
        retry_attempts=retry_attempts,
        throttle_period=throttle_period,
        return_exceptions=True,
    )

    # Convert any uncaught exceptions to ActionResponse
    action_responses = [
        r
        if isinstance(r, ActionResponse)
        else ActionResponse(
            function=action_requests[i].function,
            arguments=action_requests[i].arguments,
            output=f"{type(r).__name__}: {r}" if isinstance(r, Exception) else str(r),
        )
        for i, r in enumerate(responses)
    ]

    # Persist action messages
    _persist_action_messages(session, branch, action_requests, action_responses)

    return action_responses


async def execute_tools(
    parsed_response: Any,
    session: Session,
    branch: Branch,
    *,
    max_concurrent: int = 10,
) -> tuple[Any, list[ActionResponse]]:
    """Execute tool calls from parsed response and update with results.

    Args:
        parsed_response: Pydantic model with action_requests field.
        session: Session for services and message persistence.
        branch: Branch for message persistence.
        max_concurrent: Max concurrent executions.

    Returns:
        (updated_response, action_responses) tuple.
    """
    action_requests = getattr(parsed_response, "action_requests", None)
    if not action_requests:
        return parsed_response, []

    action_responses = await act(
        action_requests=action_requests,
        session=session,
        branch=branch,
        max_concurrent=max_concurrent,
    )

    # Update response with action_responses
    if hasattr(parsed_response, "model_copy"):
        updated = parsed_response.model_copy(update={"action_responses": action_responses})
    else:
        data = parsed_response.model_dump()
        data["action_responses"] = action_responses
        updated = type(parsed_response).model_validate(data)

    return updated, action_responses


def has_action_requests(parsed_response: Any) -> bool:
    """Check if response has action requests."""
    return bool(getattr(parsed_response, "action_requests", None))


def _persist_action_messages(
    session: Session,
    branch: Branch,
    action_requests: list[ActionRequest],
    action_responses: list[ActionResponse],
) -> None:
    """Persist action request/response messages to session/branch."""
    for req, resp in zip(action_requests, action_responses, strict=True):
        # Request message: branch → tool
        req_msg = Message(
            content=ActionRequestContent.create(
                function=req.function,
                arguments=req.arguments,
            ),
            sender=branch.id,
            recipient=req.function,
        )
        session.add_message(req_msg, branches=branch)

        # Response message: tool → branch
        resp_msg = Message(
            content=ActionResponseContent.create(
                request_id=str(req_msg.id),
                result=resp.output if not isinstance(resp.output, Exception) else None,
                error=str(resp.output) if isinstance(resp.output, Exception) else None,
            ),
            sender=req.function,
            recipient=branch.id,
        )
        session.add_message(resp_msg, branches=branch)
