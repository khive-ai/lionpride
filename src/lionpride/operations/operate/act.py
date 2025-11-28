# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Action execution for tool calls.

Unified interface for executing tool calls from LLM structured output.
Supports both stateless execution and stateful execution with message persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.libs import concurrency
from lionpride.rules import ActionRequest, ActionResponse
from lionpride.session.messages import (
    ActionRequestContent,
    ActionResponseContent,
    Message,
)

if TYPE_CHECKING:
    from lionpride.services import ServiceRegistry
    from lionpride.session import Branch, Session

__all__ = ("act", "execute_tools", "has_action_requests")


async def act(
    action_requests: list[ActionRequest],
    registry: ServiceRegistry,
    *,
    session: Session | None = None,
    branch: Branch | None = None,
    concurrent: bool = True,
    timeout: float | None = None,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
) -> list[ActionResponse]:
    """Execute tool calls from action_requests via iModel.invoke().

    Args:
        action_requests: Tool calls from LLM structured output.
        registry: ServiceRegistry containing registered tools.
        session: Optional Session for message persistence (stateful mode).
        branch: Optional Branch for message persistence (stateful mode).
        concurrent: Execute in parallel (default True).
        timeout: Timeout for each tool call.
        poll_timeout: Poll timeout for async tools.
        poll_interval: Poll interval for async tools.

    Returns:
        List of ActionResponse objects with execution results.

    Note:
        If session and branch are provided, action messages are persisted
        to the session/branch for conversation history.
    """
    if not action_requests:
        return []

    # Validate all tools exist first
    for request in action_requests:
        if not request.function:
            raise ValueError(f"Action request missing function name: {request}")

        if not registry.has(request.function):
            raise ValueError(
                f"Tool '{request.function}' not found in registry. "
                f"Available tools: {registry.list_names()}"
            )

    # Execute tools
    if concurrent:
        # Parallel execution (default, faster)
        tasks = [
            _execute_single_action(
                request,
                registry,
                timeout=timeout,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
            )
            for request in action_requests
        ]
        responses = await concurrency.gather(*tasks, return_exceptions=True)

        # Convert any exceptions to error responses
        action_responses = [
            _handle_execution_result(req, resp)
            for req, resp in zip(action_requests, responses, strict=True)
        ]
    else:
        # Sequential execution (preserves order, easier debugging)
        action_responses = []
        for request in action_requests:
            result = await _execute_single_action(
                request,
                registry,
                timeout=timeout,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
            )
            response = _handle_execution_result(request, result)
            action_responses.append(response)

    # Persist action messages if session/branch provided (stateful mode)
    if session is not None and branch is not None:
        _persist_action_messages(
            session=session,
            branch=branch,
            action_requests=action_requests,
            action_responses=action_responses,
        )

    return action_responses


async def execute_tools(
    parsed_response: Any,
    session: Session,
    branch: Branch,
    *,
    concurrent: bool = True,
) -> tuple[Any, list[ActionResponse]]:
    """Execute tool calls from parsed response and update with results.

    This is a convenience wrapper that:
    1. Extracts action_requests from parsed response
    2. Executes tools via act() with message persistence
    3. Updates parsed response with action_responses

    Args:
        parsed_response: Pydantic model with action_requests field
        session: Session for services and message persistence
        branch: Branch for message persistence
        concurrent: Execute in parallel (default True)

    Returns:
        (updated_response, action_responses) tuple
    """
    if not hasattr(parsed_response, "action_requests"):
        return parsed_response, []

    action_requests = getattr(parsed_response, "action_requests", None)
    if not action_requests:
        return parsed_response, []

    # Execute tools with message persistence
    action_responses = await act(
        action_requests=action_requests,
        registry=session.services,
        session=session,
        branch=branch,
        concurrent=concurrent,
    )

    # Update parsed response with action_responses
    updated_response = _update_response_with_actions(
        parsed_response,
        action_responses,
    )

    return updated_response, action_responses


def has_action_requests(parsed_response: Any) -> bool:
    """Check if response has action requests."""
    if not hasattr(parsed_response, "action_requests"):
        return False

    action_requests = getattr(parsed_response, "action_requests", None)
    return bool(action_requests)


async def _execute_single_action(
    request: ActionRequest,
    registry: ServiceRegistry,
    timeout: float | None = None,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
) -> Any:
    """Execute single tool call via iModel.invoke()."""
    try:
        # Get iModel from registry
        imodel = registry.get(request.function)  # type: ignore[arg-type]

        # Prepare arguments (default to empty dict if None)
        arguments = request.arguments or {}

        # Create calling with timeout configured
        calling = await imodel.create_calling(timeout=timeout, **arguments)

        # Execute via iModel.invoke() with pre-created calling
        calling = await imodel.invoke(
            calling=calling,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )

        # Extract result from Calling.execution.response
        if calling.execution.status.value != "completed":
            # Return the original error to preserve exception type
            if calling.execution.error is not None:
                return calling.execution.error
            return Exception(f"Execution {calling.execution.status.value}")

        # Return data from NormalizedResponse
        return calling.execution.response.data

    except Exception as e:
        # Return exception for error handling in _handle_execution_result
        return e


def _handle_execution_result(
    request: ActionRequest,
    result: Any,
) -> ActionResponse:
    """Convert execution result to ActionResponse."""
    # Handle exceptions as errors
    if isinstance(result, Exception):
        error_msg = f"{type(result).__name__}: {result!s}"
        return ActionResponse(
            function=request.function,
            arguments=request.arguments,
            output=error_msg,
        )

    # Success case
    return ActionResponse(
        function=request.function,
        arguments=request.arguments,
        output=result,
    )


def _persist_action_messages(
    session: Session,
    branch: Branch,
    action_requests: list[ActionRequest],
    action_responses: list[ActionResponse],
) -> None:
    """Persist action request/response messages to session/branch.

    Pattern:
    - ActionRequest: sender=branch.id (LLM via branch), recipient=tool_name
    - ActionResponse: sender=tool_name, recipient=branch.id
    """
    # Build response lookup by function name for pairing
    response_map: dict[str, ActionResponse] = {}
    for resp in action_responses:
        key = f"{resp.function}:{hash(str(resp.arguments))}"
        response_map[key] = resp

    for req in action_requests:
        # Create ActionRequest message
        # Pattern: branch (LLM) requests action from tool
        req_msg = Message(
            content=ActionRequestContent.create(
                function=req.function,
                arguments=req.arguments,
            ),
            sender=branch.id,
            recipient=req.function,  # Tool function name
            metadata={"action_type": "request"},
        )
        session.add_message(req_msg, branches=branch)

        # Find matching response
        key = f"{req.function}:{hash(str(req.arguments))}"
        if key in response_map:
            resp = response_map[key]
            # Create ActionResponse message
            # Pattern: tool responds to branch
            resp_msg = Message(
                content=ActionResponseContent.create(
                    request_id=str(req_msg.id),
                    result=resp.output if not isinstance(resp.output, Exception) else None,
                    error=str(resp.output) if isinstance(resp.output, Exception) else None,
                ),
                sender=req.function,  # Tool function name
                recipient=branch.id,
                metadata={"action_type": "response", "function": req.function},
            )
            session.add_message(resp_msg, branches=branch)


def _update_response_with_actions(
    parsed_response: Any,
    action_responses: list[ActionResponse],
) -> Any:
    """Update response with action results."""
    if hasattr(parsed_response, "model_copy"):
        # Pydantic v2 way
        return parsed_response.model_copy(update={"action_responses": action_responses})
    else:
        # Fallback: create new instance
        response_dict = parsed_response.model_dump()
        response_dict["action_responses"] = action_responses
        return type(parsed_response).model_validate(response_dict)
