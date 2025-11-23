# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.libs import concurrency

from .models import ActionRequestModel, ActionResponseModel

if TYPE_CHECKING:
    from lionpride.services import ServiceRegistry

__all__ = ("act",)


async def act(
    action_requests: list[ActionRequestModel],
    registry: ServiceRegistry,
    *,
    concurrent: bool = True,
    timeout: float | None = None,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
) -> list[ActionResponseModel]:
    """Execute tool calls from action_requests via iModel.invoke().

    Args:
        action_requests: Tool calls from LLM structured output.
        registry: ServiceRegistry containing registered tools.
        concurrent: Execute in parallel (default True).
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
        return [
            _handle_execution_result(req, resp)
            for req, resp in zip(action_requests, responses, strict=True)
        ]
    else:
        # Sequential execution (preserves order, easier debugging)
        responses = []
        for request in action_requests:
            result = await _execute_single_action(
                request,
                registry,
                timeout=timeout,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
            )
            response = _handle_execution_result(request, result)
            responses.append(response)
        return responses


async def _execute_single_action(
    request: ActionRequestModel,
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
    request: ActionRequestModel,
    result: Any,
) -> ActionResponseModel:
    """Convert execution result to ActionResponseModel."""
    # Extract function and arguments from request
    function = request.function or ""
    arguments = request.arguments or {}

    # Handle exceptions as errors
    if isinstance(result, Exception):
        error_msg = f"{type(result).__name__}: {result!s}"
        return ActionResponseModel(
            function=function,
            arguments=arguments,
            output=error_msg,
        )

    # Success case
    return ActionResponseModel(
        function=function,
        arguments=arguments,
        output=result,
    )
