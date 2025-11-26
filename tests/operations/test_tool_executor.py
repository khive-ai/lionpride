# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for actions.py (formerly tool_executor.py) coverage.

Tests for:
- execute_tools function
- _update_response_with_actions
- has_action_requests
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from lionpride.rules import ActionRequest, ActionResponse
from lionpride.session import Session


class TestToolExecutorCoverage:
    """Test tool_executor.py uncovered lines."""

    async def test_execute_tools_no_action_requests_attr(self):
        """Test line 22-23: Response without action_requests attribute."""
        from lionpride.operations.actions import execute_tools

        session = Session()
        branch = session.create_branch(name="test")

        # Object without action_requests attribute
        parsed_response = MagicMock(spec=[])  # spec=[] means no attributes

        result, responses = await execute_tools(parsed_response, session, branch)

        assert result is parsed_response
        assert responses == []

    async def test_execute_tools_empty_action_requests(self):
        """Test lines 25-27: Response with empty/None action_requests."""
        from lionpride.operations.actions import execute_tools

        session = Session()
        branch = session.create_branch(name="test")

        # Object with None action_requests
        parsed_response = MagicMock()
        parsed_response.action_requests = None

        result, responses = await execute_tools(parsed_response, session, branch)

        assert result is parsed_response
        assert responses == []

        # Object with empty list
        parsed_response.action_requests = []

        result, responses = await execute_tools(parsed_response, session, branch)

        assert result is parsed_response
        assert responses == []

    async def test_execute_tools_with_actions(self):
        """Test lines 29-44: Execute tools and update response."""
        from lionpride.operations.actions import execute_tools
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multiply", provider="tool"))
        model = iModel(backend=tool)

        session = Session()
        session.services.register(model)
        branch = session.create_branch(name="test")

        # Create response with action_requests
        class MockResponse(BaseModel):
            action_requests: list[ActionRequest]
            action_responses: list[ActionResponse] | None = None

        parsed_response = MockResponse(
            action_requests=[ActionRequest(function="multiply", arguments={"a": 3, "b": 4})]
        )

        result, responses = await execute_tools(parsed_response, session, branch)

        assert len(responses) == 1
        assert responses[0].output == 12
        # Result should be updated with action_responses
        assert result.action_responses is not None
        assert len(result.action_responses) == 1

    async def test_update_response_with_actions_with_model_copy(self):
        """Test lines 52-54: Update response using model_copy (Pydantic v2)."""
        from lionpride.operations.actions import _update_response_with_actions

        class TestResponse(BaseModel):
            value: str
            action_responses: list[ActionResponse] | None = None

        response = TestResponse(value="test")

        action_responses = [ActionResponse(function="tool", arguments={}, output="result")]

        result = _update_response_with_actions(response, action_responses)

        assert result.action_responses == action_responses
        assert result.value == "test"
        # Should be a new object (model_copy returns new instance)
        assert result is not response

    async def test_update_response_fallback_path(self):
        """Test lines 57-59: Fallback when model_copy is unavailable (duck-typed object)."""
        from lionpride.operations.actions import _update_response_with_actions

        # Create a simple object that looks like a pydantic model but doesn't have model_copy
        class DuckTypedResponse:
            def __init__(self):
                self.value = "test"
                self.action_responses = None

            def model_dump(self):
                return {"value": self.value, "action_responses": self.action_responses}

            @classmethod
            def model_validate(cls, data):
                obj = cls()
                obj.value = data.get("value", "test")
                obj.action_responses = data.get("action_responses")
                return obj

        response = DuckTypedResponse()
        action_responses = [ActionResponse(function="tool", arguments={}, output="result")]

        result = _update_response_with_actions(response, action_responses)

        assert result.action_responses == action_responses
        assert result.value == "test"

    def test_has_action_requests_no_attr(self):
        """Test lines 64-65: has_action_requests without attribute."""
        from lionpride.operations.actions import has_action_requests

        # Object without action_requests
        obj = MagicMock(spec=[])

        assert has_action_requests(obj) is False

    def test_has_action_requests_none(self):
        """Test lines 67-68: has_action_requests with None."""
        from lionpride.operations.actions import has_action_requests

        obj = MagicMock()
        obj.action_requests = None

        assert has_action_requests(obj) is False

    def test_has_action_requests_empty(self):
        """Test has_action_requests with empty list."""
        from lionpride.operations.actions import has_action_requests

        obj = MagicMock()
        obj.action_requests = []

        assert has_action_requests(obj) is False

    def test_has_action_requests_true(self):
        """Test has_action_requests with items."""
        from lionpride.operations.actions import has_action_requests

        obj = MagicMock()
        obj.action_requests = [ActionRequest(function="test", arguments={})]

        assert has_action_requests(obj) is True
