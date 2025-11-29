# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool_executor.py coverage.

Tests for:
- execute_tools function (lines 22-44)
- _update_response_with_actions (lines 52-59)
- has_action_requests (lines 64-68)
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
        from lionpride.operations.operate.act import execute_tools

        session = Session()
        branch = session.create_branch(name="test")

        # Object without action_requests attribute
        parsed_response = MagicMock(spec=[])  # spec=[] means no attributes

        result, responses = await execute_tools(parsed_response, session, branch)

        assert result is parsed_response
        assert responses == []

    async def test_execute_tools_empty_action_requests(self):
        """Test lines 25-27: Response with empty/None action_requests."""
        from lionpride.operations.operate.act import execute_tools

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
        """Test execute_tools updates response with tool results."""
        from lionpride.operations.operate.act import execute_tools
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        async def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(func_callable=multiply, config=ToolConfig(name="multiply", provider="tool"))
        model = iModel(backend=tool)

        session = Session()
        session.services.register(model)
        # Branch must have access to the tool in its resources
        branch = session.create_branch(name="test", resources={"multiply"})

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

    async def test_update_response_with_model_copy(self):
        """Test execute_tools updates response using model_copy (Pydantic v2)."""
        from lionpride.operations.operate.act import execute_tools
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        async def add_nums(a: int, b: int) -> int:
            return a + b

        tool = Tool(func_callable=add_nums, config=ToolConfig(name="add_nums", provider="tool"))
        model = iModel(backend=tool)

        session = Session()
        session.services.register(model)
        branch = session.create_branch(name="test", resources={"add_nums"})

        class TestResponse(BaseModel):
            value: str
            action_requests: list[ActionRequest]
            action_responses: list[ActionResponse] | None = None

        response = TestResponse(
            value="test",
            action_requests=[ActionRequest(function="add_nums", arguments={"a": 1, "b": 2})],
        )

        result, _responses = await execute_tools(response, session, branch)

        assert result.action_responses is not None
        assert result.value == "test"
        # Should be a new object (model_copy returns new instance)
        assert result is not response

    async def test_update_response_fallback_path(self):
        """Test execute_tools fallback when model_copy unavailable."""
        from lionpride.operations.operate.act import execute_tools
        from lionpride.services import iModel
        from lionpride.services.types.tool import Tool, ToolConfig

        async def subtract(a: int, b: int) -> int:
            return a - b

        tool = Tool(func_callable=subtract, config=ToolConfig(name="subtract", provider="tool"))
        model = iModel(backend=tool)

        session = Session()
        session.services.register(model)
        branch = session.create_branch(name="test", resources={"subtract"})

        # Use a class that implements model_dump and model_validate but NOT model_copy
        # to test the fallback path
        class DuckTypedResponse:
            def __init__(self, value="test", action_requests=None, action_responses=None):
                self.value = value
                self.action_requests = action_requests or []
                self.action_responses = action_responses

            def model_dump(self):
                return {
                    "value": self.value,
                    "action_requests": self.action_requests,
                    "action_responses": self.action_responses,
                }

            @classmethod
            def model_validate(cls, data):
                return cls(
                    value=data.get("value", "test"),
                    action_requests=data.get("action_requests", []),
                    action_responses=data.get("action_responses"),
                )

        response = DuckTypedResponse(
            action_requests=[ActionRequest(function="subtract", arguments={"a": 5, "b": 3})]
        )

        result, _responses = await execute_tools(response, session, branch)

        assert result.action_responses is not None
        assert result.value == "test"

    def test_has_action_requests_no_attr(self):
        """Test lines 64-65: has_action_requests without attribute."""
        from lionpride.operations.operate.act import has_action_requests

        # Object without action_requests
        obj = MagicMock(spec=[])

        assert has_action_requests(obj) is False

    def test_has_action_requests_none(self):
        """Test lines 67-68: has_action_requests with None."""
        from lionpride.operations.operate.act import has_action_requests

        obj = MagicMock()
        obj.action_requests = None

        assert has_action_requests(obj) is False

    def test_has_action_requests_empty(self):
        """Test has_action_requests with empty list."""
        from lionpride.operations.operate.act import has_action_requests

        obj = MagicMock()
        obj.action_requests = []

        assert has_action_requests(obj) is False

    def test_has_action_requests_true(self):
        """Test has_action_requests with items."""
        from lionpride.operations.operate.act import has_action_requests

        obj = MagicMock()
        obj.action_requests = [ActionRequest(function="test", arguments={})]

        assert has_action_requests(obj) is True
