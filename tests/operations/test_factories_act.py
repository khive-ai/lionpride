# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for action execution integration in operate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from lionpride import Event, EventStatus
from lionpride.operations.models import ActionRequestModel, ActionResponseModel
from lionpride.operations.operate.factory import operate
from lionpride.services import ServiceRegistry, iModel
from lionpride.services.providers.oai_chat import OAIChatEndpoint
from lionpride.services.types.tool import Tool, ToolConfig
from lionpride.session import Session


# Test Tools
async def mock_multiply(a: int, b: int) -> int:
    """Mock multiply tool."""
    return a * b


async def mock_add_tool(a: int, b: int) -> int:
    """Mock add tool."""
    return a + b


async def mock_error_tool(message: str) -> str:
    """Mock tool that always raises error."""
    raise ValueError(f"Test error: {message}")


async def mock_slow_tool(delay: float) -> str:
    """Mock tool with delay."""
    await asyncio.sleep(delay)
    return "completed"


# Test Response Models
class AnalysisModel(BaseModel):
    """Test response model."""

    summary: str = Field(description="Analysis summary")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class ActionAnalysisModel(BaseModel):
    """Response model with action fields (created by create_action_operative)."""

    summary: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: dict[str, Any] | None = None
    action_requests: list[ActionRequestModel] | None = None
    action_responses: list[ActionResponseModel] | None = None


@dataclass
class MockResponse:
    """Mock response from model."""

    status: str = "success"
    data: str = ""
    raw_response: dict = None
    metadata: dict = None

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {"id": "mock-id", "choices": [{"message": {"content": self.data}}]}
        if self.metadata is None:
            self.metadata = {"model": "mock-model", "usage": {}}


class TestOperateFactoryActionExecution:
    """Test action execution integration in operate."""

    @pytest.fixture
    def session_with_tools(self) -> Session:
        """Create session with registered tools."""
        session = Session()

        # Register tools
        multiply_tool = Tool(
            func_callable=mock_multiply,
            config=ToolConfig(name="multiply", provider="tool"),
        )
        add_tool = Tool(
            func_callable=mock_add_tool,
            config=ToolConfig(name="add_tool", provider="tool"),
        )
        error_tool = Tool(
            func_callable=mock_error_tool,
            config=ToolConfig(name="error_tool", provider="tool"),
        )
        slow_tool = Tool(
            func_callable=mock_slow_tool,
            config=ToolConfig(name="slow_tool", provider="tool"),
        )

        session.services.register(iModel(backend=multiply_tool))
        session.services.register(iModel(backend=add_tool))
        session.services.register(iModel(backend=error_tool))
        session.services.register(iModel(backend=slow_tool))

        return session

    @pytest.fixture
    def mock_llm_model(self):
        """Create mock LLM model."""
        endpoint = OAIChatEndpoint(config=None, name="mock_llm", api_key="mock-key")
        model = iModel(backend=endpoint)
        return model

    def _setup_llm_response(self, model: iModel, response_data: str):
        """Setup mock LLM to return specific response."""

        async def mock_invoke(
            model_name: str | None = None, messages: list | None = None, **kwargs
        ):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse(status="success", data=response_data)

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

    async def test_actions_true_creates_action_operative(self, session_with_tools, mock_llm_model):
        """Test that actions=True creates an action operative with action fields."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        # LLM returns JSON for base AnalysisModel, action fields will be added by create_action_operative
        llm_response = """{
            "analysismodel": {
                "summary": "Calculating sum and product",
                "confidence": 0.95
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}},
                {"function": "add_tool", "arguments": {"a": 10, "b": 20}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},  # Base model without action fields
                "actions": True,  # Enable action execution - this will add action fields
                "concurrent_tool_execution": True,
            },
        )

        # Result should be the dynamically created response model with executed actions
        # The model is created by create_action_operative
        assert hasattr(result, "analysismodel")  # The base model as a spec (lowercase)
        assert result.analysismodel.summary == "Calculating sum and product"
        assert result.analysismodel.confidence == 0.95

        # Actions should have been executed
        assert hasattr(result, "action_responses")
        assert len(result.action_responses) == 2
        assert result.action_responses[0].output == 15  # 5 * 3
        assert result.action_responses[1].output == 30  # 10 + 20

    async def test_concurrent_tool_execution_true(self, session_with_tools, mock_llm_model):
        """Test concurrent_tool_execution=True executes tools in parallel."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        # Use base model and let create_action_operative add the action fields
        llm_response = """{
            "analysismodel": {
                "summary": "Running multiple tools",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}},
                {"function": "add_tool", "arguments": {"a": 10, "b": 20}},
                {"function": "multiply", "arguments": {"a": 7, "b": 2}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,  # Let create_action_operative add action fields
                "concurrent_tool_execution": True,
            },
        )

        # All tools should execute and return results
        assert hasattr(result, "action_responses")
        assert len(result.action_responses) == 3
        assert result.action_responses[0].output == 15
        assert result.action_responses[1].output == 30
        assert result.action_responses[2].output == 14

        # Result should have the main response data as a spec
        assert hasattr(result, "analysismodel")
        assert result.analysismodel.summary == "Running multiple tools"
        assert result.analysismodel.confidence == 0.9

    async def test_concurrent_tool_execution_false(self, session_with_tools, mock_llm_model):
        """Test concurrent_tool_execution=False executes tools sequentially."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Running tools sequentially",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}},
                {"function": "add_tool", "arguments": {"a": 10, "b": 20}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
                "concurrent_tool_execution": False,
            },
        )

        # Tools execute sequentially but results should be same
        assert len(result.action_responses) == 2
        assert result.action_responses[0].output == 15
        assert result.action_responses[1].output == 30

    async def test_tool_execution_error_handling(self, session_with_tools, mock_llm_model):
        """Test that tool execution errors are captured in action_responses."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Testing error handling",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}},
                {"function": "error_tool", "arguments": {"message": "test error"}},
                {"function": "add_tool", "arguments": {"a": 10, "b": 20}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
                "concurrent_tool_execution": True,
            },
        )

        # First tool succeeds
        assert result.action_responses[0].output == 15

        # Second tool has error (captured, not raised)
        assert "ValueError: Test error: test error" in result.action_responses[1].output

        # Third tool succeeds (execution continues despite error)
        assert result.action_responses[2].output == 30

    async def test_empty_action_requests(self, session_with_tools, mock_llm_model):
        """Test that empty action_requests array works correctly."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "No tools needed",
                "confidence": 0.9
            },
            "action_requests": []
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Analyze",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
            },
        )

        # No action execution should occur
        assert result.action_requests == []
        # action_responses remains None/missing when no actions are executed
        # (code only updates action_responses if action_requests is truthy)
        action_responses = getattr(result, "action_responses", None)
        assert action_responses is None or action_responses == []

    async def test_null_action_requests(self, session_with_tools, mock_llm_model):
        """Test that null action_requests (no field) works correctly."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "No action requests field",
                "confidence": 0.9
            }
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Analyze",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
            },
        )

        # No action execution should occur (action_requests is None)
        # The code checks: if action_requests: ...
        assert not hasattr(result, "action_requests") or result.action_requests is None

    async def test_reason_and_actions_combined(self, session_with_tools, mock_llm_model):
        """Test that reason=True and actions=True work together."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Analysis with reasoning",
                "confidence": 0.95
            },
            "reason": {"reasoning": "Using multiplication and addition", "confidence": 0.9},
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}},
                {"function": "add_tool", "arguments": {"a": 10, "b": 20}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate with reasoning",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "reason": True,
                "actions": True,
            },
        )

        # Verify reasoning field
        assert hasattr(result, "reason")
        assert result.reason is not None
        assert result.reason.reasoning == "Using multiplication and addition"
        assert result.reason.confidence == 0.9

        # Verify action execution
        assert len(result.action_responses) == 2
        assert result.action_responses[0].output == 15
        assert result.action_responses[1].output == 30

    async def test_tool_not_in_registry_raises_error(self, session_with_tools, mock_llm_model):
        """Test that referencing non-existent tool raises error."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Invalid tool",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "nonexistent_tool", "arguments": {"a": 5}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        with pytest.raises(ValueError, match=r"Tool 'nonexistent_tool' not found in registry"):
            await operate(
                session,
                branch,
                {
                    "instruction": "Calculate",
                    "imodel": mock_llm_model,
                    "response_model": AnalysisModel,
                    "model_kwargs": {
                        "model_name": "gpt-4.1-mini"
                    },  # Use base model without action fields
                    "actions": True,
                },
            )

    async def test_missing_function_name_raises_error(self, session_with_tools, mock_llm_model):
        """Test that action_request with null function raises error."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Missing function",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": null, "arguments": {"a": 5, "b": 3}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        with pytest.raises(ValueError, match=r"Action request missing function name"):
            await operate(
                session,
                branch,
                {
                    "instruction": "Calculate",
                    "imodel": mock_llm_model,
                    "response_model": AnalysisModel,
                    "model_kwargs": {
                        "model_name": "gpt-4.1-mini"
                    },  # Use base model without action fields
                    "actions": True,
                },
            )

    async def test_action_responses_merged_into_result(self, session_with_tools, mock_llm_model):
        """Test that action_responses are properly merged into parsed_response."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Merging responses",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 6, "b": 7}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
            },
        )

        # Verify action_responses field is populated (not None)
        assert hasattr(result, "action_responses")
        assert result.action_responses is not None
        assert len(result.action_responses) == 1

        # Verify the action_response contains correct data
        assert result.action_responses[0].function == "multiply"
        assert result.action_responses[0].arguments == {"a": 6, "b": 7}
        assert result.action_responses[0].output == 42

    async def test_model_copy_update_for_frozen_models(self, session_with_tools, mock_llm_model):
        """Test that frozen models use model_copy for merging action_responses."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Frozen model test",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "add_tool", "arguments": {"a": 10, "b": 20}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
            },
        )

        # Result should be a new instance with action_responses merged
        # (uses model_copy or model_validate to create new instance)
        assert hasattr(result, "action_responses")
        assert result.action_responses[0].output == 30

    async def test_return_message_with_actions(self, session_with_tools, mock_llm_model):
        """Test that return_message=True returns (parsed_response, message) tuple."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Message return test",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
                "return_message": True,
            },
        )

        # Should return tuple (parsed_response, message)
        assert isinstance(result, tuple)
        assert len(result) == 2

        parsed_response, message = result
        assert hasattr(parsed_response, "action_responses")
        assert parsed_response.action_responses[0].output == 15

        # Message should contain the full response including action_responses
        from lionpride.session.messages import Message

        assert isinstance(message, Message)

    async def test_actions_false_no_execution(self, session_with_tools, mock_llm_model):
        """Test that actions=False prevents action execution even if LLM returns action_requests."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        # When actions=False, response should match the model directly (no spec wrapper)
        llm_response = """{
            "summary": "No action execution",
            "confidence": 0.9
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Analyze",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use regular model without action fields
                "actions": False,  # Explicitly disable - no action operative created
            },
        )

        # Result should not have action fields
        assert not hasattr(result, "action_requests")
        assert not hasattr(result, "action_responses")

    async def test_multiple_action_rounds_in_sequence(self, session_with_tools, mock_llm_model):
        """Test multiple action_requests executed in correct order."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Multiple actions",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 2, "b": 3}},
                {"function": "add_tool", "arguments": {"a": 10, "b": 5}},
                {"function": "multiply", "arguments": {"a": 4, "b": 5}},
                {"function": "add_tool", "arguments": {"a": 1, "b": 1}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
                "concurrent_tool_execution": True,
            },
        )

        # All actions should execute and maintain order in results
        assert len(result.action_responses) == 4
        assert result.action_responses[0].output == 6  # 2 * 3
        assert result.action_responses[1].output == 15  # 10 + 5
        assert result.action_responses[2].output == 20  # 4 * 5
        assert result.action_responses[3].output == 2  # 1 + 1

    @pytest.mark.skip(reason="get_tool_schemas() not yet implemented in ServiceRegistry")
    async def test_tool_schemas_integration(self, session_with_tools, mock_llm_model):
        """Test that tools=True provides tool schemas to LLM."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Using available tools",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}}
            ]
        }"""

        invoke_called = False
        received_messages = []

        async def mock_invoke_with_tools(
            model_name: str | None = None, messages: list | None = None, **kwargs
        ):
            nonlocal invoke_called, received_messages
            invoke_called = True
            received_messages = messages

            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse(status="success", data=llm_response)

            return MockCalling()

        object.__setattr__(mock_llm_model, "invoke", AsyncMock(side_effect=mock_invoke_with_tools))

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
                "tools": True,  # This should inject tool_schemas
            },
        )

        # Verify invoke was called
        assert invoke_called

        # The instruction message should contain tool_schemas
        # (this is injected by operate when tools=True)
        assert result.action_responses[0].output == 15

    @pytest.mark.skip(reason="get_tool_schemas() not yet implemented in ServiceRegistry")
    async def test_specific_tools_list(self, session_with_tools, mock_llm_model):
        """Test that tools=['multiply'] only provides specific tools."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Using specific tool",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        result = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
                "tools": ["multiply"],  # Only provide multiply tool
            },
        )

        # Should execute successfully with the specified tool
        assert result.action_responses[0].output == 15

    async def test_response_str_updated_after_action_execution(
        self, session_with_tools, mock_llm_model
    ):
        """Test that response_str is updated with tool execution results."""
        session = session_with_tools
        session.services.register(mock_llm_model, update=True)
        branch = session.create_branch(name="test")

        llm_response = """{
            "analysismodel": {
                "summary": "Response update test",
                "confidence": 0.9
            },
            "action_requests": [
                {"function": "multiply", "arguments": {"a": 5, "b": 3}}
            ]
        }"""
        self._setup_llm_response(mock_llm_model, llm_response)

        parsed_response, message = await operate(
            session,
            branch,
            {
                "instruction": "Calculate",
                "imodel": mock_llm_model,
                "response_model": AnalysisModel,
                "model_kwargs": {
                    "model_name": "gpt-4.1-mini"
                },  # Use base model without action fields
                "actions": True,
                "return_message": True,
            },
        )

        # Verify action execution happened correctly
        assert hasattr(parsed_response, "action_responses")
        assert len(parsed_response.action_responses) == 1
        assert parsed_response.action_responses[0].output == 15

        # Verify message was created
        from lionpride.session.messages import Message

        assert isinstance(message, Message)
