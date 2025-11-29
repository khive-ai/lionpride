# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for operations system.

Tests cover:
- DependencyAwareExecutor (dependency logic, context inheritance, aggregation)
- OperationDispatcher (registration, retrieval)
- Builder (graph construction, cycle detection)
- Factories (generate, operate with validation)
"""

import asyncio

import pytest
from pydantic import BaseModel, Field

from lionpride.errors import ExecutionError
from lionpride.operations import Builder, OperationRegistry, flow
from lionpride.operations.flow import DependencyAwareExecutor
from lionpride.operations.node import create_operation
from lionpride.operations.operate.factory import operate
from lionpride.operations.operate.generate import generate
from lionpride.operations.operate.types import (
    CommunicateParams,
    GenerateParams,
    OperateParams,
    ParseParams,
)
from lionpride.session import Session
from lionpride.session.messages import InstructionContent, Message


@pytest.fixture
def mock_model():
    """Create a mock iModel for testing without API calls."""
    from dataclasses import dataclass
    from unittest.mock import AsyncMock

    from lionpride import Event, EventStatus
    from lionpride.services.providers.oai_chat import OAIChatEndpoint
    from lionpride.services.types.imodel import iModel

    @dataclass
    class MockResponse:
        status: str = "success"
        data: str = ""
        raw_response: dict = None
        metadata: dict = None

        def __post_init__(self):
            if self.raw_response is None:
                self.raw_response = {"id": "mock-id", "choices": []}
            if self.metadata is None:
                self.metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

    # Create mock endpoint
    endpoint = OAIChatEndpoint(config=None, name="mock_model", api_key="mock-key")

    # Create iModel
    model = iModel(backend=endpoint)

    # Mock the invoke method to return successful response
    async def mock_invoke(model_name: str | None = None, messages: list | None = None, **kwargs):
        class MockCalling(Event):
            def __init__(self, response_data: str):
                super().__init__()
                self.status = EventStatus.COMPLETED
                # Directly set execution response
                self.execution.response = MockResponse(status="success", data=response_data)

        # Extract response from kwargs or use default
        response = kwargs.get("_test_response", "mock response")
        return MockCalling(response)

    # Use object.__setattr__ to bypass Pydantic validation
    object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke))

    return model


@pytest.fixture
def session_with_model(mock_model):
    """Create session with registered mock model."""
    session = Session()
    session.services.register(mock_model, update=True)
    return session, mock_model


class ExampleOutput(BaseModel):
    """Example output for structured operations."""

    analysis: str = Field(..., description="Analysis result")
    confidence: float = Field(..., description="Confidence score")


# -------------------------------------------------------------------------
# OperationRegistry Tests
# -------------------------------------------------------------------------


class TestOperationRegistry:
    """Test per-session operation registry."""

    def test_per_session_isolation(self):
        """Test registries are independent per-session."""
        registry1 = OperationRegistry(auto_register_defaults=False)
        registry2 = OperationRegistry(auto_register_defaults=False)

        async def test_factory(session, branch, parameters):
            return "test result"

        registry1.register("test_op", test_factory)

        # registry1 has it, registry2 doesn't
        assert registry1.has("test_op")
        assert not registry2.has("test_op")

    def test_factory_registration(self):
        """Test registering and retrieving factories."""
        registry = OperationRegistry(auto_register_defaults=False)

        async def test_factory(session, branch, parameters):
            return "test result"

        registry.register("test_op", test_factory)
        assert "test_op" in registry.list_names()

        retrieved = registry.get("test_op")
        assert retrieved is test_factory

    def test_factory_not_found_raises(self):
        """Test retrieving non-existent factory raises KeyError."""
        registry = OperationRegistry(auto_register_defaults=False)
        with pytest.raises(KeyError, match=r"not registered"):
            registry.get("nonexistent")

    def test_list_names(self):
        """Test listing all registered operation names."""
        registry = OperationRegistry(auto_register_defaults=False)

        async def factory1(session, branch, parameters):
            pass

        async def factory2(session, branch, parameters):
            pass

        registry.register("op1", factory1)
        registry.register("op2", factory2)

        names = registry.list_names()
        assert "op1" in names
        assert "op2" in names

    def test_auto_register_defaults(self):
        """Test default operations are auto-registered."""
        registry = OperationRegistry(auto_register_defaults=True)

        # Default operations should be registered
        assert registry.has("operate")
        assert registry.has("react")
        assert registry.has("communicate")
        assert registry.has("generate")


# -------------------------------------------------------------------------
# Builder Tests
# -------------------------------------------------------------------------


class TestBuilder:
    """Test operation graph builder."""

    def test_add_operation(self):
        """Test adding operations to builder."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "Test"})

        assert "task1" in builder._nodes
        assert len(builder.graph.nodes) == 1

    def test_add_with_dependencies(self):
        """Test adding operations with dependencies."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=["task1"])

        # Verify dependency edge exists
        task1 = builder._nodes["task1"]
        task2 = builder._nodes["task2"]
        successors = builder.graph.get_successors(task1)
        assert task2 in successors

    def test_depends_on_method(self):
        """Test adding dependencies via depends_on method."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.depends_on("task2", "task1")

        task1 = builder._nodes["task1"]
        task2 = builder._nodes["task2"]
        successors = builder.graph.get_successors(task1)
        assert task2 in successors

    def test_cycle_detection(self):
        """Test that cycles are detected during build."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        # Create cycle
        builder.depends_on("task2", "task1")
        builder.depends_on("task1", "task2")

        with pytest.raises(ValueError, match="cycle"):
            builder.build()

    def test_aggregation_operation(self):
        """Test creating aggregation operations."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add_aggregation(
            "summary", "operate", {"instruction": "Summarize"}, source_names=["task1", "task2"]
        )

        # Verify aggregation metadata
        agg_op = builder._nodes["summary"]
        assert agg_op.metadata.get("aggregation") is True
        assert "aggregation_sources" in agg_op.metadata

    def test_duplicate_name_error(self):
        """Test error on duplicate operation names."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="already exists"):
            builder.add("task1", "generate", {"instruction": "Duplicate"})

    def test_missing_dependency_error(self):
        """Test error when dependency not found."""
        builder = Builder()

        with pytest.raises(ValueError, match="not found"):
            builder.add("task1", "generate", {"instruction": "Test"}, depends_on=["missing"])


# -------------------------------------------------------------------------
# DependencyAwareExecutor Tests
# -------------------------------------------------------------------------


class TestDependencyAwareExecutor:
    """Test dependency-aware execution engine."""

    async def test_basic_execution(self, session_with_model):
        """Test executing a single operation."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            GenerateParams(
                instruction="Say hello",
                imodel=model,
                return_as="text",
            ),
        )
        graph = builder.build()

        results = await flow(session, graph, branch=branch, verbose=False)

        assert "task1" in results
        assert "mock response" in results["task1"]

    async def test_dependency_coordination(self, session_with_model):
        """Test operations wait for dependencies."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        execution_order = []

        # Create custom factories that track execution order
        async def factory_with_tracking(session, branch, parameters):
            task_name = parameters.get("task_name")
            execution_order.append(task_name)
            await asyncio.sleep(0.01)  # Small delay
            return f"result_{task_name}"

        # Register factories
        # Register to session's per-session registry
        session.operations.register("tracked", factory_with_tracking)

        builder = Builder()
        builder.add("task1", "tracked", {"task_name": "task1"})
        builder.add("task2", "tracked", {"task_name": "task2"}, depends_on=["task1"])
        graph = builder.build()

        await flow(session, graph, branch=branch, verbose=False)

        # Verify task1 executed before task2
        assert execution_order.index("task1") < execution_order.index("task2")

    async def test_parallel_execution(self, session_with_model):
        """Test independent operations run in parallel."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            GenerateParams(instruction="First", imodel=model, return_as="text"),
        )
        builder.add(
            "task2",
            "generate",
            GenerateParams(instruction="Second", imodel=model, return_as="text"),
        )
        # No dependencies - should run in parallel

        graph = builder.build()

        import time

        start = time.time()
        results = await flow(session, graph, branch=branch, verbose=False)
        elapsed = time.time() - start

        # Both should complete
        assert "task1" in results
        assert "task2" in results

        # If truly parallel, should be faster than sequential
        # (This is a weak assertion but demonstrates parallelism)
        assert elapsed < 1.0  # Very generous timeout

    async def test_aggregation_support(self, session_with_model):
        """Test aggregation operations wait for all sources."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            GenerateParams(instruction="First", imodel=model, return_as="text"),
        )
        builder.add(
            "task2",
            "generate",
            GenerateParams(instruction="Second", imodel=model, return_as="text"),
        )
        builder.add_aggregation(
            "summary",
            "generate",
            GenerateParams(instruction="Summarize", imodel=model, return_as="text"),
            source_names=["task1", "task2"],
        )

        graph = builder.build()
        results = await flow(session, graph, branch=branch, verbose=False)

        # All tasks should complete
        assert "task1" in results
        assert "task2" in results
        assert "summary" in results

    async def test_error_handling_stop_on_error(self, session_with_model):
        """Test execution captures errors when operations fail."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        # Create failing factory
        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Intentional failure")

        # Register to session's per-session registry
        session.operations.register("failing", failing_factory)

        builder = Builder()
        builder.add("task1", "failing", {})
        graph = builder.build()

        # Note: gather(return_exceptions=True) means exceptions don't propagate
        # The executor captures the error but returns empty results
        results = await flow(session, graph, branch=branch, stop_on_error=True, verbose=False)

        # Verify operation failed (no result for task1)
        assert "task1" not in results

    async def test_max_concurrent(self, session_with_model):
        """Test concurrency limiting with max_concurrent."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0

        async def concurrent_tracker(session, branch, parameters):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return "done"

        # Register to session's per-session registry
        session.operations.register("concurrent_tracker", concurrent_tracker)

        # Create 10 independent operations
        builder = Builder()
        for i in range(10):
            builder.add(f"task{i}", "concurrent_tracker", {})

        graph = builder.build()

        # Limit to 3 concurrent
        await flow(session, graph, branch=branch, max_concurrent=3, verbose=False)

        # Should never exceed 3 concurrent
        assert max_concurrent_seen <= 3


# -------------------------------------------------------------------------
# Factory Tests
# -------------------------------------------------------------------------


class TestFactories:
    """Test operation factories."""

    async def test_generate_basic(self, session_with_model):
        """Test generate factory - stateless, returns text by default."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            imodel="mock_model",
            instruction="Say hello",
            return_as="text",
        )
        result = await generate(session, branch, params)

        assert "mock response" in result
        # Verify NO messages added (stateless)
        messages = session.messages[branch]
        assert len(messages) == 0

    async def test_generate_with_model_params(self, mock_model):
        """Test generate factory with model parameters."""
        session = Session()
        session.services.register(mock_model, update=True)
        branch = session.create_branch(name="test", resources={mock_model.name})

        params = GenerateParams(
            imodel="mock_model",
            instruction="Test",
            return_as="text",
            imodel_kwargs={"model": "gpt-4", "temperature": 0.7},
        )
        result = await generate(session, branch, params)

        assert "mock response" in result

    async def test_operate_with_response_model(self):
        """Test operate factory with Pydantic response model."""
        from dataclasses import dataclass
        from unittest.mock import AsyncMock

        from lionpride import Event, EventStatus
        from lionpride.services.providers.oai_chat import OAIChatEndpoint
        from lionpride.services.types.imodel import iModel

        @dataclass
        class MockResponse:
            status: str = "success"
            data: str = ""
            raw_response: dict | None = None
            metadata: dict | None = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"content": self.data}
                if self.metadata is None:
                    self.metadata = {}

        # Create model with structured JSON response
        endpoint = OAIChatEndpoint(config=None, name="mock", api_key="mock-key")
        model = iModel(backend=endpoint)

        async def mock_invoke_json(
            model_name: str | None = None, messages: list | None = None, **kwargs
        ):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    # JSON needs to be wrapped with the model name (lowercase)
                    self.execution.response = MockResponse(
                        status="success",
                        data='{"exampleoutput": {"analysis": "test analysis", "confidence": 0.85}}',
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_json))

        session = Session()
        session.services.register(model, update=True)
        branch = session.create_branch(
            name="test", capabilities={"exampleoutput"}, resources={model.name}
        )

        # Use flat params pattern - OperateParams inherits from CommunicateParams
        params = OperateParams(
            generate=GenerateParams(
                instruction="Analyze",
                imodel=model,
                request_model=ExampleOutput,
                imodel_kwargs={"model_name": "gpt-4.1-mini"},
            ),
            parse=ParseParams(),
            strict_validation=False,
            capabilities={"exampleoutput"},  # Explicit capabilities required
        )
        result = await operate(session, branch, params)

        # Result should be a model with exampleoutput field containing ExampleOutput instance
        assert hasattr(result, "exampleoutput")
        assert result.exampleoutput.analysis == "test analysis"
        assert result.exampleoutput.confidence == 0.85

    async def test_operate_skip_validation(self, session_with_model):
        """Test operate factory with skip_validation=True."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # skip_validation path doesn't need capabilities
        params = OperateParams(
            generate=GenerateParams(
                instruction="Test",
                imodel=model,
                imodel_kwargs={"model_name": "gpt-4.1-mini"},
            ),
            parse=ParseParams(),
            skip_validation=True,
        )
        result = await operate(session, branch, params)

        # Should return raw response when skipping validation
        assert result == "mock response"

    async def test_factory_return_as_message(self, session_with_model):
        """Test generate with return_as='message'."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            imodel="mock_model",
            instruction="Test",
            return_as="message",
        )
        result = await generate(session, branch, params)

        # Should return Message instance with metadata
        assert isinstance(result, Message)
        assert "mock response" in str(result.content)
        assert "raw_response" in result.metadata

    async def test_factory_error_on_failed_status(self):
        """Test factories raise error when model returns failed status."""
        from unittest.mock import AsyncMock

        from lionpride import Event, EventStatus
        from lionpride.services.providers.oai_chat import OAIChatEndpoint
        from lionpride.services.types.imodel import iModel

        # Create model that returns failed status
        endpoint = OAIChatEndpoint(config=None, name="failing", api_key="mock-key")
        model = iModel(backend=endpoint)

        async def mock_invoke_failed(
            model_name: str | None = None, messages: list | None = None, **kwargs
        ):
            class FailedCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.FAILED
                    self.execution.error = "Model invocation failed"

            return FailedCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_failed))

        session = Session()
        session.services.register(model, update=True)
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Test",
            imodel=model,
            return_as="text",
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
        )
        with pytest.raises(ExecutionError, match="Generation did not complete"):
            await generate(session, branch, params)


# -------------------------------------------------------------------------
# Session.conduct() Tests
# -------------------------------------------------------------------------


class TestSessionConduct:
    """Test Session.conduct() unified operation interface."""

    async def test_conduct_basic_generate(self, session_with_model):
        """Test basic conduct() with generate operation."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Hello",
            imodel=model.name,
            return_as="text",
        )
        op = await session.conduct("generate", branch, params=params)

        # Verify operation lifecycle
        from lionpride.core import EventStatus

        assert op.status == EventStatus.COMPLETED
        assert op.response is not None
        assert "mock response" in op.response

    async def test_conduct_with_explicit_imodel(self, mock_model):
        """Test conduct() with explicitly passed imodel."""
        session = Session()
        session.services.register(mock_model, update=True)
        branch = session.create_branch(name="test", resources={mock_model.name})

        # Pass imodel explicitly via params
        params = GenerateParams(
            instruction="Hello",
            imodel=mock_model.name,
            return_as="text",
        )
        op = await session.conduct("generate", branch, params=params)

        from lionpride.core import EventStatus

        assert op.status == EventStatus.COMPLETED
        assert op.response is not None

    async def test_conduct_explicit_imodel_overrides_default(self, mock_model):
        """Test explicit imodel overrides default_generate_model."""
        from dataclasses import dataclass
        from unittest.mock import AsyncMock

        from lionpride import Event, EventStatus
        from lionpride.services.providers.oai_chat import OAIChatEndpoint
        from lionpride.services.types.imodel import iModel

        @dataclass
        class MockResponse:
            status: str = "success"
            data: str = ""
            raw_response: dict = None
            metadata: dict = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"id": "mock-id", "choices": []}
                if self.metadata is None:
                    self.metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

        # Create second model with different response
        endpoint = OAIChatEndpoint(config=None, name="explicit_model", api_key="mock-key")
        explicit_model = iModel(backend=endpoint)

        async def explicit_mock_invoke(model_name=None, messages=None, **kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    self.execution.response = MockResponse(
                        status="success", data="explicit model response"
                    )

            return MockCalling()

        object.__setattr__(explicit_model, "invoke", AsyncMock(side_effect=explicit_mock_invoke))

        session = Session(default_generate_model=mock_model)
        session.services.register(mock_model, update=True)
        session.services.register(explicit_model, update=True)
        branch = session.create_branch(name="test", resources={explicit_model.name})

        # Pass explicit imodel via params
        params = GenerateParams(
            instruction="Hello",
            imodel=explicit_model.name,
            return_as="text",
        )
        op = await session.conduct("generate", branch, params=params)

        assert "explicit model response" in op.response

    async def test_conduct_without_imodel_fails(self):
        """Test conduct() without imodel fails during operation execution."""
        from lionpride.core import EventStatus

        session = Session()
        branch = session.create_branch(name="test")

        # Without imodel, the operation should fail (no model registered)
        params = GenerateParams(
            instruction="Hello",
            imodel="nonexistent_model",  # Model doesn't exist in registry
            return_as="text",
        )
        op = await session.conduct("generate", branch, params=params)

        # Operation should fail due to missing imodel
        assert op.status == EventStatus.FAILED

    async def test_conduct_branch_uses_default(self, session_with_model):
        """Test conduct() uses default_branch when None passed."""
        session, model = session_with_model
        branch1 = session.create_branch(name="first", resources={model.name})
        _branch2 = session.create_branch(name="second")

        # Explicitly set default branch
        session.set_default_branch(branch1)
        assert session.default_branch is branch1

        # Pass None for branch - should use default_branch
        params = GenerateParams(
            instruction="Hello",
            imodel=model.name,
            return_as="text",
        )
        op = await session.conduct("generate", None, params=params)

        from lionpride.core import EventStatus

        assert op.status == EventStatus.COMPLETED
        assert op._branch is branch1

    async def test_conduct_no_branch_raises_error(self, mock_model):
        """Test conduct() raises error when no branch and no default set."""
        session = Session()
        session.services.register(mock_model, update=True)
        # No branches created yet
        assert len(session.branches) == 0
        assert session.default_branch is None

        params = GenerateParams(
            instruction="Hello",
            imodel=mock_model.name,
            return_as="text",
        )
        with pytest.raises(RuntimeError, match="No branch provided and no default branch set"):
            await session.conduct("generate", None, params=params)

    async def test_conduct_branch_by_uuid(self, session_with_model):
        """Test conduct() accepts branch by UUID."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Pass UUID string
        params = GenerateParams(
            instruction="Hello",
            imodel=model.name,
            return_as="text",
        )
        op = await session.conduct("generate", str(branch.id), params=params)

        from lionpride.core import EventStatus

        assert op.status == EventStatus.COMPLETED

    async def test_conduct_branch_by_name(self, session_with_model):
        """Test conduct() accepts branch by name."""
        session, model = session_with_model
        _branch = session.create_branch(name="named_branch", resources={model.name})

        # Pass branch name (string lookup)
        params = GenerateParams(
            instruction="Hello",
            imodel=model.name,
            return_as="text",
        )
        op = await session.conduct("generate", "named_branch", params=params)

        from lionpride.core import EventStatus

        assert op.status == EventStatus.COMPLETED

    async def test_conduct_unknown_operation_fails(self, session_with_model):
        """Test conduct() with unregistered operation returns FAILED status."""
        from lionpride.core import EventStatus

        session, model = session_with_model
        branch = session.create_branch(name="test")

        # KeyError is caught by Event.invoke() machinery
        params = GenerateParams(
            instruction="Hello",
            imodel=model.name,
            return_as="text",
        )
        op = await session.conduct("nonexistent_operation", branch, params=params)

        # Verify operation failed with appropriate error
        assert op.status == EventStatus.FAILED
        assert "not registered" in str(op.execution.error)

    async def test_conduct_returns_operation_node(self, session_with_model):
        """Test conduct() returns Operation node with full lifecycle."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Hello",
            imodel=model.name,
            return_as="text",
        )
        op = await session.conduct("generate", branch, params=params)

        from lionpride.operations import Operation

        # Verify it's an Operation node
        assert isinstance(op, Operation)
        assert op.operation_type == "generate"
        # Parameters contain operation-specific args (imodel resolved internally)
        assert op.parameters is not None

        # Verify lifecycle properties
        from lionpride.core import EventStatus

        assert op.status == EventStatus.COMPLETED
        assert op.response is not None
        assert op.execution is not None

    async def test_conduct_operation_is_bound(self, session_with_model):
        """Test conducted operation is bound to session/branch."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Hello",
            imodel=model.name,
            return_as="text",
        )
        op = await session.conduct("generate", branch, params=params)

        # Verify binding
        assert op._session is session
        assert op._branch is branch

    async def test_conduct_operate_with_response_model(self):
        """Test conduct() with operate and response_model."""
        from dataclasses import dataclass
        from unittest.mock import AsyncMock

        from lionpride import Event, EventStatus
        from lionpride.services.providers.oai_chat import OAIChatEndpoint
        from lionpride.services.types.imodel import iModel

        @dataclass
        class MockResponse:
            status: str = "success"
            data: str = ""
            raw_response: dict | None = None
            metadata: dict | None = None

            def __post_init__(self):
                if self.raw_response is None:
                    self.raw_response = {"content": self.data}
                if self.metadata is None:
                    self.metadata = {}

        endpoint = OAIChatEndpoint(config=None, name="json_model", api_key="mock-key")
        json_model = iModel(backend=endpoint)

        async def mock_invoke_json(model_name=None, messages=None, **kwargs):
            class MockCalling(Event):
                def __init__(self):
                    super().__init__()
                    self.status = EventStatus.COMPLETED
                    # Response must be wrapped in lowercase model name as key
                    self.execution.response = MockResponse(
                        status="success",
                        data='{"exampleoutput": {"analysis": "test result", "confidence": 0.95}}',
                    )

            return MockCalling()

        object.__setattr__(json_model, "invoke", AsyncMock(side_effect=mock_invoke_json))

        session = Session()
        session.services.register(json_model, update=True)
        branch = session.create_branch(
            name="test",
            resources={json_model.name},
            capabilities={"exampleoutput"},  # Required for operate with response_model
        )

        # Use flat params pattern - OperateParams inherits from CommunicateParams
        params = OperateParams(
            generate=GenerateParams(
                instruction="Analyze this",
                imodel=json_model.name,
                imodel_kwargs={"model_name": "gpt-4.1-mini"},
                request_model=ExampleOutput,
            ),
            parse=ParseParams(),
            strict_validation=False,
            capabilities={"exampleoutput"},  # Explicit capabilities required
        )
        op = await session.conduct("operate", branch, params=params)

        # Debug: check operation status and error
        from lionpride.core import EventStatus

        assert op.status == EventStatus.COMPLETED, f"Operation failed: {op.execution.error}"

        # Verify structured response - Operable wraps the model in a field with lowercase model name
        assert hasattr(op.response, "exampleoutput")
        assert op.response.exampleoutput.analysis == "test result"
        assert op.response.exampleoutput.confidence == 0.95


# -------------------------------------------------------------------------
# Operation Lifecycle Tests
# -------------------------------------------------------------------------


class TestOperationLifecycle:
    """Test Operation bind/invoke lifecycle."""

    async def test_operation_require_binding_error(self):
        """Test _require_binding raises RuntimeError when not bound."""
        from lionpride.operations import Operation

        op = Operation(
            operation_type="generate",
            parameters={"instruction": "Test"},
        )

        with pytest.raises(RuntimeError, match="not bound"):
            op._require_binding()

    async def test_operation_bind_returns_self(self, session_with_model):
        """Test bind() returns self for chaining."""
        from lionpride.operations import Operation

        session, model = session_with_model
        branch = session.create_branch(name="test")

        op = Operation(
            operation_type="generate",
            parameters={"imodel": model, "messages": [{"role": "user", "content": "Hi"}]},
        )

        result = op.bind(session, branch)
        assert result is op

    async def test_operation_chained_bind_invoke(self, session_with_model):
        """Test fluent bind().invoke() pattern."""
        from lionpride.core import EventStatus
        from lionpride.operations import Operation

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Hi",
            imodel=model.name,
            return_as="text",
        )
        op = Operation(operation_type="generate", parameters=params)

        # Fluent pattern
        await op.bind(session, branch).invoke()

        assert op.status == EventStatus.COMPLETED
        assert op.response is not None


# -------------------------------------------------------------------------
# Session.default_branch Tests
# -------------------------------------------------------------------------


class TestSessionDefaultBranch:
    """Test Session.default_branch property and set_default_branch() method."""

    def test_set_default_branch_by_instance(self):
        """Test set_default_branch() with Branch instance."""
        session = Session()
        _branch1 = session.create_branch(name="first")
        branch2 = session.create_branch(name="second")

        assert session.default_branch is None

        session.set_default_branch(branch2)
        assert session.default_branch is branch2

    def test_set_default_branch_by_uuid(self):
        """Test set_default_branch() with UUID."""
        session = Session()
        _branch1 = session.create_branch(name="first")
        branch2 = session.create_branch(name="second")

        session.set_default_branch(branch2.id)
        assert session.default_branch is branch2

    def test_set_default_branch_by_name(self):
        """Test set_default_branch() with branch name."""
        session = Session()
        _branch1 = session.create_branch(name="first")
        branch2 = session.create_branch(name="second")

        session.set_default_branch("second")
        assert session.default_branch is branch2

    def test_set_default_branch_not_found_raises(self):
        """Test set_default_branch() raises NotFoundError for non-existent branch."""
        from uuid import uuid4

        from lionpride.errors import NotFoundError

        session = Session()
        _branch = session.create_branch(name="test")

        fake_uuid = uuid4()
        with pytest.raises(NotFoundError):
            session.set_default_branch(fake_uuid)

    def test_set_default_branch_switches_between_branches(self):
        """Test switching default branch multiple times."""
        session = Session()
        branch1 = session.create_branch(name="first")
        branch2 = session.create_branch(name="second")

        # No default initially
        assert session.default_branch is None

        session.set_default_branch(branch1)
        assert session.default_branch is branch1

        session.set_default_branch(branch2)
        assert session.default_branch is branch2

        session.set_default_branch(branch1)
        assert session.default_branch is branch1

    def test_default_branch_returns_none_after_deletion(self):
        """Test default_branch returns None if deleted."""
        session = Session()
        branch = session.create_branch(name="test")
        session.set_default_branch(branch)

        assert session.default_branch is branch

        # Remove branch from conversations
        session.conversations.remove_progression(branch.id)

        # Should return None gracefully
        assert session.default_branch is None

    async def test_conduct_raises_after_default_deleted(self, mock_model):
        """Test conduct() raises RuntimeError after default branch deleted."""
        session = Session()
        session.services.register(mock_model, update=True)
        branch = session.create_branch(name="test")
        session.set_default_branch(branch)

        assert session.default_branch is branch

        # Delete the branch
        session.conversations.remove_progression(branch.id)

        # conduct() should raise RuntimeError
        params = GenerateParams(
            instruction="Hello",
            imodel=mock_model.name,
            return_as="text",
        )
        with pytest.raises(RuntimeError, match="No branch provided"):
            await session.conduct("generate", None, params=params)

    def test_default_branch_via_init(self):
        """Test default_branch set via __init__ parameter."""
        session = Session(default_branch="main")

        assert session.default_branch is not None
        assert session.default_branch.name == "main"


# -------------------------------------------------------------------------
# OperationRegistry Edge Cases
# -------------------------------------------------------------------------


class TestOperationRegistryEdgeCases:
    """Test OperationRegistry edge cases and dunder methods."""

    def test_registry_unregister(self):
        """Test unregistering an operation."""
        registry = OperationRegistry(auto_register_defaults=False)

        async def test_factory(session, branch, parameters):
            return "test"

        registry.register("test_op", test_factory)
        assert registry.has("test_op")

        result = registry.unregister("test_op")
        assert result is True
        assert not registry.has("test_op")

    def test_registry_unregister_nonexistent(self):
        """Test unregistering non-existent operation returns False."""
        registry = OperationRegistry(auto_register_defaults=False)
        result = registry.unregister("nonexistent")
        assert result is False

    def test_registry_duplicate_registration_error(self):
        """Test duplicate registration raises ValueError."""
        registry = OperationRegistry(auto_register_defaults=False)

        async def factory1(session, branch, parameters):
            return "first"

        async def factory2(session, branch, parameters):
            return "second"

        registry.register("test_op", factory1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_op", factory2)

    def test_registry_duplicate_with_override(self):
        """Test duplicate registration with override=True succeeds."""
        registry = OperationRegistry(auto_register_defaults=False)

        async def factory1(session, branch, parameters):
            return "first"

        async def factory2(session, branch, parameters):
            return "second"

        registry.register("test_op", factory1)
        registry.register("test_op", factory2, override=True)

        assert registry.get("test_op") is factory2

    def test_registry_contains(self):
        """Test __contains__ (in operator)."""
        registry = OperationRegistry(auto_register_defaults=False)

        async def test_factory(session, branch, parameters):
            return "test"

        registry.register("test_op", test_factory)

        assert "test_op" in registry
        assert "nonexistent" not in registry

    def test_registry_len(self):
        """Test __len__."""
        registry = OperationRegistry(auto_register_defaults=False)
        assert len(registry) == 0

        async def test_factory(session, branch, parameters):
            return "test"

        registry.register("op1", test_factory)
        assert len(registry) == 1

        registry.register("op2", test_factory)
        assert len(registry) == 2

    def test_registry_repr(self):
        """Test __repr__."""
        registry = OperationRegistry(auto_register_defaults=False)

        async def test_factory(session, branch, parameters):
            return "test"

        registry.register("test_op", test_factory)

        repr_str = repr(registry)
        assert "OperationRegistry" in repr_str
        assert "test_op" in repr_str


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


class TestOperationsIntegration:
    """Integration tests for full operation flows."""

    async def test_multi_level_dependency_graph(self, session_with_model):
        """Test complex multi-level dependency graph."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        builder = Builder()

        # Helper to create generate params
        def gen_params(instruction: str):
            return GenerateParams(
                instruction=instruction,
                imodel=model.name,
                return_as="text",
                imodel_kwargs={"model_name": "gpt-4.1-mini"},
            )

        # Level 1: Root
        builder.add("root", "generate", gen_params("Root"))

        # Level 2: Depends on root
        builder.add("level2_a", "generate", gen_params("2A"), depends_on=["root"])
        builder.add("level2_b", "generate", gen_params("2B"), depends_on=["root"])

        # Level 3: Depends on level 2
        builder.add("level3_a", "generate", gen_params("3A"), depends_on=["level2_a"])
        builder.add("level3_b", "generate", gen_params("3B"), depends_on=["level2_b"])

        # Aggregation: Depends on all level 3
        builder.add_aggregation(
            "final", "generate", gen_params("Final"), source_names=["level3_a", "level3_b"]
        )

        graph = builder.build()
        results = await flow(session, graph, branch=branch, verbose=False)

        # All operations should complete
        assert len(results) == 6
        assert all(
            task in results
            for task in ["root", "level2_a", "level2_b", "level3_a", "level3_b", "final"]
        )

    async def test_session_message_integration(self, session_with_model):
        """Test generate is stateless - doesn't add messages to session."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        # Add system message
        from lionpride.session.messages import Message, SystemContent

        sys_msg = Message(
            content=SystemContent(system_message="You are helpful"),
            sender="system",
            recipient="user",
        )
        session.add_message(sys_msg, branches=branch)

        # Execute generate operation
        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "imodel": "mock_model",
                "messages": [{"role": "user", "content": "Test"}],
            },
        )
        graph = builder.build()

        await flow(session, graph, branch=branch, verbose=False)

        # Verify only system message remains (generate is stateless)
        messages = session.messages[branch]
        assert len(messages) == 1  # only system message

        # Verify system message is still there
        system = session.get_branch_system(branch)
        assert system is not None
        assert system.id == sys_msg.id
