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

from lionpride.operations import Builder, OperationRegistry, flow
from lionpride.operations.flow import DependencyAwareExecutor
from lionpride.operations.node import create_operation
from lionpride.operations.operate.factory import operate
from lionpride.operations.operate.generate import generate
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
        assert "aggregation_sources" in agg_op.parameters

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
        branch = session.create_branch(name="test")

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "Say hello",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        graph = builder.build()

        results = await flow(session, branch, graph, verbose=False)

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

        await flow(session, branch, graph, verbose=False)

        # Verify task1 executed before task2
        assert execution_order.index("task1") < execution_order.index("task2")

    async def test_parallel_execution(self, session_with_model):
        """Test independent operations run in parallel."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "First",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        builder.add(
            "task2",
            "generate",
            {
                "instruction": "Second",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        # No dependencies - should run in parallel

        graph = builder.build()

        import time

        start = time.time()
        results = await flow(session, branch, graph, verbose=False)
        elapsed = time.time() - start

        # Both should complete
        assert "task1" in results
        assert "task2" in results

        # If truly parallel, should be faster than sequential
        # (This is a weak assertion but demonstrates parallelism)
        assert elapsed < 1.0  # Very generous timeout

    async def test_context_inheritance(self, session_with_model):
        """Test operations receive predecessor results in context."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        # Create factory that uses context
        async def context_consumer(session, branch, parameters):
            context = parameters.get("context", {})
            return {"received_context": context}

        # Register to session's per-session registry
        session.operations.register("context_consumer", context_consumer)

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "First",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        builder.add("task2", "context_consumer", {}, depends_on=["task1"])
        graph = builder.build()

        results = await flow(session, branch, graph, verbose=False)

        # Verify task2 received task1's result in context
        assert "task2" in results
        context = results["task2"]["received_context"]
        assert "task1_result" in context

    async def test_aggregation_support(self, session_with_model):
        """Test aggregation operations wait for all sources."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "First",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        builder.add(
            "task2",
            "generate",
            {
                "instruction": "Second",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        builder.add_aggregation(
            "summary",
            "generate",
            {
                "instruction": "Summarize",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
            source_names=["task1", "task2"],
        )

        graph = builder.build()
        results = await flow(session, branch, graph, verbose=False)

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
        results = await flow(session, branch, graph, stop_on_error=True, verbose=False)

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
        await flow(session, branch, graph, max_concurrent=3, verbose=False)

        # Should never exceed 3 concurrent
        assert max_concurrent_seen <= 3


# -------------------------------------------------------------------------
# Factory Tests
# -------------------------------------------------------------------------


class TestFactories:
    """Test operation factories."""

    async def test_generate_basic(self, session_with_model):
        """Test generate factory - stateless, returns text by default."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        result = await generate(
            session,
            branch,
            {
                "imodel": "mock_model",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )

        assert "mock response" in result
        # Verify NO messages added (stateless)
        messages = session.messages[branch]
        assert len(messages) == 0

    async def test_generate_with_model_params(self, mock_model):
        """Test generate factory with model parameters."""
        session = Session()
        branch = session.create_branch(name="test")
        session.services.register(mock_model, update=True)

        result = await generate(
            session,
            branch,
            {
                "imodel": "mock_model",
                "messages": [{"role": "user", "content": "Test"}],
                "model": "gpt-4",
                "temperature": 0.7,
            },
        )

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
                    self.execution.response = MockResponse(
                        status="success", data='{"analysis": "test analysis", "confidence": 0.85}'
                    )

            return MockCalling()

        object.__setattr__(model, "invoke", AsyncMock(side_effect=mock_invoke_json))

        session = Session()
        session.services.register(model, update=True)
        branch = session.create_branch(name="test")

        result = await operate(
            session,
            branch,
            {
                "instruction": "Analyze",
                "imodel": model,
                "response_model": ExampleOutput,
                "skip_validation": False,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )

        # Result should be parsed ExampleOutput instance
        assert isinstance(result, ExampleOutput)
        assert result.analysis == "test analysis"
        assert result.confidence == 0.85

    async def test_operate_skip_validation(self, session_with_model):
        """Test operate factory with skip_validation=True."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        result = await operate(
            session,
            branch,
            {
                "instruction": "Test",
                "imodel": model,
                "response_model": ExampleOutput,
                "skip_validation": True,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )

        # Should return raw response when skipping validation
        assert result == "mock response"

    async def test_factory_return_as_message(self, session_with_model):
        """Test generate with return_as='message'."""
        session, _model = session_with_model
        branch = session.create_branch(name="test")

        result = await generate(
            session,
            branch,
            {
                "imodel": "mock_model",
                "messages": [{"role": "user", "content": "Test"}],
                "return_as": "message",
            },
        )

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
        branch = session.create_branch(name="test")

        with pytest.raises(RuntimeError, match="Model invocation failed"):
            await generate(
                session,
                branch,
                {
                    "instruction": "Test",
                    "imodel": model,
                    "model_kwargs": {"model_name": "gpt-4.1-mini"},
                },
            )


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


class TestOperationsIntegration:
    """Integration tests for full operation flows."""

    async def test_multi_level_dependency_graph(self, session_with_model):
        """Test complex multi-level dependency graph."""
        session, model = session_with_model
        branch = session.create_branch(name="test")

        builder = Builder()

        # Level 1: Root
        builder.add(
            "root",
            "generate",
            {
                "instruction": "Root",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )

        # Level 2: Depends on root
        builder.add(
            "level2_a",
            "generate",
            {"instruction": "2A", "imodel": model, "model_kwargs": {"model_name": "gpt-4.1-mini"}},
            depends_on=["root"],
        )
        builder.add(
            "level2_b",
            "generate",
            {"instruction": "2B", "imodel": model, "model_kwargs": {"model_name": "gpt-4.1-mini"}},
            depends_on=["root"],
        )

        # Level 3: Depends on level 2
        builder.add(
            "level3_a",
            "generate",
            {"instruction": "3A", "imodel": model, "model_kwargs": {"model_name": "gpt-4.1-mini"}},
            depends_on=["level2_a"],
        )
        builder.add(
            "level3_b",
            "generate",
            {"instruction": "3B", "imodel": model, "model_kwargs": {"model_name": "gpt-4.1-mini"}},
            depends_on=["level2_b"],
        )

        # Aggregation: Depends on all level 3
        builder.add_aggregation(
            "final",
            "generate",
            {
                "instruction": "Final",
                "imodel": model,
                "model_kwargs": {"model_name": "gpt-4.1-mini"},
            },
            source_names=["level3_a", "level3_b"],
        )

        graph = builder.build()
        results = await flow(session, branch, graph, verbose=False)

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

        await flow(session, branch, graph, verbose=False)

        # Verify only system message remains (generate is stateless)
        messages = session.messages[branch]
        assert len(messages) == 1  # only system message

        # Verify system message is still there
        system = session.get_branch_system(branch)
        assert system is not None
        assert system.id == sys_msg.id
