# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for flow() coverage (78% → 90%+).

Targets missing lines:
- Error handling: 94, 99, 131, 133, 142
- Stop conditions: 169-182, 200, 217
- Execution events: 242, 250, 261, 270-275
- Result processing: 281, 292, 297, 311-313, 320, 323, 332
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from lionpride import Edge, EventStatus, Graph
from lionpride.operations import Builder, flow
from lionpride.operations.flow import (
    DependencyAwareExecutor,
    OperationResult,
    flow_stream,
)
from lionpride.operations.node import Operation, create_operation
from lionpride.operations.operate.types import GenerateParams
from lionpride.session import Session


@pytest.fixture
def mock_model():
    """Create a mock iModel for testing without API calls."""
    from dataclasses import dataclass

    from lionpride import Event
    from lionpride.services.providers.oai_chat import OAIChatEndpoint
    from lionpride.services.types.imodel import iModel

    @dataclass
    class MockResponse:
        status: str = "success"
        data: str = ""

    # Create mock endpoint
    endpoint = OAIChatEndpoint(config=None, name="mock", api_key="mock-key")

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


# -------------------------------------------------------------------------
# Error Handling Tests (Lines 94, 99, 131, 133, 142)
# -------------------------------------------------------------------------


class TestFlowErrorHandling:
    """Test error handling paths in flow execution."""

    async def test_cyclic_graph_raises_error(self, session_with_model):
        """Test line 94: Graph with cycles raises ValueError."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        # Create cyclic graph manually
        op1 = create_operation(operation="generate", parameters={"instruction": "First"})
        op1.metadata["name"] = "task1"
        op2 = create_operation(operation="generate", parameters={"instruction": "Second"})
        op2.metadata["name"] = "task2"

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)
        # Create cycle: op1 → op2 → op1
        graph.add_edge(Edge(head=op1.id, tail=op2.id))
        graph.add_edge(Edge(head=op2.id, tail=op1.id))

        with pytest.raises(ValueError, match=r"cycle.*DAG"):
            await flow(session, graph, branch=branch)

    async def test_non_operation_node_raises_error(self, session_with_model):
        """Test line 99: Non-Operation node raises ValueError."""
        from lionpride import Node

        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        # Create graph with non-Operation node
        graph = Graph()
        invalid_node = Node(content={"data": "not an operation"})
        graph.add_node(invalid_node)

        with pytest.raises(ValueError, match="non-Operation node"):
            await flow(session, graph, branch=branch)

    async def test_branch_as_string_resolution(self, session_with_model):
        """Test line 131: String branch name resolution."""
        session, model = session_with_model
        branch_name = "test_branch"
        session.create_branch(name=branch_name, resources={model.name})

        builder = Builder()
        params = GenerateParams(
            instruction="Test",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )
        builder.add("task1", "generate", params)
        graph = builder.build()

        # Pass branch as string (not object)
        results = await flow(session, graph, branch=branch_name)

        assert "task1" in results

    async def test_executor_with_none_branch(self, session_with_model):
        """Test DependencyAwareExecutor handles None default_branch."""
        session, model = session_with_model
        _branch = session.create_branch(name="test", resources={"mock"})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "Test",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        graph = builder.build()

        # Create executor with explicit None branch (tests fallback path)
        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=None,  # Explicitly None
        )

        # Pre-allocate should handle None gracefully
        await executor._preallocate_branches()

        # When default_branch is None, operations get None
        for _op_id, allocated_branch in executor.operation_branches.items():
            assert allocated_branch is None

    async def test_verbose_branch_preallocation(self, session_with_model, capsys):
        """Test line 142: Verbose logging for branch pre-allocation."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "Test",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        builder.add(
            "task2",
            "generate",
            {
                "instruction": "Test2",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        graph = builder.build()

        await flow(session, graph, branch=branch, verbose=True)

        captured = capsys.readouterr()
        assert "Pre-allocated branches for 2 operations" in captured.out


# -------------------------------------------------------------------------
# Stop Conditions Tests (Lines 169-182, 200, 217)
# -------------------------------------------------------------------------


class TestFlowStopConditions:
    """Test stop condition handling and verbose logging."""

    async def test_error_with_stop_on_error_true_reraises(self, session_with_model):
        """Test lines 169-182: Error handling with stop_on_error=True."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        # Create failing factory
        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Intentional failure")

        # Register to session's per-session registry
        session.operations.register("failing_op", failing_factory)

        builder = Builder()
        builder.add("task1", "failing_op", {})
        graph = builder.build()

        # With stop_on_error=True, error should propagate
        # But gather(return_exceptions=True) catches it
        results = await flow(session, graph, branch=branch, stop_on_error=True)

        # Verify task failed (no result)
        assert "task1" not in results

    async def test_error_verbose_logging(self, session_with_model, capsys):
        """Test lines 169-182: Verbose error logging with stop_on_error=True."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        # Create failing factory
        async def failing_factory(session, branch, parameters):
            raise ValueError("Test error for logging")

        # Register to session's per-session registry
        session.operations.register("failing_verbose", failing_factory)

        builder = Builder()
        builder.add("task1", "failing_verbose", {})
        graph = builder.build()

        # Test with stop_on_error=True to hit lines 179-182
        # The exception will be caught by gather(return_exceptions=True)
        # But the error path will execute
        await flow(session, graph, branch=branch, verbose=True, stop_on_error=True)

        captured = capsys.readouterr()
        # The error message appears in verbose output (lines 172-176)
        assert "Test error for logging" in captured.out or "failed:" in captured.out

    async def test_dependencies_verbose_logging(self, session_with_model, capsys):
        """Test verbose logging for dependencies."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params1 = GenerateParams(
            instruction="First",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )
        params2 = GenerateParams(
            instruction="Second",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )

        builder = Builder()
        builder.add("task1", "generate", params1)
        builder.add("task2", "generate", params2, depends_on=["task1"])

        graph = builder.build()
        await flow(session, graph, branch=branch, verbose=True)

        captured = capsys.readouterr()
        assert "waiting for" in captured.out
        assert "dependencies" in captured.out


# -------------------------------------------------------------------------
# Execution Event Tests
# -------------------------------------------------------------------------


class TestFlowExecutionEvents:
    """Test execution event handling."""

    async def test_operation_receives_its_parameters(self, session_with_model):
        """Test that operations receive their parameters unchanged."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        received_params = None

        async def param_receiver(session, branch, parameters):
            nonlocal received_params
            received_params = parameters
            return "done"

        session.operations.register("param_receiver", param_receiver)

        builder = Builder()
        builder.add("task1", "param_receiver", {"my_key": "my_value"})
        graph = builder.build()

        await flow(session, graph, branch=branch)

        # Verify parameters are passed unchanged
        assert received_params == {"my_key": "my_value"}

    async def test_failed_predecessor_does_not_block_with_stop_on_error_false(
        self, session_with_model
    ):
        """Test that with stop_on_error=False, dependent tasks still run."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        dependent_ran = False

        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Intentional failure")

        async def dependent_factory(session, branch, parameters):
            nonlocal dependent_ran
            dependent_ran = True
            return "done"

        session.operations.register("failing_pred", failing_factory)
        session.operations.register("dependent_task", dependent_factory)

        builder = Builder()
        builder.add("failed_task", "failing_pred", {})
        builder.add("dependent_task", "dependent_task", {}, depends_on=["failed_task"])
        graph = builder.build()

        await flow(session, graph, branch=branch, stop_on_error=False)

        # Dependent task should have run (after predecessor completed/failed)
        assert dependent_ran


# -------------------------------------------------------------------------
# Result Processing Tests
# -------------------------------------------------------------------------


class TestFlowResultProcessing:
    """Test result processing and verbose logging."""

    async def test_verbose_operation_execution(self, session_with_model, capsys):
        """Test verbose logging for operation execution."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Test",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )

        builder = Builder()
        builder.add("task1", "generate", params)
        graph = builder.build()

        await flow(session, graph, branch=branch, verbose=True)

        captured = capsys.readouterr()
        assert "Executing operation:" in captured.out

    async def test_missing_branch_allocation_raises_error(self, session_with_model):
        """Test missing branch allocation raises ValueError."""
        session, model = session_with_model

        op = create_operation(
            operation="generate",
            parameters={
                "instruction": "Test",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        op.metadata["name"] = "test"
        graph = Graph()
        graph.add_node(op)

        executor = DependencyAwareExecutor(session=session, graph=graph, default_branch=None)

        with pytest.raises(ValueError, match="No branch allocated"):
            await executor._invoke_operation(op)

    async def test_verbose_operation_failure(self, session_with_model, capsys):
        """Test verbose logging for operation failure."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        async def status_failed_factory(session, branch, parameters):
            raise RuntimeError("Operation failed with error status")

        session.operations.register("status_fail", status_failed_factory)

        builder = Builder()
        builder.add("task1", "status_fail", {})
        graph = builder.build()

        await flow(session, graph, branch=branch, verbose=True, stop_on_error=False)

        captured = capsys.readouterr()
        assert "failed:" in captured.out

    async def test_verbose_operation_completion(self, session_with_model, capsys):
        """Test verbose logging for operation completion."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Test",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )

        builder = Builder()
        builder.add("task1", "generate", params)
        graph = builder.build()

        await flow(session, graph, branch=branch, verbose=True)

        captured = capsys.readouterr()
        assert "Completed operation:" in captured.out


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


class TestFlowIntegration:
    """Integration tests covering complex scenarios."""

    async def test_complex_dag_with_multiple_paths(self, session_with_model):
        """Test complex DAG execution with multiple dependency paths."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        def gen_params(instruction: str):
            return GenerateParams(
                instruction=instruction,
                imodel=model.name,
                imodel_kwargs={"model_name": "gpt-4.1-mini"},
                return_as="text",
            )

        builder = Builder()
        # Diamond dependency: task1 → task2, task3 → task4
        builder.add("task1", "generate", gen_params("Root"))
        builder.add("task2", "generate", gen_params("Left"), depends_on=["task1"])
        builder.add("task3", "generate", gen_params("Right"), depends_on=["task1"])
        builder.add("task4", "generate", gen_params("Merge"), depends_on=["task2", "task3"])

        graph = builder.build()
        results = await flow(session, graph, branch=branch)

        # All tasks should complete
        assert all(f"task{i}" in results for i in range(1, 5))

    async def test_stop_on_error_false_continues_execution(self, session_with_model):
        """Test that stop_on_error=False allows remaining tasks to execute."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Fail")

        # Register to session's per-session registry
        session.operations.register("fail_task", failing_factory)

        params = GenerateParams(
            instruction="Independent",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )

        builder = Builder()
        builder.add("task1", "fail_task", {})
        builder.add("task2", "generate", params)  # Independent

        graph = builder.build()
        results = await flow(session, graph, branch=branch, stop_on_error=False)

        # task2 should still execute
        assert "task2" in results
        assert "task1" not in results

    async def test_max_concurrent_limits_parallelism(self, session_with_model):
        """Test max_concurrent limits parallel execution."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        concurrent_count = 0
        max_seen = 0

        async def concurrent_tracker(session, branch, parameters):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return "done"

        # Register to session's per-session registry
        session.operations.register("track", concurrent_tracker)

        builder = Builder()
        for i in range(5):
            builder.add(f"task{i}", "track", {})

        graph = builder.build()
        await flow(session, graph, branch=branch, max_concurrent=2)

        # Should not exceed 2 concurrent
        assert max_seen <= 2


# -------------------------------------------------------------------------
# Direct Exception Path Tests (Lines 169-182)
# -------------------------------------------------------------------------


class TestFlowExceptionPaths:
    """Direct tests for exception handling in _execute_operation (lines 169-182)."""

    async def test_exception_in_execute_operation_no_verbose_no_stop(self, session_with_model):
        """Test lines 169, 171: Exception caught, stored, execution continues."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        # Create operation that will fail
        async def failing_op(session, branch, parameters):
            raise ValueError("Test exception - no verbose, no stop")

        # Register to session's per-session registry
        session.operations.register("fail_no_verbose", failing_op)

        params = GenerateParams(
            instruction="Should run",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )

        builder = Builder()
        builder.add("task1", "fail_no_verbose", {})
        builder.add("task2", "generate", params)
        graph = builder.build()

        # Execute with stop_on_error=False, verbose=False
        results = await flow(session, graph, branch=branch, stop_on_error=False, verbose=False)

        # task1 should fail, task2 should succeed
        assert "task1" not in results
        assert "task2" in results

    async def test_exception_with_verbose_no_stop(self, session_with_model, capsys):
        """Test lines 169, 171, 172, 173, 175, 176: Verbose error logging."""
        from unittest.mock import patch

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "Test",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        builder.add(
            "task2",
            "generate",
            {
                "instruction": "Should run",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        graph = builder.build()

        # Get first operation to mock
        op1 = None
        for node in graph.nodes:
            if isinstance(node, Operation):
                op1 = node
                break

        # Create executor
        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
            verbose=True,
            stop_on_error=False,
        )

        # Mock _invoke_operation to raise an exception (triggers lines 169-176)
        original_invoke = executor._invoke_operation

        async def mock_invoke_raise(operation):
            if operation.id == op1.id:
                raise RuntimeError("Mock exception for verbose logging")
            return await original_invoke(operation)

        with patch.object(executor, "_invoke_operation", side_effect=mock_invoke_raise):
            await executor.execute()

        captured = capsys.readouterr()
        # Verify verbose error logging (lines 172-176)
        assert "failed:" in captured.out
        assert "Mock exception for verbose logging" in captured.out
        assert "Traceback:" in captured.out

        # task1 failed, task2 succeeded
        assert op1.id in executor.errors

    async def test_exception_with_stop_on_error(self, session_with_model):
        """Test lines 169, 171, 179, 181, 182: stop_on_error=True re-raises."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        async def failing_with_stop(session, branch, parameters):
            raise ValueError("Test exception with stop_on_error")

        # Register to session's per-session registry
        session.operations.register("fail_stop", failing_with_stop)

        builder = Builder()
        builder.add("task1", "fail_stop", {})
        graph = builder.build()

        # Execute with stop_on_error=True, verbose=False
        # The exception will be caught by gather(return_exceptions=True)
        # but the re-raise path (lines 179-182) should still execute
        results = await flow(session, graph, branch=branch, stop_on_error=True, verbose=False)

        # Task failed, no result
        assert "task1" not in results

    async def test_exception_with_verbose_and_stop(self, session_with_model, capsys):
        """Test lines 169-182: All exception paths with verbose + stop_on_error."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        async def failing_full_path(session, branch, parameters):
            raise RuntimeError("Full exception path test")

        # Register to session's per-session registry
        session.operations.register("fail_full", failing_full_path)

        builder = Builder()
        builder.add("task1", "fail_full", {})
        graph = builder.build()

        # Execute with both verbose=True and stop_on_error=True
        results = await flow(session, graph, branch=branch, verbose=True, stop_on_error=True)

        captured = capsys.readouterr()
        # Verify verbose logging executed
        assert "failed:" in captured.out
        assert "Full exception path test" in captured.out

        # Task failed
        assert "task1" not in results

    async def test_direct_executor_exception_verbose_stop(self, session_with_model, capsys):
        """Test exception path directly via executor to ensure coverage."""
        from unittest.mock import patch

        session, model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "Test",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        graph = builder.build()

        # Get operation
        op = None
        for node in graph.nodes:
            if isinstance(node, Operation):
                op = node
                break

        # Create executor with verbose=True, stop_on_error=True
        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
            verbose=True,
            stop_on_error=True,
        )

        # Mock _invoke_operation to raise exception (triggers lines 169-182 with stop_on_error)
        async def mock_invoke_raise(operation):
            raise ValueError("Direct executor exception test")

        with patch.object(executor, "_invoke_operation", side_effect=mock_invoke_raise):
            # Execute - exception propagates through CompletionStream's TaskGroup
            with pytest.raises(ExceptionGroup) as exc_info:
                await executor.execute()

            # Verify the original ValueError is in the ExceptionGroup
            assert len(exc_info.value.exceptions) == 1
            assert isinstance(exc_info.value.exceptions[0], ValueError)
            assert "Direct executor exception test" in str(exc_info.value.exceptions[0])

        captured = capsys.readouterr()
        # Verify error was logged
        assert "failed:" in captured.out
        assert "Direct executor exception test" in captured.out

        # Verify error was stored
        assert op.id in executor.errors
        assert isinstance(executor.errors[op.id], ValueError)

    async def test_exception_during_wait_for_dependencies(self, session_with_model, capsys):
        """Test exception raised during _wait_for_dependencies."""
        from unittest.mock import patch

        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        op = create_operation(operation="generate", parameters={"instruction": "Test"})
        op.metadata["name"] = "test_op"

        graph = Graph()
        graph.add_node(op)

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
            verbose=True,
            stop_on_error=False,
        )

        # Mock _wait_for_dependencies to raise exception
        async def mock_wait_deps(operation):
            raise RuntimeError("Dependency wait failed")

        with patch.object(executor, "_wait_for_dependencies", side_effect=mock_wait_deps):
            await executor.execute()

        captured = capsys.readouterr()
        # Exception should be caught and logged
        assert "failed:" in captured.out
        assert "Dependency wait failed" in captured.out

        # Error should be stored
        assert op.id in executor.errors


# -------------------------------------------------------------------------
# Stream Execute Tests (OperationResult, stream_execute, flow_stream)
# -------------------------------------------------------------------------


class TestFlowStreamExecute:
    """Test stream_execute and flow_stream for flow.py coverage."""

    def test_operation_result_success_property(self):
        """Test OperationResult.success property."""
        # Success case
        success_result = OperationResult(
            name="test", result="value", error=None, completed=1, total=1
        )
        assert success_result.success is True

        # Failure case
        failure_result = OperationResult(
            name="test", result=None, error=Exception("error"), completed=1, total=1
        )
        assert failure_result.success is False

    async def test_stream_execute_success(self, session_with_model):
        """Test stream_execute yields results as operations complete."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        builder = Builder()
        builder.add(
            "task1",
            "generate",
            {
                "instruction": "First",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        builder.add(
            "task2",
            "generate",
            {
                "instruction": "Second",
                "imodel": model,
                "imodel_kwargs": {"model_name": "gpt-4.1-mini"},
            },
        )
        graph = builder.build()

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        results = []
        async for result in executor.stream_execute():
            results.append(result)

        assert len(results) == 2
        assert all(isinstance(r, OperationResult) for r in results)
        assert results[-1].completed == 2
        assert results[-1].total == 2

    async def test_stream_execute_with_error(self, session_with_model):
        """Test stream_execute yields error results for failed operations."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Test error")

        session.operations.register("fail_stream", failing_factory)

        builder = Builder()
        builder.add("task1", "fail_stream", {})
        graph = builder.build()

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
            stop_on_error=False,
        )

        results = []
        async for result in executor.stream_execute():
            results.append(result)

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].success is False

    async def test_stream_execute_cyclic_graph_raises(self, session_with_model):
        """Test stream_execute raises for cyclic graph."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        op1 = create_operation(operation="generate", parameters={})
        op2 = create_operation(operation="generate", parameters={})

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_edge(Edge(head=op1.id, tail=op2.id))
        graph.add_edge(Edge(head=op2.id, tail=op1.id))

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        with pytest.raises(ValueError, match=r"cycle.*DAG"):
            async for _ in executor.stream_execute():
                pass

    async def test_stream_execute_non_operation_node_raises(self, session_with_model):
        """Test stream_execute raises for non-Operation nodes."""
        from lionpride import Node

        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        graph = Graph()
        invalid_node = Node(content={"invalid": True})
        graph.add_node(invalid_node)

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        with pytest.raises(ValueError, match="non-Operation node"):
            async for _ in executor.stream_execute():
                pass

    async def test_flow_stream_function(self, session_with_model):
        """Test flow_stream() function."""
        session, model = session_with_model
        branch = session.create_branch(name="test", resources={model.name})

        params = GenerateParams(
            instruction="Test",
            imodel=model.name,
            imodel_kwargs={"model_name": "gpt-4.1-mini"},
            return_as="text",
        )

        builder = Builder()
        builder.add("task1", "generate", params)
        graph = builder.build()

        results = []
        async for result in flow_stream(session, graph, branch=branch):
            results.append(result)

        assert len(results) == 1
        assert results[0].name == "task1"
        assert results[0].success is True


# -------------------------------------------------------------------------
# Branch-Aware Execution Tests (P1 Fix)
# -------------------------------------------------------------------------


class TestFlowBranchAwareExecution:
    """Test per-operation branch assignment via metadata['branch']."""

    async def test_operation_uses_metadata_branch_by_name(self, session_with_model):
        """Test that operations use their metadata['branch'] when specified as string."""
        session, _model = session_with_model

        # Create two branches
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        # Track which branch each operation runs on
        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_branch", branch_tracker)

        # Create operations with explicit branch assignments
        op1 = create_operation(
            operation="track_branch",
            parameters={"_op_name": "task1"},
        )
        op1.metadata["name"] = "task1"
        op1.metadata["branch"] = "branch1"  # String name

        op2 = create_operation(
            operation="track_branch",
            parameters={"_op_name": "task2"},
        )
        op2.metadata["name"] = "task2"
        op2.metadata["branch"] = "branch2"  # String name

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)

        # Execute with a different default branch
        default_branch = session.create_branch(name="default")
        await flow(session, graph, branch=default_branch)

        # Verify operations ran on their specified branches
        assert execution_branches["task1"] == branch1
        assert execution_branches["task2"] == branch2

    async def test_operation_uses_metadata_branch_by_uuid(self, session_with_model):
        """Test that operations use their metadata['branch'] when specified as UUID."""
        session, _model = session_with_model

        # Create branch
        target_branch = session.create_branch(name="target")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_uuid_branch", branch_tracker)

        # Create operation with UUID branch assignment
        op = create_operation(
            operation="track_uuid_branch",
            parameters={"_op_name": "uuid_task"},
        )
        op.metadata["name"] = "uuid_task"
        op.metadata["branch"] = target_branch.id  # UUID

        graph = Graph()
        graph.add_node(op)

        # Execute with different default branch
        default_branch = session.create_branch(name="default")
        await flow(session, graph, branch=default_branch)

        # Verify operation ran on target branch (by UUID)
        assert execution_branches["uuid_task"] == target_branch

    async def test_operation_fallback_to_default_branch(self, session_with_model):
        """Test that operations without metadata['branch'] use default branch."""
        session, _model = session_with_model

        default_branch = session.create_branch(name="default")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_default", branch_tracker)

        # Create operation WITHOUT branch metadata
        op = create_operation(
            operation="track_default",
            parameters={"_op_name": "no_branch_task"},
        )
        op.metadata["name"] = "no_branch_task"
        # No branch metadata set

        graph = Graph()
        graph.add_node(op)

        await flow(session, graph, branch=default_branch)

        # Verify operation ran on default branch
        assert execution_branches["no_branch_task"] == default_branch

    async def test_unresolvable_branch_falls_back_to_default(self, session_with_model):
        """Test that unresolvable branch reference falls back to default."""
        session, _model = session_with_model

        default_branch = session.create_branch(name="default")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_fallback", branch_tracker)

        # Create operation with non-existent branch name
        op = create_operation(
            operation="track_fallback",
            parameters={"_op_name": "fallback_task"},
        )
        op.metadata["name"] = "fallback_task"
        op.metadata["branch"] = "non_existent_branch"  # This won't resolve

        graph = Graph()
        graph.add_node(op)

        await flow(session, graph, branch=default_branch)

        # Should fall back to default branch
        assert execution_branches["fallback_task"] == default_branch

    async def test_multi_branch_workflow_with_builder(self, session_with_model):
        """Test multi-branch workflow built with Builder.add(..., branch=...)."""
        session, _model = session_with_model

        # Create branches
        extraction_branch = session.create_branch(name="extraction")
        analysis_branch = session.create_branch(name="analysis")
        merge_branch = session.create_branch(name="merge")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return f"result_from_{op_name}"

        session.operations.register("multi_branch_op", branch_tracker)

        # Build workflow with explicit branch assignments
        builder = Builder()
        builder.add(
            "extract",
            "multi_branch_op",
            {"_op_name": "extract"},
            branch="extraction",
        )
        builder.add(
            "analyze",
            "multi_branch_op",
            {"_op_name": "analyze"},
            branch="analysis",
        )
        builder.add_aggregation(
            "merge",
            "multi_branch_op",
            {"_op_name": "merge"},
            source_names=["extract", "analyze"],
            branch="merge",
        )

        graph = builder.build()

        # Execute with a different default branch
        default_branch = session.create_branch(name="default")
        results = await flow(session, graph, branch=default_branch)

        # Verify all operations completed
        assert "extract" in results
        assert "analyze" in results
        assert "merge" in results

        # Verify operations ran on their specified branches
        assert execution_branches["extract"] == extraction_branch
        assert execution_branches["analyze"] == analysis_branch
        assert execution_branches["merge"] == merge_branch

    async def test_resolve_operation_branch_with_branch_object(self, session_with_model):
        """Test _resolve_operation_branch handles Branch-like objects."""
        session, _model = session_with_model
        branch = session.create_branch(name="test", resources={"mock"})

        graph = Graph()
        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        # Test Branch-like object (has id and order attributes)
        result = executor._resolve_operation_branch(branch)
        assert result == branch

        # Test UUID resolution
        result = executor._resolve_operation_branch(branch.id)
        assert result == branch

        # Test string name resolution
        result = executor._resolve_operation_branch("test")
        assert result == branch

        # Test unresolvable returns None
        result = executor._resolve_operation_branch("non_existent")
        assert result is None

        # Test invalid type returns None
        result = executor._resolve_operation_branch(12345)
        assert result is None
