# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for ReportExecutor.

The new ReportExecutor uses completion events for dependency coordination
and passes context between forms via report.available_data.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from lionpride.work import FormResult, Report, ReportExecutor, execute_report, stream_report


class SimpleOutput(BaseModel):
    """Simple output model for testing."""

    result: str = Field(description="Result value")


class AnalysisOutput(BaseModel):
    """Analysis output for testing."""

    summary: str
    score: float


class InsightsOutput(BaseModel):
    """Insights output for testing."""

    patterns: list[str]


class TestFormResult:
    """Tests for FormResult dataclass."""

    def test_success_property_true_when_no_error(self):
        """Test success is True when error is None."""
        result = FormResult(name="test", result="value", error=None)
        assert result.success is True

    def test_success_property_false_when_error(self):
        """Test success is False when error is set."""
        result = FormResult(name="test", result=None, error=ValueError("failed"))
        assert result.success is False

    def test_progress_tracking(self):
        """Test completed/total tracking."""
        result = FormResult(name="step1", result="v", completed=3, total=5)
        assert result.completed == 3
        assert result.total == 5


class TestReportExecutor:
    """Tests for ReportExecutor class."""

    @pytest.mark.asyncio
    async def test_execute_returns_deliverable(self):
        """Test that execute returns the deliverable."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["input -> output"]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        mock_output = SimpleOutput(result="success")

        with patch("lionpride.work.executor.operate") as mock_operate:
            mock_operate.return_value = mock_output
            executor = ReportExecutor(session=mock_session, report=report)
            deliverable = await executor.execute()

        assert "output" in deliverable
        assert isinstance(deliverable["output"], SimpleOutput)
        assert deliverable["output"].result == "success"

    @pytest.mark.asyncio
    async def test_stream_execute_yields_form_results(self):
        """Test that stream_execute yields FormResult objects."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["input -> output"]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        mock_output = SimpleOutput(result="streamed")

        with patch("lionpride.work.executor.operate") as mock_operate:
            mock_operate.return_value = mock_output
            executor = ReportExecutor(session=mock_session, report=report)

            results = []
            async for result in executor.stream_execute():
                results.append(result)

        assert len(results) == 1
        assert isinstance(results[0], FormResult)
        assert results[0].name == "output"
        assert results[0].success is True
        assert results[0].completed == 1
        assert results[0].total == 1

    @pytest.mark.asyncio
    async def test_sequential_dependencies_via_completion_events(self):
        """Test that sequential dependencies are handled via completion events."""

        class TestReport(Report):
            step1: SimpleOutput | None = None
            step2: SimpleOutput | None = None

            assignment: str = "input -> step2"
            form_assignments: list[str] = [
                "input -> step1",
                "step1 -> step2",
            ]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        execution_order = []

        async def mock_operate_side_effect(session, branch, params):
            if "step1" in params.capabilities:
                execution_order.append("step1")
                return SimpleOutput(result="s1")
            elif "step2" in params.capabilities:
                execution_order.append("step2")
                return SimpleOutput(result="s2")
            return SimpleOutput(result="unknown")

        with patch("lionpride.work.executor.operate", side_effect=mock_operate_side_effect):
            executor = ReportExecutor(session=mock_session, report=report)
            result = await executor.execute()

        # Verify sequential execution (step1 before step2)
        assert execution_order == ["step1", "step2"]
        assert "step2" in result

    @pytest.mark.asyncio
    async def test_parallel_execution_with_max_concurrent(self):
        """Test parallel execution respects max_concurrent."""

        class TestReport(Report):
            a: SimpleOutput | None = None
            b: SimpleOutput | None = None
            c: SimpleOutput | None = None

            assignment: str = "input -> c"
            form_assignments: list[str] = [
                "input -> a",
                "input -> b",
                "a, b -> c",
            ]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        async def mock_operate_side_effect(session, branch, params):
            if "a" in params.capabilities:
                return SimpleOutput(result="a")
            elif "b" in params.capabilities:
                return SimpleOutput(result="b")
            elif "c" in params.capabilities:
                return SimpleOutput(result="c")
            return SimpleOutput(result="unknown")

        with patch("lionpride.work.executor.operate", side_effect=mock_operate_side_effect):
            executor = ReportExecutor(session=mock_session, report=report, max_concurrent=1)
            result = await executor.execute()

        # Verify all forms completed
        assert len(report.completed_forms) == 3
        assert "c" in result

    @pytest.mark.asyncio
    async def test_context_passing_between_forms(self):
        """Test that context is passed via available_data."""

        class TestReport(Report):
            analysis: AnalysisOutput | None = None
            insights: InsightsOutput | None = None

            assignment: str = "topic -> insights"
            form_assignments: list[str] = [
                "topic -> analysis",
                "analysis -> insights",
            ]

        report = TestReport()
        report.initialize(topic="test topic")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        contexts_received = []

        async def mock_operate_side_effect(session, branch, params):
            contexts_received.append(params.generate.context)
            if "analysis" in params.capabilities:
                return AnalysisOutput(summary="test", score=0.9)
            elif "insights" in params.capabilities:
                return InsightsOutput(patterns=["p1"])
            return None

        with patch("lionpride.work.executor.operate", side_effect=mock_operate_side_effect):
            executor = ReportExecutor(session=mock_session, report=report)
            await executor.execute()

        # First call (analysis): should have topic in context
        assert contexts_received[0] is not None
        assert "topic" in contexts_received[0]

        # Second call (insights): should have analysis in context
        assert contexts_received[1] is not None
        assert "analysis" in contexts_received[1]

    @pytest.mark.asyncio
    async def test_branch_prefix_resolution(self):
        """Test that branch prefix is respected."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["worker: input -> output"]

        report = TestReport()
        report.initialize(input="test")

        # Check that form has branch_name set
        form = next(iter(report.forms))
        assert form.branch_name == "worker"

        mock_session = MagicMock()
        mock_branch = MagicMock()
        mock_branch.resources = {"default_model"}
        mock_worker_branch = MagicMock()
        mock_worker_branch.resources = {"default_model"}
        mock_session.default_branch = mock_branch
        mock_session.get_branch = MagicMock(return_value=mock_worker_branch)

        with patch("lionpride.work.executor.operate") as mock_operate:
            mock_operate.return_value = SimpleOutput(result="done")
            executor = ReportExecutor(session=mock_session, report=report, branch=mock_branch)
            await executor.execute()

        # Verify get_branch was called with the worker branch name
        mock_session.get_branch.assert_called_with("worker")


class TestCycleDetection:
    """Tests for cycle detection in form dependencies."""

    @pytest.mark.asyncio
    async def test_detects_direct_cycle(self):
        """Test that executor detects A -> B -> A cycle."""

        class TestReport(Report):
            a: SimpleOutput | None = None
            b: SimpleOutput | None = None

            assignment: str = "input -> b"
            form_assignments: list[str] = [
                "b -> a",  # a depends on b
                "a -> b",  # b depends on a - cycle!
            ]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        executor = ReportExecutor(session=mock_session, report=report)

        with pytest.raises(RuntimeError, match="Circular dependency"):
            await executor.execute()


class TestExecuteReportFunction:
    """Tests for execute_report convenience function."""

    @pytest.mark.asyncio
    async def test_execute_report_returns_deliverable(self):
        """Test that execute_report returns the deliverable."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["input -> output"]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        mock_output = SimpleOutput(result="success")

        with patch("lionpride.work.executor.operate") as mock_operate:
            mock_operate.return_value = mock_output
            deliverable = await execute_report(session=mock_session, report=report)

        assert "output" in deliverable
        assert deliverable["output"].result == "success"


class TestStreamReportFunction:
    """Tests for stream_report convenience function."""

    @pytest.mark.asyncio
    async def test_stream_report_yields_form_results(self):
        """Test that stream_report yields FormResult objects."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["input -> output"]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        mock_output = SimpleOutput(result="streamed")

        with patch("lionpride.work.executor.operate") as mock_operate:
            mock_operate.return_value = mock_output
            results = []
            async for result in stream_report(session=mock_session, report=report):
                results.append(result)

        assert len(results) == 1
        assert isinstance(results[0], FormResult)
        assert results[0].name == "output"
