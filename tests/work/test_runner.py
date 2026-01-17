# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for flow_report runner.

The new flow_report uses Report's next_forms() scheduling and calls operate()
directly instead of building a graph and calling flow().
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from lionpride.work import Report, flow_report


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


class TestFlowReportUnit:
    """Unit tests for flow_report."""

    @pytest.mark.asyncio
    async def test_flow_report_returns_deliverable(self):
        """Test that flow_report returns the deliverable."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["input -> output"]

        report = TestReport()
        report.initialize(input="test")

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {
            "default_model"
        }  # Single resource for auto-resolve
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        mock_output = SimpleOutput(result="success")

        with patch("lionpride.work.executor.operate") as mock_operate:
            mock_operate.return_value = mock_output
            deliverable = await flow_report(
                session=mock_session,
                report=report,
            )

        assert "output" in deliverable
        assert isinstance(deliverable["output"], SimpleOutput)
        assert deliverable["output"].result == "success"

    @pytest.mark.asyncio
    async def test_flow_report_fills_forms(self):
        """Test that flow_report fills forms with results."""

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

        mock_output = SimpleOutput(result="filled")

        with patch("lionpride.work.executor.operate") as mock_operate:
            mock_operate.return_value = mock_output
            await flow_report(
                session=mock_session,
                report=report,
            )

        # Check form was filled
        form = next(iter(report.forms))
        assert form.filled is True
        assert form.output == mock_output

        # Check form was marked completed
        assert len(report.completed_forms) == 1

    @pytest.mark.asyncio
    async def test_flow_report_handles_sequential_dependencies(self):
        """Test that flow_report handles sequential form execution."""

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

        # Track execution order
        execution_order = []

        async def mock_operate_side_effect(session, branch, params):
            # Determine which form based on capabilities
            if "step1" in params.capabilities:
                execution_order.append("step1")
                return SimpleOutput(result="s1")
            elif "step2" in params.capabilities:
                execution_order.append("step2")
                return SimpleOutput(result="s2")
            return SimpleOutput(result="unknown")

        with patch("lionpride.work.executor.operate", side_effect=mock_operate_side_effect):
            result = await flow_report(
                session=mock_session,
                report=report,
            )

        # Verify sequential execution (step1 before step2)
        assert execution_order == ["step1", "step2"]
        assert "step2" in result

    @pytest.mark.asyncio
    async def test_flow_report_parallel_execution(self):
        """Test that independent forms can execute in parallel."""

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
            result = await flow_report(
                session=mock_session,
                report=report,
            )

        # Verify all forms completed
        assert len(report.completed_forms) == 3
        assert "c" in result

    @pytest.mark.asyncio
    async def test_flow_report_with_branch_prefix(self):
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
            await flow_report(
                session=mock_session,
                branch=mock_branch,
                report=report,
            )

        # Verify get_branch was called with the worker branch name
        mock_session.get_branch.assert_called_with("worker")

    @pytest.mark.asyncio
    async def test_flow_report_verbose_output(self, caplog):
        """Test that verbose mode logs progress."""
        import logging

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

        with (
            caplog.at_level(logging.DEBUG, logger="lionpride.work.executor"),
            patch("lionpride.work.executor.operate") as mock_operate,
        ):
            mock_operate.return_value = SimpleOutput(result="done")
            await flow_report(
                session=mock_session,
                report=report,
                verbose=True,
            )

        assert "Executing report" in caplog.text
        assert "Forms: 1" in caplog.text
        assert "Executing form: output" in caplog.text

    @pytest.mark.asyncio
    async def test_flow_report_context_from_available_data(self):
        """Test that context is built from report.available_data."""

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

        # Track context passed to operate
        contexts_received = []

        async def mock_operate_side_effect(session, branch, params):
            contexts_received.append(params.generate.context)
            if "analysis" in params.capabilities:
                return AnalysisOutput(summary="test", score=0.9)
            elif "insights" in params.capabilities:
                return InsightsOutput(patterns=["p1"])
            return None

        with patch("lionpride.work.executor.operate", side_effect=mock_operate_side_effect):
            await flow_report(
                session=mock_session,
                report=report,
            )

        # First call (analysis): should have topic in context
        assert contexts_received[0] is not None
        assert "topic" in contexts_received[0]

        # Second call (insights): should have analysis in context
        assert contexts_received[1] is not None
        assert "analysis" in contexts_received[1]

    @pytest.mark.asyncio
    async def test_flow_report_missing_input_fails(self):
        """Test that flow_report fails when required inputs are missing.

        The new executor-based implementation will attempt to execute
        and fail (rather than detect deadlock via polling).
        """

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["missing_field -> output"]

        report = TestReport()
        report.initialize(input="test")  # Note: missing_field not provided

        mock_session = MagicMock()
        mock_session.default_branch = MagicMock()
        mock_session.default_branch.resources = {"default_model"}
        mock_session.get_branch = MagicMock(return_value=mock_session.default_branch)

        # The executor will fail because the form tries to execute without inputs
        with pytest.raises(ExceptionGroup) as exc_info:
            await flow_report(
                session=mock_session,
                report=report,
            )
        # Check the inner exception has the right message
        assert any("missing required inputs" in str(e) for e in exc_info.value.exceptions)

    @pytest.mark.asyncio
    async def test_flow_report_max_concurrent(self):
        """Test that max_concurrent limits parallel execution."""

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
            result = await flow_report(
                session=mock_session,
                report=report,
                max_concurrent=1,  # Limit to 1 concurrent execution
            )

        # Verify all forms completed
        assert len(report.completed_forms) == 3
        assert "c" in result
