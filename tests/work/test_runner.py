# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for flow_report runner."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from lionpride.operations import flow as operations_flow
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
    async def test_flow_report_builds_graph(self):
        """Test that flow_report builds correct graph structure."""

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

        # Mock the flow function to capture the graph
        captured_graph = None

        async def mock_flow(session, branch, graph, context, max_concurrent, verbose):
            nonlocal captured_graph
            captured_graph = graph
            # Return mock results
            return {
                "analysis": AnalysisOutput(summary="test", score=0.9),
                "insights": InsightsOutput(patterns=["p1"]),
            }

        mock_session = MagicMock()
        mock_branch = MagicMock()

        with patch("lionpride.operations.flow.flow", mock_flow):
            await flow_report(
                session=mock_session,
                branch=mock_branch,
                report=report,
                verbose=False,
            )

        # Verify graph structure
        assert captured_graph is not None
        assert len(captured_graph.nodes) == 2

    @pytest.mark.asyncio
    async def test_flow_report_returns_deliverable(self):
        """Test that flow_report returns the deliverable."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["input -> output"]

        report = TestReport()
        report.initialize(input="test")

        async def mock_flow(session, branch, graph, context, max_concurrent, verbose):
            return {"output": SimpleOutput(result="success")}

        mock_session = MagicMock()
        mock_branch = MagicMock()

        with patch("lionpride.operations.flow.flow", mock_flow):
            deliverable = await flow_report(
                session=mock_session,
                branch=mock_branch,
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

        mock_output = SimpleOutput(result="filled")

        async def mock_flow(session, branch, graph, context, max_concurrent, verbose):
            return {"output": mock_output}

        mock_session = MagicMock()
        mock_branch = MagicMock()

        with patch("lionpride.operations.flow.flow", mock_flow):
            await flow_report(
                session=mock_session,
                branch=mock_branch,
                report=report,
            )

        # Check form was filled
        form = next(iter(report.forms))
        assert form.filled is True
        assert form.output == mock_output

        # Check form was marked completed
        assert len(report.completed_forms) == 1

    @pytest.mark.asyncio
    async def test_flow_report_handles_dependencies(self):
        """Test that flow_report correctly identifies dependencies."""

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

        captured_graph = None

        async def mock_flow(session, branch, graph, context, max_concurrent, verbose):
            nonlocal captured_graph
            captured_graph = graph
            return {
                "step1": SimpleOutput(result="s1"),
                "step2": SimpleOutput(result="s2"),
            }

        mock_session = MagicMock()
        mock_branch = MagicMock()

        with patch("lionpride.operations.flow.flow", mock_flow):
            await flow_report(
                session=mock_session,
                branch=mock_branch,
                report=report,
            )

        # Graph should have 2 nodes and 1 edge (step1 -> step2)
        assert len(captured_graph.nodes) == 2
        assert len(captured_graph.edges) == 1

    @pytest.mark.asyncio
    async def test_flow_report_parallel_independence(self):
        """Test that independent forms have no dependency edges."""

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

        captured_graph = None

        async def mock_flow(session, branch, graph, context, max_concurrent, verbose):
            nonlocal captured_graph
            captured_graph = graph
            return {
                "a": SimpleOutput(result="a"),
                "b": SimpleOutput(result="b"),
                "c": SimpleOutput(result="c"),
            }

        mock_session = MagicMock()
        mock_branch = MagicMock()

        with patch("lionpride.operations.flow.flow", mock_flow):
            await flow_report(
                session=mock_session,
                branch=mock_branch,
                report=report,
            )

        # Graph should have 3 nodes
        # Edges: a depends on nothing, b depends on nothing, c depends on a and b
        # So we have edges for c's dependencies
        assert len(captured_graph.nodes) == 3
        # The number of edges depends on how Builder creates them
        # c depends on both a and b, so at least 2 edges
        assert len(captured_graph.edges) >= 2

    @pytest.mark.asyncio
    async def test_flow_report_with_branch_prefix(self):
        """Test that branch prefix is passed to builder."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["worker: input -> output"]

        report = TestReport()
        report.initialize(input="test")

        # Check that form has branch_name set
        form = next(iter(report.forms))
        assert form.branch_name == "worker"

        async def mock_flow(session, branch, graph, context, max_concurrent, verbose):
            return {"output": SimpleOutput(result="done")}

        mock_session = MagicMock()
        mock_branch = MagicMock()

        with patch("lionpride.operations.flow.flow", mock_flow):
            await flow_report(
                session=mock_session,
                branch=mock_branch,
                report=report,
            )

    @pytest.mark.asyncio
    async def test_flow_report_verbose_output(self, capsys):
        """Test that verbose mode prints progress."""

        class TestReport(Report):
            output: SimpleOutput | None = None

            assignment: str = "input -> output"
            form_assignments: list[str] = ["input -> output"]

        report = TestReport()
        report.initialize(input="test")

        async def mock_flow(session, branch, graph, context, max_concurrent, verbose):
            return {"output": SimpleOutput(result="done")}

        mock_session = MagicMock()
        mock_branch = MagicMock()

        with patch("lionpride.operations.flow.flow", mock_flow):
            await flow_report(
                session=mock_session,
                branch=mock_branch,
                report=report,
                verbose=True,
            )

        captured = capsys.readouterr()
        assert "Compiled 1 forms to graph" in captured.out
        assert "Graph: 1 nodes" in captured.out
