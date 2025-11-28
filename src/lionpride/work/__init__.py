# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Work system - declarative workflow orchestration.

Core concepts:
- Form: Unit of work with assignment DSL ("branch: a, b -> c")
- Report: Workflow orchestrator - subclass to define output schemas
- flow_report: Executes report via compiled graph (parallel-capable)

Usage:
    from lionpride.work import Report, flow_report

    class MyReport(Report):
        # Output schemas as class attributes
        analysis: AnalysisModel
        insights: InsightsModel
        recommendations: RecommendationsModel

        # Workflow definition
        assignment = "topic -> recommendations"
        form_assignments = [
            "topic -> analysis",
            "analysis -> insights",
            "insights -> recommendations",
        ]

    # Initialize and execute
    report = MyReport()
    report.initialize(topic="AI in software development")
    result = await flow_report(session, branch, report)
"""

from .form import Form, parse_assignment
from .report import Report
from .runner import flow_report

__all__ = (
    "Form",
    "Report",
    "flow_report",
    "parse_assignment",
)
