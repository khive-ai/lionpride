# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Work system - declarative workflow orchestration.

Core concepts:
- Form: Unit of work with assignment DSL
- Report: Workflow orchestrator - subclass to define output schemas
- FormResources: Capability declarations for workflow steps
- flow_report: Executes report via compiled graph (parallel-capable)

Assignment DSL:
    "[branch:] [operation(] inputs -> outputs [)] [| resources]"

Examples:
    "context -> plan"
    "orchestrator: operate(context -> plan) | api:gpt4mini"
    "planner: react(a, b -> c) | api_gen:gpt5, api_parse:gpt4, tool:*"

Usage:
    from lionpride.work import Report, flow_report

    class MyReport(Report):
        # Output schemas as class attributes
        analysis: AnalysisModel
        plan: PlanModel

        # Workflow definition with capabilities
        assignment = "context -> plan"
        form_assignments = [
            "orchestrator: context -> analysis | api:gpt4mini",
            "planner: react(analysis -> plan) | api_gen:gpt4o, tool:*",
        ]

    # Initialize and execute
    report = MyReport()
    report.initialize(context="Build a web app")
    result = await flow_report(session, branch, report)
"""

from .capabilities import (
    AmbiguousResourceError,
    CapabilityError,
    FormResources,
    ParsedAssignment,
)
from .form import Form, parse_assignment
from .report import Report
from .runner import flow_report

__all__ = (
    "AmbiguousResourceError",
    "CapabilityError",
    "Form",
    "FormResources",
    "ParsedAssignment",
    "Report",
    "flow_report",
    "parse_assignment",
)
