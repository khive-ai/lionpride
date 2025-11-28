# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Form/Report workflow demo using lionpride.work module.

Demonstrates:
- Declarative workflow definition via Report subclass
- Class attributes define output schemas
- Assignment-based dependency resolution
- Automatic form scheduling
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from lionpride import Session, iModel
from lionpride.work import Report, flow_report

# =============================================================================
# Pydantic Models (structured outputs)
# =============================================================================


class Analysis(BaseModel):
    """Analysis of a topic."""

    summary: str = Field(description="Brief summary of the topic")
    key_points: list[str] = Field(description="Key points identified")
    challenges: list[str] = Field(description="Main challenges or issues")


class Insights(BaseModel):
    """Insights derived from analysis."""

    patterns: list[str] = Field(description="Patterns observed")
    opportunities: list[str] = Field(description="Opportunities identified")
    risks: list[str] = Field(description="Potential risks")


class Recommendations(BaseModel):
    """Final recommendations."""

    actions: list[str] = Field(description="Recommended actions")
    priorities: list[str] = Field(description="Priority order")
    next_steps: str = Field(description="Immediate next steps")


# =============================================================================
# Report Definition - class attributes define output schemas
# =============================================================================


class TopicAnalysisReport(Report):
    """Workflow for analyzing a topic and producing recommendations."""

    # Output schemas (optional - filled during execution)
    analysis: Analysis | None = None
    insights: Insights | None = None
    recommendations: Recommendations | None = None

    # Workflow definition
    assignment: str = "topic -> recommendations"
    form_assignments: list[str] = [
        "topic -> analysis",
        "analysis -> insights",
        "insights -> recommendations",
    ]


# =============================================================================
# Demo
# =============================================================================


async def main():
    """Run the workflow demo."""
    # Create session with model
    model = iModel(
        provider="openai",
        model="gpt-4.1-mini",
        name="gpt4mini",
    )
    session = Session(default_generate_model=model)

    # Create branch with capabilities and resources
    branch = session.create_branch(
        name="workflow",
        capabilities={"analysis", "insights", "recommendations"},
        resources={"gpt4mini"},
    )

    # Create report instance
    report = TopicAnalysisReport()

    # Input
    topic = """
    The adoption of AI coding assistants in software development teams.
    Consider productivity impacts, code quality, learning curves, and team dynamics.
    """

    # Initialize with input data
    report.initialize(topic=topic.strip())

    print("=" * 60)
    print("Form/Report Workflow Demo")
    print("=" * 60)
    print(f"\nWorkflow: {report.assignment}")
    print(f"Forms: {report.form_assignments}")
    print(f"\nTopic: {topic.strip()[:100]}...")

    # Execute workflow via compiled graph
    print("\n" + "-" * 60)
    print("EXECUTION (via flow_report)")
    print("-" * 60)

    deliverable = await flow_report(
        session=session,
        branch=branch,
        report=report,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("DELIVERABLE")
    print("=" * 60)

    recommendations = deliverable.get("recommendations")
    if recommendations:
        # Extract data from model
        if hasattr(recommendations, "model_dump"):
            rec_data = recommendations.model_dump()
            # Handle nested structure
            if "recommendations" in rec_data and isinstance(rec_data["recommendations"], dict):
                rec_data = rec_data["recommendations"]
        elif isinstance(recommendations, dict):
            rec_data = recommendations
        else:
            rec_data = {}

        if "actions" in rec_data:
            print("\nActions:")
            for i, action in enumerate(rec_data["actions"][:5], 1):
                print(f"  {i}. {action[:80]}...")

        if "priorities" in rec_data:
            print("\nPriorities:")
            for i, priority in enumerate(rec_data["priorities"][:3], 1):
                print(f"  {i}. {priority[:80]}...")

        if "next_steps" in rec_data:
            print("\nNext Steps:")
            print(f"  {rec_data['next_steps'][:200]}...")

    # Show intermediate data via Pile
    print("\n" + "=" * 60)
    print("INTERMEDIATE DATA (via Pile[Form])")
    print("=" * 60)

    for form in report.completed_forms:
        print(f"\n{form.assignment}:")
        if form.output and hasattr(form.output, "model_dump"):
            data = form.output.model_dump()
            for k, v in data.items():
                if isinstance(v, list):
                    print(f"  {k}: {len(v)} items")
                elif isinstance(v, str):
                    print(f"  {k}: {v[:60]}...")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
