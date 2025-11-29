# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Example: Code Review Report with claude_code provider.

This demonstrates a realistic use case for the Report and flow_report system:
- Multi-step workflow with data dependencies
- Parallel execution of independent forms
- Using Claude Code CLI as the LLM provider

Usage:
    python examples/code_review_report.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from lionpride import Session, iModel
from lionpride.work import Report, flow_report

# ============================================================================
# Output Schemas - Define the structure of each workflow step's output
# ============================================================================


class CodeAnalysis(BaseModel):
    """Analyze code structure and patterns.

    Examine the provided code and identify its purpose, design patterns,
    dependencies, and overall complexity.
    """

    summary: str = Field(description="Brief summary of what the code does")
    patterns_found: list[str] = Field(description="Design patterns identified")
    dependencies: list[str] = Field(description="Key dependencies/imports")
    complexity_assessment: str = Field(description="Low/Medium/High complexity rating")


class SecurityReview(BaseModel):
    """Perform security analysis of the code.

    Identify vulnerabilities, security anti-patterns, and provide
    recommendations for improving security posture.
    """

    vulnerabilities: list[str] = Field(
        default_factory=list, description="Potential vulnerabilities found"
    )
    best_practices: list[str] = Field(description="Security best practices observed")
    recommendations: list[str] = Field(description="Security improvement suggestions")
    risk_level: str = Field(description="Low/Medium/High risk assessment")


class FinalReport(BaseModel):
    """Synthesize analysis and security findings into executive report.

    Combine the code analysis and security review results to produce
    a consolidated report with actionable recommendations.
    """

    executive_summary: str = Field(description="High-level summary for stakeholders")
    key_findings: list[str] = Field(description="Most important findings from both analyses")
    action_items: list[str] = Field(description="Prioritized action items")
    overall_quality: str = Field(description="Poor/Fair/Good/Excellent rating")


# ============================================================================
# Report Definition - Declare workflow as class attributes
# ============================================================================


class CodeReviewReport(Report):
    """Code review workflow with parallel analysis and consolidated report.

    Workflow:
        code_snippet ──┬──> analysis ──┐
                       └──> security ──┴──> report

    The analysis and security steps run in parallel (independent),
    then the final report aggregates both.
    """

    # Output schemas as class attributes (type introspection)
    analysis: CodeAnalysis | None = None
    security: SecurityReview | None = None
    report: FinalReport | None = None

    # Workflow contract: inputs -> final outputs (need type annotation for Pydantic)
    assignment: str = "code_snippet -> report"

    # Form assignments with explicit operations and resources
    # Each parallel step uses a different model instance to enable concurrency
    # DSL: "[branch:] inputs -> outputs [| api:model_name]"
    form_assignments: list[str] = [
        # Step 1: Analyze code structure (uses sonnet_analysis)
        "code_snippet -> analysis | api:sonnet_analysis",
        # Step 2: Security review (uses sonnet_security - can run in parallel)
        "code_snippet -> security | api:sonnet_security",
        # Step 3: Consolidate into final report (uses sonnet_report)
        # Include code_snippet so final step has context of original code
        "code_snippet, analysis, security -> report | api:sonnet_report",
    ]


# ============================================================================
# Example Execution
# ============================================================================


async def run_code_review(code_snippet: str, verbose: bool = True) -> dict:
    """Execute the code review workflow.

    Args:
        code_snippet: The code to review
        verbose: Print progress info

    Returns:
        Final deliverable dict with 'report' key
    """
    # Create separate model instances for parallel execution
    # Claude Code CLI can't handle concurrent calls on same instance
    sonnet_analysis = iModel(
        provider="claude_code",
        model="sonnet",
        name="sonnet_analysis",
        verbose=True,
    )
    sonnet_security = iModel(
        provider="claude_code",
        model="sonnet",
        name="sonnet_security",
        verbose=True,
    )
    sonnet_report = iModel(
        provider="claude_code",
        model="sonnet",
        name="sonnet_report",
        verbose=True,
    )

    # Create session and register all models
    session = Session(default_generate_model=sonnet_analysis)
    session.services.register(sonnet_security)
    session.services.register(sonnet_report)

    # Create branch with access to all models
    # Capabilities match Report class attribute names (not lowercase class names)
    branch = session.create_branch(
        name="review",
        resources={"sonnet_analysis", "sonnet_security", "sonnet_report"},
        capabilities={"analysis", "security", "report"},
    )

    # Create and initialize the report with instruction
    report = CodeReviewReport()
    report.initialize(
        instruction="Review this code for quality, patterns, and security vulnerabilities",
        code_snippet=code_snippet,
    )

    if verbose:
        print(f"Initialized report: {report}")
        print(f"Input fields: {report.input_fields}")
        print(f"Output fields: {report.output_fields}")
        print(f"Forms: {len(report.forms)}")

    # Execute via flow_report
    # This compiles forms to a DAG and executes with dependency resolution
    result = await flow_report(
        session=session,
        report=report,
        branch=branch,
        verbose=verbose,
    )

    return result


# ============================================================================
# Main - Demo with sample code
# ============================================================================


SAMPLE_CODE = '''
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user credentials against database."""
    import hashlib
    from database import get_user

    user = get_user(username)
    if not user:
        return False

    # Hash password and compare
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return user.password_hash == hashed


class UserService:
    """Service for user management operations."""

    def __init__(self, db_connection):
        self.db = db_connection
        self._cache = {}

    async def get_user_profile(self, user_id: int) -> dict:
        """Fetch user profile with caching."""
        if user_id in self._cache:
            return self._cache[user_id]

        query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL query
        result = await self.db.execute(query)
        self._cache[user_id] = result
        return result

    def update_password(self, user_id: int, new_password: str) -> None:
        """Update user password."""
        import hashlib
        hashed = hashlib.sha256(new_password.encode()).hexdigest()
        self.db.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hashed, user_id)
        )
'''


async def main():
    """Run the code review example."""
    print("=" * 60)
    print("Code Review Report Example")
    print("=" * 60)
    print()
    print("Code to review:")
    print("-" * 40)
    print(SAMPLE_CODE)
    print("-" * 40)
    print()

    try:
        result = await run_code_review(SAMPLE_CODE, verbose=True)

        print()
        print("=" * 60)
        print("Final Deliverable")
        print("=" * 60)

        if result.get("report"):
            final = result["report"]
            if isinstance(final, FinalReport):
                print(f"\nExecutive Summary:\n  {final.executive_summary}")
                print("\nKey Findings:")
                for finding in final.key_findings:
                    print(f"  - {finding}")
                print("\nAction Items:")
                for item in final.action_items:
                    print(f"  - {item}")
                print(f"\nOverall Quality: {final.overall_quality}")
            else:
                print(f"Result: {final}")
        else:
            print(f"Raw result: {result}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
