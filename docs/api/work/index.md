# Work

> Declarative workflow orchestration with dependency-aware execution

## Overview

The `work` module provides declarative workflow orchestration for complex multi-step LLM tasks. It uses a **Form + Report** pattern where Forms define units of work and Reports orchestrate workflows with automatic dependency resolution.

**Core Concepts:**

- **Form**: Unit of work with assignment DSL defining inputs and outputs
- **Report**: Workflow orchestrator - subclass to define output schemas as class attributes
- **FormResources**: Capability declarations for workflow steps (API models, tools)
- **flow_report**: Executes report via compiled graph with parallel-capable execution

**Assignment DSL:**

```text
"[branch:] [operation(] inputs -> outputs [)] [| resources]"
```

The work module enables building complex multi-agent workflows without manual dependency management. Dependencies are inferred from field dataflow, and independent forms execute in parallel automatically.

## Classes

### Core

| Class | Description |
|-------|-------------|
| `Form` | Unit of work with assignment DSL |
| `Report` | Workflow orchestrator base class |
| `flow_report` | Execute report via compiled dependency graph |
| `parse_assignment` | Parse assignment DSL strings |

### Capabilities

| Class | Description |
|-------|-------------|
| `FormResources` | Capability declarations for workflow steps |
| `ParsedAssignment` | Parsed assignment structure |

### Errors

| Class | Description |
|-------|-------------|
| `CapabilityError` | Missing required capability |
| `AmbiguousResourceError` | Ambiguous resource specification |

## Quick Start

```python
from lionpride import Session, iModel
from lionpride.work import Report, Form, flow_report
from pydantic import BaseModel

# Define output schemas
class Analysis(BaseModel):
    summary: str
    key_points: list[str]

class Plan(BaseModel):
    steps: list[str]
    timeline: str

# Define workflow as Report subclass
class ResearchReport(Report):
    # Output schemas as class attributes
    analysis: Analysis | None = None
    plan: Plan | None = None

    # Main workflow assignment
    assignment = "topic -> plan"

    # Form assignments define the workflow steps
    form_assignments = [
        "topic -> analysis",           # Step 1: Analyze topic
        "analysis -> plan",            # Step 2: Create plan from analysis
    ]

# Execute workflow
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)
branch = session.create_branch(
    capabilities={"analysis", "plan"},
    resources={model.name},
)

report = ResearchReport()
report.initialize(topic="Building AI agents with lionpride")

result = await flow_report(session, report, branch=branch)
print(result.analysis.summary)
print(result.plan.steps)
```

## Assignment DSL

### Basic Format

```python
# Simple: inputs -> outputs
"context -> plan"

# Multiple inputs/outputs
"a, b -> c, d"

# With branch name prefix
"orchestrator: context -> plan"
```

### With Operations

```python
# Specify operation type
"operate(context -> plan)"
"react(analysis -> comprehensive_plan)"
"communicate(data -> summary)"
```

### With Resources

```python
# Specify API model
"context -> analysis | api:gpt4mini"

# Multiple resources
"react(a, b -> c) | api_gen:gpt4o, api_parse:gpt4mini, tool:*"

# All tools available
"research(topic -> findings) | tool:*"
```

## Advanced Example

```python
from lionpride.work import Report, flow_report
from pydantic import BaseModel, Field

class Research(BaseModel):
    findings: list[str]
    sources: list[str]

class Analysis(BaseModel):
    insights: list[str]
    confidence: float = Field(ge=0, le=1)

class Recommendation(BaseModel):
    action: str
    reasoning: str
    priority: str

class ComprehensiveReport(Report):
    # Output schemas
    research: Research | None = None
    analysis: Analysis | None = None
    recommendation: Recommendation | None = None

    # Main assignment
    assignment = "topic -> recommendation"

    # Workflow with capabilities
    form_assignments = [
        # Research with tool access
        "researcher: react(topic -> research) | api_gen:gpt4o, tool:search,web_browse",

        # Analysis uses research output
        "analyst: operate(research -> analysis) | api:gpt4mini",

        # Recommendation uses both
        "advisor: operate(research, analysis -> recommendation) | api:gpt4o",
    ]

# Dependency graph automatically resolved:
# topic -> research -> analysis -> recommendation
#                  \-> recommendation (parallel with analysis -> recommendation)
```

## See Also

- [Operations](../operations/index.md) - Operations used within forms
- [Session](../session/index.md) - Session executes workflow
- [Rules](../rules/index.md) - Output validation
- [User Guide: Workflows](../../user_guide/workflows.md) - Tutorial and patterns
