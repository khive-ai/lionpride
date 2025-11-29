# Work System - Declarative Workflow Orchestration

The work system provides capability-based workflow orchestration through declarative DSL.

## Core Concepts

- **Form**: Unit of work with assignment DSL declaring data flow and resources
- **Report**: Workflow orchestrator - subclass to define output schemas
- **FormResources**: Capability declarations specifying what APIs/tools each step can access
- **flow_report**: Executes report via compiled graph with automatic dependency resolution

## Assignment DSL

```text
[branch:] [operation(] inputs -> outputs [)] [| resources]
```

### Components

| Component | Required | Description |
|-----------|----------|-------------|
| `branch:` | No | Named branch prefix for this step |
| `operation()` | No | Operation type (default: `operate`) |
| `inputs -> outputs` | Yes | Data flow declaration |
| `\| resources` | No | API and tool declarations |

### Operations

| Operation | Description |
|-----------|-------------|
| `generate` | Raw LLM completion |
| `parse` | Structured extraction from text |
| `communicate` | Generate + parse combined |
| `operate` | Full structured output with validation (default) |
| `react` | Multi-turn operation loop |

### Resource Types

| Type | Description | Example |
|------|-------------|---------|
| `api` | Default model for all roles | `api:gpt4o` |
| `api_gen` | Model for generation | `api_gen:gpt5` |
| `api_parse` | Model for parsing/extraction | `api_parse:gpt4mini` |
| `api_interpret` | Model for react interpretation | `api_interpret:gemini` |
| `tool` | Tool access | `tool:search`, `tool:*` |

### Resource Resolution

**APIs**: `api_gen` → `api` → branch's only model → error if ambiguous

**Tools**: Explicit tools only, or `tool:*` for all branch tools. No tools by default (safe).

## Examples

### Simple (defaults)

```python
"context -> plan"
```

- Operation: `operate` (default)
- Uses branch's only model (or error if multiple)
- No tools

### With branch prefix

```python
"orchestrator: context -> plan"
```

- Executes on branch named "orchestrator"

### With operation

```python
"react(context -> plan)"
```

- Uses react loop for multi-turn execution

### With single API

```python
"context -> plan | api:gpt4mini"
```

- Uses gpt4mini for all API roles (gen, parse, interpret)

### With role-specific APIs

```python
"orchestrator: react(context -> plan) | api_gen:gpt5, api_parse:gpt4mini"
```

- Generation: gpt5
- Parsing: gpt4mini
- Interpretation: falls back to gpt5 (api_gen)

### With tools

```python
"planner: context -> plan | api:gpt4o, tool:search, tool:calendar"
```

- Specific tools: search, calendar

```python
"executor: plan -> result | api:gpt4o, tool:*"
```

- All tools available to branch

### Full example

```python
"orchestrator: react(context, history -> plan) | api_gen:gpt5, api_parse:gpt4, api_interpret:gemini, tool:search, tool:scheduler"
```

## Usage

### Define a Report

```python
from pydantic import BaseModel
from lionpride.work import Report, flow_report

class Analysis(BaseModel):
    summary: str
    key_points: list[str]

class Plan(BaseModel):
    steps: list[str]
    timeline: str

class MyReport(Report):
    # Output schemas as class attributes
    analysis: Analysis
    plan: Plan

    # Workflow contract
    assignment = "context -> plan"

    # Form assignments with capabilities
    form_assignments = [
        "analyzer: context -> analysis | api:gpt4mini",
        "planner: react(analysis -> plan) | api_gen:gpt4o, tool:search",
    ]
```

### Execute

```python
from lionpride import Session, iModel

# Setup
model = iModel(provider="openai", model="gpt-4o-mini", name="gpt4mini")
session = Session(default_generate_model=model)
branch = session.create_branch(
    name="main",
    resources={"gpt4mini", "gpt4o", "search"},  # Available resources
)

# Run workflow
report = MyReport()
report.initialize(context="Build a REST API for user management")
result = await flow_report(session, branch, report)
```

## Capability Enforcement

Resources declared in form assignments are validated against branch resources:

```python
# Branch has: {"gpt4mini", "search"}
# Form declares: api:gpt4o  ← ERROR: gpt4o not in branch resources
```

This enforces the capability-based security model:

- Forms can only use what they declare
- Declarations must be subset of branch resources
- Branch resources must be subset of session services

## Form Lifecycle

1. **Parse**: Assignment DSL parsed to derive fields and resources
2. **Validate**: Resources validated against branch capabilities
3. **Check**: `is_workable()` checks if inputs are available
4. **Execute**: Operation runs with declared resources
5. **Fill**: Output stored, form marked complete
6. **Propagate**: Output fields become available for dependent forms

## Scheduling

Forms execute based on data dependencies, not declaration order:

```python
form_assignments = [
    "c -> d",      # Runs third (needs c)
    "a -> b",      # Runs first (has a)
    "b -> c",      # Runs second (needs b)
]
```

Independent forms run in parallel automatically.
