# Report

> Declarative workflow orchestrator that schedules forms based on field availability.

## Overview

`Report` defines workflows through class attributes with type annotations. The system introspects annotations to derive field types, schedule form execution based on dependencies, and validate outputs against Pydantic models.

**Key principles:** Class attributes = output schemas | Model docstrings = LLM instructions | Dependencies inferred from dataflow | Parallel execution automatic

Inherits from `Element` (UUID identity, timestamps, serialization).

## Class Signature

```python
class Report(Element):
    """Workflow orchestrator - schedules forms based on field availability."""

    # Declaration (class attributes)
    assignment: str = Field(
        default="",
        description="Overall workflow: 'inputs -> final_outputs'",
    )
    form_assignments: list[str] = Field(
        default_factory=list,
        description="List of form assignments: ['a,b->c', 'c->d', ...]",
    )
    instruction: str = Field(
        default="",
        description="Overall workflow goal/instruction for LLM context",
    )

    # Derived
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)

    # Runtime state
    forms: Pile[Form] = Field(...)
    completed_forms: Pile[Form] = Field(...)
    available_data: dict[str, Any] = Field(default_factory=dict)
```

## Parameters

Report is designed to be subclassed, not instantiated directly with parameters. Define workflows through class attributes:

| Class Attribute | Type | Description |
|-----------------|------|-------------|
| `assignment` | `str` | Overall workflow contract: `"inputs -> final_outputs"` |
| `form_assignments` | `list[str]` | Form DSL strings defining individual steps |
| `instruction` | `str` | Overall workflow goal (passed to LLM for context) |
| Output schemas | `Type \| None` | Class attributes with type annotations become output fields |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Unique identifier (inherited from Element) |
| `created_at` | `datetime` | UTC timestamp (inherited from Element) |
| `metadata` | `dict[str, Any]` | Arbitrary metadata (inherited from Element) |
| `input_fields` | `list[str]` | Derived from `assignment` (before `->`) |
| `output_fields` | `list[str]` | Derived from `assignment` (after `->`) |
| `forms` | `Pile[Form]` | All forms created from `form_assignments` |
| `completed_forms` | `Pile[Form]` | Forms that have been executed |
| `available_data` | `dict[str, Any]` | Current state of all field values |
| `progress` | `tuple[int, int]` | Property: `(completed_count, total_count)` |

## Methods

### Initialization

#### `initialize()`

Provide initial input data and optional workflow instruction.

**Signature:**

```python
def initialize(self, *, instruction: str | None = None, **inputs: Any) -> None: ...
```

**Parameters:**

- `instruction` (str, optional): Overall workflow goal (passed to LLM for context)
- `**inputs`: Initial field values matching `input_fields`

**Raises:**

- `ValueError`: If required input field is missing

**Examples:**

```python
>>> report = MyReport()
>>> report.initialize(topic="AI coding assistants")
>>> report.available_data
{'topic': 'AI coding assistants'}
```

### Scheduling

#### `next_forms()`

Get forms that are ready to execute (all inputs available).

**Signature:**

```python
def next_forms(self) -> list[Form]: ...
```

**Returns:**

- `list[Form]`: Forms with all inputs available and not yet filled

**Examples:**

```python
>>> ready = report.next_forms()
>>> [f.output_fields[0] for f in ready]
['analysis']  # First form ready
```

#### `complete_form()`

Mark a form as completed and update available data.

**Signature:**

```python
def complete_form(self, form: Form) -> None: ...
```

**Parameters:**

- `form` (Form): The completed form (must have `filled=True`)

**Raises:**

- `ValueError`: If form is not filled

**Examples:**

```python
>>> form.fill(output=analysis_result)
>>> report.complete_form(form)
>>> 'analysis' in report.available_data
True
```

### Status

#### `is_complete()`

Check if all output fields are available.

**Signature:**

```python
def is_complete(self) -> bool: ...
```

**Returns:**

- `bool`: True if workflow is complete (all `output_fields` have values)

**Examples:**

```python
>>> report.is_complete()
False
>>> # ... execute all forms ...
>>> report.is_complete()
True
```

#### `progress`

Property returning progress as (completed, total).

**Signature:**

```python
@property
def progress(self) -> tuple[int, int]: ...
```

**Returns:**

- `tuple[int, int]`: `(completed_forms_count, total_forms_count)`

**Examples:**

```python
>>> report.progress
(1, 3)  # 1 of 3 forms completed
```

### Output Retrieval

#### `get_deliverable()`

Get final deliverable based on `output_fields`.

**Signature:**

```python
def get_deliverable(self) -> dict[str, Any]: ...
```

**Returns:**

- `dict[str, Any]`: Output field names mapped to their values

**Examples:**

```python
>>> report.get_deliverable()
{'insights': Insights(patterns=['...'], recommendations=['...'])}
```

#### `get_all_data()`

Get all accumulated data (inputs and intermediate outputs).

**Signature:**

```python
def get_all_data(self) -> dict[str, Any]: ...
```

**Returns:**

- `dict[str, Any]`: Complete copy of `available_data`

### Type Introspection

#### `get_field_type()`

Get the type annotation for an output field.

**Signature:**

```python
def get_field_type(self, field: str) -> type | None: ...
```

**Parameters:**

- `field` (str): Field name to look up

**Returns:**

- `type | None`: The type annotation, or None if not declared. Unwraps `Optional[X]` / `X | None` to return `X`.

**Examples:**

```python
>>> report.get_field_type('analysis')
<class 'Analysis'>
>>> report.get_field_type('score')
<class 'float'>
>>> report.get_field_type('nonexistent')
None
```

#### `get_request_model()`

Get Pydantic model for an output field (for structured output validation).

**Signature:**

```python
def get_request_model(self, field: str) -> type[BaseModel] | None: ...
```

**Parameters:**

- `field` (str): Field name to look up

**Returns:**

- `type[BaseModel] | None`: The Pydantic model class, or None if field is primitive or not declared

**Examples:**

```python
>>> report.get_request_model('analysis')
<class 'Analysis'>  # Pydantic model
>>> report.get_request_model('score')
None  # float is not a BaseModel
```

## Assignment DSL

The assignment DSL specifies data flow and resource requirements.

### Basic Syntax

```text
[branch:] [operation(] inputs -> outputs [)] [| resources]
```

### Components

| Component | Description | Example |
|-----------|-------------|---------|
| `branch:` | Optional branch prefix | `"orchestrator: a -> b"` |
| `operation()` | Operation type (default: `operate`) | `"react(a -> b)"` |
| `inputs` | Comma-separated input fields | `"topic, context"` |
| `->` | Data flow operator (required) | |
| `outputs` | Comma-separated output fields | `"analysis, score"` |
| `\| resources` | Resource declarations | `"\| api:gpt4, tool:*"` |

### Operations

| Operation | Description |
|-----------|-------------|
| `generate` | Raw LLM completion |
| `parse` | Structured data extraction |
| `communicate` | Generate + parse |
| `operate` | Full structured output (default) |
| `react` | Multi-turn reasoning loop |

### Resources

| Resource Type | Description | Example |
|---------------|-------------|---------|
| `api:` | Default model for all roles | `api:gpt4` |
| `api_gen:` | Model for generation | `api_gen:gpt4` |
| `api_parse:` | Model for parsing | `api_parse:gpt4mini` |
| `api_interpret:` | Model for react interpretation | `api_interpret:gpt4` |
| `tool:` | Tool access (use `*` for all) | `tool:search`, `tool:*` |

### Examples

```python
# Simple data flow
"topic -> analysis"

# Multiple inputs/outputs
"topic, context -> analysis, score"

# With branch prefix
"orchestrator: topic -> plan"

# With operation
"react(context -> decision)"

# Full DSL
"planner: react(topic, context -> plan) | api_gen:gpt4, api_parse:gpt4mini, tool:*"
```

## Usage Patterns

### Basic Workflow

```python
from pydantic import BaseModel
from lionpride import Session, iModel
from lionpride.work import Report, flow_report

# Output schemas - docstrings become LLM instructions
class Analysis(BaseModel):
    '''Analyze the topic and extract key insights.'''
    summary: str
    key_points: list[str]

class Insights(BaseModel):
    '''Synthesize analysis into recommendations.'''
    patterns: list[str]
    recommendations: list[str]

# Workflow via class attributes
class ResearchReport(Report):
    analysis: Analysis | None = None
    insights: Insights | None = None

    assignment = "topic -> insights"
    form_assignments = [
        "topic -> analysis",
        "analysis -> insights",
    ]

# Execute
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)
branch = session.create_branch(
    capabilities={"analysis", "insights"},
    resources={model.name},
)

report = ResearchReport()
report.initialize(topic="AI coding assistants")
result = await flow_report(session, report, branch=branch)
```

### Parallel Execution

```python
class ParallelReport(Report):
    market: MarketAnalysis | None = None
    tech: TechAnalysis | None = None
    combined: Combined | None = None

    assignment = "topic -> combined"
    form_assignments = [
        "topic -> market",              # Parallel
        "topic -> tech",                # Parallel
        "market, tech -> combined",     # Waits for both
    ]

# market and tech run concurrently; combined waits for both
result = await flow_report(session, report, branch=branch)
```

## Common Pitfalls

- **Missing type annotation**: Use `analysis: Analysis | None = None` (not bare `analysis: Analysis`)
- **No model docstring**: Add docstrings to Pydantic models - they become LLM instructions
- **Uninitialized report**: Always call `report.initialize(**inputs)` before `flow_report()`

## Design Rationale

**Class Attributes**: Type introspection via `get_type_hints()`, IDE autocomplete, self-documenting workflow specs.

**Docstrings as Instructions**: Co-located schema + instructions; same docstring serves users and LLMs.

**Inferred Dependencies**: No explicit graph construction; dependencies derived from dataflow are always accurate.

## See Also

- [`Form`](form.md) - Declarative unit of work with assignment DSL
- [`flow_report`](flow_report.md) - Workflow execution function
- [`Session`](../session/session.md) - Session for orchestrating operations
- [`operate()`](../operations/operate.md) - Core operation function
