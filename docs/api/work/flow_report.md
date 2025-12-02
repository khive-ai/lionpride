# flow_report

> Execute a Report workflow with automatic scheduling and parallel form execution

## Overview

`flow_report` orchestrates Report workflows, automatically determining execution order from field dependencies and executing independent forms in parallel.

**Capabilities:** Automatic scheduling via `next_forms()` | Parallel execution with optional `max_concurrent` limit | Deadlock detection | Branch resolution via DSL prefix

**Use for:** Multi-step workflows, DAG pipelines, parallel analyses. **Not for:** Single-step operations (use `operate`), simple chat (use `Session.conduct`).

## Function Signature

```python
from lionpride.work import flow_report, Report
from lionpride.session import Session, Branch

async def flow_report(
    session: Session,
    report: Report,
    *,
    branch: Branch | str | None = None,
    max_concurrent: int | None = None,
    reason: bool = False,
    actions: bool = False,
    verbose: bool = False,
) -> dict[str, Any]: ...
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `Session` | required | Session with services and branches |
| `report` | `Report` | required | Initialized Report with `form_assignments` |
| `branch` | `Branch \| str \| None` | `None` | Default branch (forms can override via DSL prefix) |
| `max_concurrent` | `int \| None` | `None` | Max parallel forms (None = unlimited) |
| `reason` | `bool` | `False` | Include reasoning in outputs |
| `actions` | `bool` | `False` | Enable tool use |
| `verbose` | `bool` | `False` | Print execution progress |

## Returns

**dict[str, Any]**

The final deliverable dictionary containing all output field values.

- Keys are the output field names from `report.output_fields`
- Values are the computed results (Pydantic models or primitives)
- Equivalent to calling `report.get_deliverable()` after completion

## Raises

### RuntimeError

Raised when deadlock is detected:

- No forms are ready to execute (`report.next_forms()` returns empty)
- But pending forms exist (workflow not complete)
- Indicates missing inputs or circular dependencies

### NotFoundError

Raised if:

- Branch name not found in session (via `session.get_branch()`)

## Execution Model

The `flow_report` function implements a **dependency-driven scheduling loop**:

```text
START
  |
  v
+------------------------+
| while not complete()   |<------------------+
+------------------------+                   |
  |                                          |
  v                                          |
+------------------------+                   |
| ready = next_forms()   |                   |
+------------------------+                   |
  |                                          |
  +--[empty?]--+                             |
  |            |                             |
  | yes        | no                          |
  v            v                             |
+--------+   +------------------------+      |
|DEADLOCK|   | len(ready) == 1?       |      |
| ERROR  |   +------------------------+      |
+--------+     |            |                |
               | yes        | no             |
               v            v                |
         +---------+   +-----------+         |
         | execute |   | parallel  |         |
         | single  |   | execute   |         |
         +---------+   | (semaphore)|        |
               |       +-----------+         |
               v            |                |
         +---------+        |                |
         | fill()  |<-------+                |
         | complete|                         |
         +---------+                         |
               |                             |
               +-----------------------------+
  |
  v
+------------------------+
| return deliverable()   |
+------------------------+
```

### Scheduling Algorithm

1. **Loop**: Continue until `report.is_complete()` returns `True`
2. **Get Ready Forms**: Call `report.next_forms()` to find forms with all inputs available
3. **Deadlock Check**: If no ready forms but pending forms exist, raise `RuntimeError`
4. **Execute**:
   - Single form: Execute directly via `operate()`
   - Multiple forms: Execute in parallel (with optional semaphore limit)
5. **Update State**: For each result:
   - `form.fill(output)` - Store result in form
   - `report.complete_form(form)` - Update `available_data`
6. **Repeat**: Loop to step 2

### Parallel Execution

When multiple forms are ready simultaneously:

```python
# Without max_concurrent (unlimited parallelism)
tasks = [execute_one(f) for f in ready_forms]
results = await gather(*tasks)

# With max_concurrent (semaphore-limited)
sem = Semaphore(max_concurrent)
async def limited_execute(form):
    async with sem:
        return await execute_one(form)
tasks = [limited_execute(f) for f in ready_forms]
results = await gather(*tasks)
```

## Usage Patterns

### Sequential Workflow

```python
from lionpride import Session, iModel
from lionpride.work import Report, flow_report
from pydantic import BaseModel

class Analysis(BaseModel):
    """Analyze the topic."""
    summary: str
    key_points: list[str]

class Conclusion(BaseModel):
    """Synthesize into conclusion."""
    recommendation: str

class SequentialReport(Report):
    analysis: Analysis | None = None
    conclusion: Conclusion | None = None

    assignment = "topic -> conclusion"
    form_assignments = [
        "topic -> analysis",
        "analysis -> conclusion",
    ]

model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)
branch = session.create_branch(
    capabilities={"analysis", "conclusion"},
    resources={model.name},
)

report = SequentialReport()
report.initialize(topic="AI assistants")
result = await flow_report(session, report, branch=branch)
```

### Parallel Execution

```python
class ParallelReport(Report):
    tech: TechAnalysis | None = None
    market: MarketAnalysis | None = None
    final: FinalReport | None = None

    assignment = "company -> final"
    form_assignments = [
        "company -> tech",        # Parallel
        "company -> market",      # Parallel
        "tech, market -> final",  # Waits for both
    ]

report = ParallelReport()
report.initialize(company="Acme Corp")
result = await flow_report(session, report, branch=branch, max_concurrent=2)
```

## Common Pitfalls

- **Uninitialized report**: Always call `report.initialize(**inputs)` before `flow_report()`
- **Circular dependencies**: Ensure DAG structure - `"a -> b"` and `"b -> a"` causes deadlock
- **Missing output schema**: Declare all outputs as class attributes: `analysis: Analysis | None = None`

## Design Rationale

**Dependency-Based Scheduling**: Users declare data dependencies, not execution order. Independent forms parallelize automatically.

**Loop Until Complete**: Handles any DAG without explicit topological sort; clear iteration boundaries for debugging.

**Deadlock Detection**: Fail fast on misconfiguration rather than silent hang.

## See Also

- [Report](report.md): Workflow definition with class-attribute schemas
- [Form](form.md): Individual form contracts with assignment DSL
- [operate](../operations/operate.md): Single-step structured output
- [Session](../session/session.md): Service and branch management
