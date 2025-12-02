# Form

> Declarative unit of work with assignment-based field contracts.

## Overview

`Form` declares data dependencies through an assignment DSL, enabling automatic dependency resolution and parallel execution. Forms are **pure data contracts** - they specify inputs/outputs while schemas live on the parent `Report`. Execution order derives from field dependencies, not explicit edges.

## Class Signature

```python
class Form(Element):
    """Declarative unit of work with assignment-based field contracts."""

    def __init__(
        self,
        assignment: str,
        *,
        branch_name: str | None = None,
        operation: str = "operate",
        input_fields: list[str] | None = None,
        output_fields: list[str] | None = None,
        resources: FormResources | None = None,
        output: Any = None,
        filled: bool = False,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assignment` | `str` | **required** | Assignment DSL string specifying dataflow and resources |
| `branch_name` | `str or None` | `None` | Derived from assignment; optional branch prefix |
| `operation` | `str` | `"operate"` | Derived from assignment; operation type |
| `input_fields` | `list[str]` | `[]` | Derived from assignment; input field names |
| `output_fields` | `list[str]` | `[]` | Derived from assignment; output field names |
| `resources` | `FormResources` | `FormResources()` | Derived from assignment; resource declarations |
| `output` | `Any` | `None` | Runtime: populated after execution |
| `filled` | `bool` | `False` | Runtime: True after execution completes |

Note: While `branch_name`, `operation`, `input_fields`, `output_fields`, and `resources` can be provided explicitly, they are typically derived automatically from the `assignment` string via `model_post_init()`.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Unique identifier (inherited from Element, auto-generated, frozen) |
| `created_at` | `datetime` | UTC timestamp (inherited from Element, auto-generated, frozen) |
| `metadata` | `dict[str, Any]` | Arbitrary metadata (inherited from Element) |
| `assignment` | `str` | Full DSL string with operation, dataflow, and resources |
| `branch_name` | `str or None` | Optional branch prefix from DSL (e.g., `"orchestrator"`) |
| `operation` | `str` | Operation type: `"generate"`, `"parse"`, `"communicate"`, `"operate"`, or `"react"` |
| `input_fields` | `list[str]` | Field names required as input |
| `output_fields` | `list[str]` | Field names produced as output |
| `resources` | `FormResources` | Parsed resource declarations (APIs, tools) |
| `output` | `Any` | Structured output after execution |
| `filled` | `bool` | Whether this form has been executed |

## Assignment DSL

```text
[branch:] [operation(] inputs -> outputs [)] [| resources]
```

**Examples:**

- `"topic -> analysis"` - Simple dataflow
- `"planner: react(context -> plan) | api:gpt4o, tool:*"` - Full specification

**Resource types:** `api` (default) | `api_gen` | `api_parse` | `api_interpret` | `tool` (or `tool:*` for all)

## Methods

### Workability Check

#### `is_workable()`

Check if all input fields are available and form is not yet filled.

**Signature:**

```python
def is_workable(self, available_data: dict[str, Any]) -> bool: ...
```

**Parameters:**

- `available_data` (dict[str, Any]): Currently available field values

**Returns:**

- `bool`: True if all inputs are available (not None) and form is not yet filled

**Examples:**

```python
>>> form = Form(assignment="a, b -> c")
>>> form.is_workable({"a": 1, "b": 2})
True
>>> form.is_workable({"a": 1})  # missing 'b'
False
>>> form.is_workable({"a": 1, "b": None})  # 'b' is None
False
>>> form.fill(output="result")
>>> form.is_workable({"a": 1, "b": 2})  # already filled
False
```

### Input Extraction

#### `get_inputs()`

Extract input data for this form from available data.

**Signature:**

```python
def get_inputs(self, available_data: dict[str, Any]) -> dict[str, Any]: ...
```

**Parameters:**

- `available_data` (dict[str, Any]): All available data

**Returns:**

- `dict[str, Any]`: Dictionary of input field values that exist in available_data

**Examples:**

```python
>>> form = Form(assignment="a, b -> c")
>>> form.get_inputs({"a": 1, "b": 2, "extra": 3})
{'a': 1, 'b': 2}
>>> form.get_inputs({"a": 1})  # partial data
{'a': 1}
```

### Completion

#### `fill()`

Mark form as filled with output.

**Signature:**

```python
def fill(self, output: Any) -> None: ...
```

**Parameters:**

- `output` (Any): The structured output from execution

**Examples:**

```python
>>> form = Form(assignment="topic -> analysis")
>>> form.filled
False
>>> form.fill(analysis_result)
>>> form.filled
True
>>> form.output
<analysis_result>
```

### Output Extraction

#### `get_output_data()`

Extract output field values from the output.

**Signature:**

```python
def get_output_data(self) -> dict[str, Any]: ...
```

**Returns:**

- `dict[str, Any]`: Dictionary mapping output field names to values

**Examples:**

```python
>>> from pydantic import BaseModel
>>> class Analysis(BaseModel):
...     summary: str
...     score: float
>>> form = Form(assignment="topic -> summary, score")
>>> form.fill(Analysis(summary="Great", score=0.95))
>>> form.get_output_data()
{'summary': 'Great', 'score': 0.95}
```

### Resource Validation

#### `validate_resources()`

Validate form resources are a subset of branch resources.

**Signature:**

```python
def validate_resources(self, branch: Branch) -> None: ...
```

**Parameters:**

- `branch` (Branch): Branch to validate against

**Raises:**

- `CapabilityError`: If any resource is not in branch

**Examples:**

```python
>>> form = Form(assignment="a -> b | api:gpt4o")
>>> form.validate_resources(branch)  # raises if gpt4o not in branch.resources
```

## FormResources

The `FormResources` dataclass holds parsed resource declarations:

```python
@dataclass(frozen=True)
class FormResources:
    api: str | None = None           # Default model for all roles
    api_gen: str | None = None       # Model for generation
    api_parse: str | None = None     # Model for parsing
    api_interpret: str | None = None # Model for interpretation
    tools: frozenset[str] | Literal["*"] | None = None
```

**Key Methods:**

- `get_gen_model()` - Returns `api_gen` or falls back to `api`
- `get_parse_model()` - Returns `api_parse` or falls back to `api`
- `get_interpret_model()` - Returns `api_interpret` or falls back to `api`
- `resolve_gen_model(branch)` - Resolve against branch with validation
- `resolve_parse_model(branch)` - Resolve against branch with validation
- `resolve_tools(branch)` - Resolve tool set against branch

## Protocol Implementations

Form inherits from `Element` and implements:

- **Observable**: `id` property (UUID identifier for object identity)
- **Serializable**: `to_dict()`, `to_json()` for serialization
- **Deserializable**: `from_dict()`, `from_json()` for deserialization
- **Hashable**: `__hash__()` based on ID (identity-based hashing)

## Usage Patterns

### Basic Usage

```python
from lionpride.work import Form

form = Form(assignment="topic -> analysis")
form.input_fields   # ['topic']
form.output_fields  # ['analysis']
form.operation      # 'operate'

# Check and execute
if form.is_workable({"topic": "AI agents"}):
    inputs = form.get_inputs({"topic": "AI agents"})
    form.fill(result)
```

### With Resources

```python
form = Form(assignment="planner: react(context -> plan) | api_gen:gpt4o, tool:search")
form.branch_name           # 'planner'
form.resources.api_gen     # 'gpt4o'
form.resources.tools       # frozenset({'search'})
```

## Common Pitfalls

- **Double fill**: Check `form.filled` or use `is_workable()` before execution
- **None as missing**: `is_workable()` treats `None` as unavailable; use sentinel values if needed

## Design Rationale

**Separation of Concerns**: Forms declare dataflow; schemas live on Report. Enables reusability and testability.

**DSL over Graphs**: `"a -> b"` is clearer than edge objects; workflow engine infers parallelization automatically.

## See Also

- [`Report`](report.md): Workflow orchestrator that contains Forms
- [`flow_report`](flow_report.md): Executes Report workflows
- [`Session`](../session/session.md): Session that manages branches and services
