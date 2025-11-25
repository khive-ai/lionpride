# OperationGraphBuilder

> Fluent builder for operation DAGs (directed acyclic graphs)

## Overview

The `OperationGraphBuilder` class provides a fluent interface for constructing operation workflows as directed acyclic graphs (DAGs). It separates graph structure definition from execution, enabling graph reuse, serialization, and analysis before execution.

**Key Capabilities:**

- **Fluent API**: Chainable methods for readable graph construction
- **Dependency Management**: Explicit and implicit dependency linking
- **DAG Validation**: Automatic cycle detection
- **Aggregation Support**: Collect results from multiple operations
- **Incremental Building**: Add operations and mark executed for multi-phase workflows
- **Reusability**: Build once, execute with different sessions/branches

## Signature

```python
class OperationGraphBuilder:
    def __init__(self, graph: Graph | None = None): ...
```

## Parameters

- `graph` (Graph, optional): Existing graph to extend (default: creates new Graph)

## Core Methods

### `add()`

Add operation to graph with optional dependencies.

```python
def add(
    self,
    name: str,
    operation: OperationType | str,
    parameters: dict[str, Any] | BaseModel | None = None,
    depends_on: list[str] | None = None,
    inherit_context: bool = False,
    **kwargs,
) -> OperationGraphBuilder: ...
```

**Parameters**:

- `name` (str): Unique operation name
- `operation` (OperationType | str): Operation type ("communicate", "operate", etc.)
- `parameters` (dict | BaseModel, optional): Operation parameters
- `depends_on` (list[str], optional): Names of operations this depends on
- `inherit_context` (bool): Inherit context from primary dependency
- `**kwargs`: Additional parameters (merged with `parameters`)

**Returns**: `self` for chaining

**Example**:

```python
builder = OperationGraphBuilder()
builder.add(
    "extract",
    "communicate",
    instruction="Extract data",
    imodel="gpt-4o",
)
```

### `depends_on()`

Add dependency relationships between operations.

```python
def depends_on(
    self,
    target: str,
    *dependencies: str,
    label: list[str] | None = None,
) -> OperationGraphBuilder: ...
```

**Parameters**:

- `target` (str): Target operation name
- `*dependencies` (str): Dependency operation names
- `label` (list[str], optional): Edge labels

**Returns**: `self` for chaining

**Example**:

```python
builder.depends_on("summarize", "extract", "analyze")
# summarize depends on both extract and analyze
```

### `sequence()`

Create sequential dependency chain.

```python
def sequence(
    self,
    *operations: str,
    label: list[str] | None = None,
) -> OperationGraphBuilder: ...
```

**Parameters**:

- `*operations` (str): Operation names in order
- `label` (list[str], optional): Edge labels

**Returns**: `self` for chaining

**Example**:

```python
builder.sequence("fetch", "process", "store")
# Creates: fetch → process → store
```

### `parallel()`

Mark operations as parallel (no-op for clarity).

```python
def parallel(self, *operations: str) -> OperationGraphBuilder: ...
```

**Parameters**:

- `*operations` (str): Operation names to mark as parallel

**Returns**: `self` for chaining

**Note**: This is a no-op for documentation. Operations are naturally parallel if no dependencies exist.

**Example**:

```python
builder.parallel("analyze_sentiment", "extract_entities", "classify")
# No edges added - operations can run concurrently
```

### `add_aggregation()`

Add aggregation operation that collects from multiple sources.

```python
def add_aggregation(
    self,
    name: str,
    operation: OperationType | str,
    parameters: dict[str, Any] | BaseModel | None = None,
    source_names: list[str] | None = None,
    inherit_context: bool = False,
    inherit_from_source: int = 0,
    **kwargs,
) -> OperationGraphBuilder: ...
```

**Parameters**:

- `name` (str): Aggregation operation name
- `operation` (OperationType | str): Operation type
- `parameters` (dict | BaseModel, optional): Operation parameters
- `source_names` (list[str], optional): Source operations (default: current heads)
- `inherit_context` (bool): Inherit context from a source
- `inherit_from_source` (int): Index of source to inherit from
- `**kwargs`: Additional parameters

**Returns**: `self` for chaining

**Example**:

```python
builder.add("task1", "communicate", instruction="Task 1", imodel="gpt-4o")
builder.add("task2", "communicate", instruction="Task 2", imodel="gpt-4o")
builder.add_aggregation(
    "combine",
    "communicate",
    instruction="Combine results",
    imodel="gpt-4o",
    source_names=["task1", "task2"],
)
# combine depends on both task1 and task2
```

### `build()`

Build and validate operation graph.

```python
def build(self) -> Graph: ...
```

**Returns**: Validated Graph (DAG)

**Raises**: `ValueError` if graph has cycles

**Example**:

```python
graph = builder.build()
# Validates DAG and returns graph
```

### `clear()`

Clear all operations and start fresh.

```python
def clear(self) -> OperationGraphBuilder: ...
```

**Returns**: `self` for chaining

## Usage Patterns

### Simple Sequential Workflow

```python
from lionpride.operations import Builder

builder = Builder()  # Alias for OperationGraphBuilder

builder.add(
    "fetch",
    "communicate",
    instruction="Fetch data from API",
    imodel="gpt-4o",
)

builder.add(
    "process",
    "operate",
    instruction="Process and validate",
    imodel="gpt-4o",
    response_model=DataModel,
    depends_on=["fetch"],
)

builder.add(
    "summarize",
    "communicate",
    instruction="Summarize results",
    imodel="gpt-4o",
    depends_on=["process"],
)

graph = builder.build()
```

### Parallel Operations with Aggregation

```python
builder = Builder()

# Parallel analysis tasks
builder.add("sentiment", "operate", instruction="Analyze sentiment", imodel="gpt-4o")
builder.add("entities", "operate", instruction="Extract entities", imodel="gpt-4o")
builder.add("topics", "operate", instruction="Classify topics", imodel="gpt-4o")

# Mark as parallel (optional, for documentation)
builder.parallel("sentiment", "entities", "topics")

# Aggregate results
builder.add_aggregation(
    "combine",
    "communicate",
    instruction="Combine analysis results",
    imodel="gpt-4o",
    source_names=["sentiment", "entities", "topics"],
)

graph = builder.build()
```

### Complex Dependency Graph

```python
builder = Builder()

# Layer 1: Data sources
builder.add("source1", "communicate", instruction="Fetch source 1", imodel="gpt-4o")
builder.add("source2", "communicate", instruction="Fetch source 2", imodel="gpt-4o")

# Layer 2: Processing (depends on sources)
builder.add("process1", "operate",
    instruction="Process data",
    imodel="gpt-4o",
    response_model=Model1,
    depends_on=["source1"]
)

builder.add("process2", "operate",
    instruction="Process data",
    imodel="gpt-4o",
    response_model=Model2,
    depends_on=["source2"]
)

# Layer 3: Cross-analysis (depends on both processors)
builder.add("analyze", "react",
    instruction="Cross-analyze results",
    imodel="gpt-4o",
    tools=[AnalysisTool],
    model_name="gpt-4o",
    depends_on=["process1", "process2"]
)

# Layer 4: Final report
builder.add("report", "communicate",
    instruction="Generate final report",
    imodel="gpt-4o",
    depends_on=["analyze"]
)

graph = builder.build()
```

### Conditional Workflows

```python
builder = Builder()

# Initial analysis
builder.add("check", "operate",
    instruction="Check data quality",
    imodel="gpt-4o",
    response_model=QualityCheck,
)

# Path A: High quality data
builder.add("fast_process", "communicate",
    instruction="Quick processing",
    imodel="gpt-4o",
    depends_on=["check"],
)

# Path B: Low quality data
builder.add("deep_process", "react",
    instruction="Deep cleaning and processing",
    imodel="gpt-4o",
    tools=[CleaningTool],
    model_name="gpt-4o",
    depends_on=["check"],
)

# Note: Both paths execute; runtime logic determines which result to use
graph = builder.build()
```

### Using Sequence Helper

```python
builder = Builder()

# Add operations
builder.add("step1", "communicate", instruction="Step 1", imodel="gpt-4o")
builder.add("step2", "communicate", instruction="Step 2", imodel="gpt-4o")
builder.add("step3", "communicate", instruction="Step 3", imodel="gpt-4o")

# Create sequence: step1 → step2 → step3
builder.sequence("step1", "step2", "step3")

graph = builder.build()
```

### Incremental Building

```python
builder = Builder()

# Phase 1: Initial operations
builder.add("init", "communicate", instruction="Initialize", imodel="gpt-4o")
graph1 = builder.build()

# Execute phase 1
results1 = await flow(session, "main", graph1, ipu)

# Mark as executed
builder.mark_executed("init")

# Phase 2: Add dependent operations
builder.add("next", "communicate",
    instruction="Continue from init",
    imodel="gpt-4o",
    depends_on=["init"]
)
graph2 = builder.build()

# Execute phase 2 (only "next" runs, "init" already executed)
results2 = await flow(session, "main", graph2, ipu)
```

## Advanced Features

### Context Inheritance

```python
builder = Builder()

builder.add("fetch", "communicate",
    instruction="Fetch data",
    imodel="gpt-4o",
)

builder.add("process", "operate",
    instruction="Process using fetched data",
    imodel="gpt-4o",
    response_model=ProcessedData,
    depends_on=["fetch"],
    inherit_context=True,  # Inherit context from "fetch"
)

# process operation receives "fetch_result" in context
```

### Custom Parameters

```python
from pydantic import BaseModel

class CustomParams(BaseModel):
    instruction: str
    imodel: str
    temperature: float
    max_tokens: int

params = CustomParams(
    instruction="Generate text",
    imodel="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
)

builder.add("gen", "generate", parameters=params)
```

### Graph Inspection

```python
builder = Builder()
builder.add("op1", "communicate", instruction="Op 1", imodel="gpt-4o")
builder.add("op2", "communicate", instruction="Op 2", imodel="gpt-4o")

# Get operation node
node = builder.get("op1")
print(node.metadata["name"])  # "op1"
print(node.metadata["operation_type"])  # "communicate"

# Get by UUID
node_by_id = builder.get_by_id(node.id)

# Check unexecuted
unexecuted = builder.get_unexecuted_nodes()
print(len(unexecuted))  # 2
```

## Common Pitfalls

- **Duplicate names**: Adding operations with same name
  - **Solution**: Use unique names for each operation

- **Missing dependencies**: Referencing non-existent operation in `depends_on`
  - **Solution**: Ensure dependency operations are added before referencing

- **Circular dependencies**: Creating cycles in the graph
  - **Solution**: `build()` validates DAG and raises ValueError on cycles

- **Parameter conflicts**: Providing both `parameters` and `**kwargs` with overlapping keys
  - **Solution**: `**kwargs` override `parameters` - use one or the other

- **Execution order confusion**: Expecting specific execution order for parallel operations
  - **Solution**: Only operations with dependencies have guaranteed order

## Design Rationale

### Why Separate Builder from Executor?

Separating graph building from execution enables:

- **Reusability**: Build once, execute with different sessions
- **Serialization**: Store graph definitions
- **Testing**: Validate graph structure without execution
- **Optimization**: Analyze dependencies before execution

### Why Fluent Interface?

Method chaining provides:

- **Readability**: Clear workflow structure in code
- **Convenience**: Less verbose than separate statements
- **Consistency**: Standard pattern across the framework

### Why Node Metadata?

Storing operation specs in node metadata allows:

- **Late Binding**: Create Operations at execution time with session context
- **Flexibility**: Same graph with different sessions/branches
- **Serialization**: Graph can be serialized without session references

## See Also

- **Related Components**:
  - [`flow`](flow.md): Execute built graphs
  - [`DependencyAwareExecutor`](flow.md#dependencyawareexecutor): Graph executor
  - [Graph](../base/graph.md): Underlying graph structure

- **Related Operations**:
  - [`communicate`](communicate.md): Chat operation
  - [`operate`](operate.md): Structured operation
  - [`react`](react.md): ReAct operation

- **User Guide**:
  - Workflow patterns (documentation pending)
  - Graph optimization (documentation pending)

## Examples

### Complete Example: Content Pipeline

```python
from lionpride.operations import Builder
from pydantic import BaseModel

# Define models
class Article(BaseModel):
    title: str
    content: str
    url: str

class Summary(BaseModel):
    key_points: list[str]
    sentiment: str

class Report(BaseModel):
    articles: list[Article]
    summaries: list[Summary]
    overall_sentiment: str

# Build pipeline
builder = Builder()

# Fetch articles from multiple sources
builder.add("fetch_tech", "communicate",
    instruction="Fetch tech articles",
    imodel="gpt-4o",
)

builder.add("fetch_business", "communicate",
    instruction="Fetch business articles",
    imodel="gpt-4o",
)

# Process articles (parallel)
builder.add("summarize_tech", "operate",
    instruction="Summarize tech articles",
    imodel="gpt-4o",
    response_model=Summary,
    depends_on=["fetch_tech"],
)

builder.add("summarize_business", "operate",
    instruction="Summarize business articles",
    imodel="gpt-4o",
    response_model=Summary,
    depends_on=["fetch_business"],
)

# Aggregate and generate final report
builder.add_aggregation("final_report", "operate",
    instruction="Generate combined report",
    imodel="gpt-4o",
    response_model=Report,
    source_names=["summarize_tech", "summarize_business"],
)

# Build and execute
graph = builder.build()

# Execute
from lionpride.operations import flow

results = await flow(
    session,
    "main",
    graph,
    ipu,
    max_concurrent=3,
    verbose=True,
)

# Access results
final_report = results["final_report"]
print(final_report.overall_sentiment)
```
