# flow() and DependencyAwareExecutor

> Dependency-aware execution of operation graphs with IPU validation

## Overview

The `flow()` function and `DependencyAwareExecutor` class execute operation graphs built by `OperationGraphBuilder`. They handle dependency coordination, concurrent execution, and IPU-validated operation invocation. Execution is dependency-aware, meaning operations wait for their dependencies before executing.

**Key Capabilities:**

- **Dependency Coordination**: Automatic waiting for predecessor operations
- **Concurrent Execution**: Parallel execution with configurable limits
- **IPU Validation**: All operations validated via IPU before execution
- **Context Propagation**: Predecessor results passed to dependent operations
- **Streaming Results**: Incremental result delivery via async generator
- **Error Handling**: Stop-on-error or continue modes

## flow() Function

### Signature

```python
async def flow(
    session: Session,
    branch: Branch | str,
    graph: Graph,
    ipu: IPU,
    *,
    context: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
    verbose: bool = False,
) -> dict[str, Any]
```

### Parameters

- `session` (Session): Session for operation execution
- `branch` (Branch | str): Branch for message context
- `graph` (Graph): Operation graph (DAG) to execute
- `ipu` (IPU): IPU for validated execution
- `context` (dict, optional): Shared context for all operations
- `max_concurrent` (int, optional): Max concurrent operations (default: unlimited)
- `stop_on_error` (bool): Stop on first error (default: True)
- `verbose` (bool): Print progress (default: False)

### Returns

Dictionary mapping operation names to their results.

### Raises

- `ValueError`: If graph has cycles (not a DAG)
- `RuntimeError`: If operation execution fails (when `stop_on_error=True`)

### Example

```python
from lionpride.operations import Builder, flow

# Build graph
builder = Builder()
builder.add("step1", "communicate", instruction="Step 1", imodel="gpt-4o")
builder.add("step2", "communicate", instruction="Step 2", imodel="gpt-4o", depends_on=["step1"])
graph = builder.build()

# Execute
results = await flow(
    session,
    "main",
    graph,
    ipu,
    max_concurrent=5,
    verbose=True,
)

# Access results by operation name
print(results["step1"])
print(results["step2"])
```

## flow_stream() Function

### Signature

```python
async def flow_stream(
    session: Session,
    branch: Branch | str,
    graph: Graph,
    ipu: IPU,
    *,
    context: dict[str, Any] | None = None,
    max_concurrent: int | None = None,
    stop_on_error: bool = True,
) -> AsyncGenerator[OperationResult, None]
```

### Parameters

Same as `flow()` except `verbose` (not supported in streaming mode).

### Yields

`OperationResult` objects as operations complete.

### Example

```python
from lionpride.operations import flow_stream

async for result in flow_stream(session, "main", graph, ipu):
    print(f"[{result.completed}/{result.total}] {result.name}")
    if result.success:
        print(f"  Result: {result.result}")
    else:
        print(f"  Error: {result.error}")
```

## OperationResult

Result object yielded by `flow_stream()`:

```python
@dataclass
class OperationResult:
    name: str                    # Operation name
    result: Any                  # Operation result (None if failed)
    error: Exception | None      # Exception if operation failed
    completed: int               # Number of operations completed so far
    total: int                   # Total number of operations

    @property
    def success(self) -> bool:   # Whether the operation succeeded
        return self.error is None
```

## DependencyAwareExecutor

Low-level executor class for operation graphs.

### Signature

```python
class DependencyAwareExecutor:
    def __init__(
        self,
        session: Session,
        ipu: IPU,
        graph: Graph,
        context: dict[str, Any] | None = None,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        verbose: bool = False,
        default_branch: Branch | str | None = None,
    ): ...
```

### Parameters

- `session` (Session): Session for operation execution
- `ipu` (IPU): IPU for validated execution
- `graph` (Graph): Operation graph to execute
- `context` (dict, optional): Shared context for all operations
- `max_concurrent` (int, optional): Max concurrent operations
- `stop_on_error` (bool): Stop on first error
- `verbose` (bool): Print progress
- `default_branch` (Branch | str, optional): Default branch for operations

### Methods

#### `execute()`

Execute the graph and return all results.

```python
async def execute(self) -> dict[str, Any]: ...
```

**Returns**: Dictionary mapping operation names to results

#### `stream_execute()`

Execute the graph, yielding results as operations complete.

```python
async def stream_execute(self) -> AsyncGenerator[OperationResult, None]: ...
```

**Yields**: `OperationResult` objects

### Example

```python
executor = DependencyAwareExecutor(
    session=session,
    ipu=ipu,
    graph=graph,
    max_concurrent=3,
    verbose=True,
)

results = await executor.execute()
```

## Execution Model

### Dependency Resolution

Operations wait for all dependencies before executing:

```text
Graph:
  A → B → D
  A → C → D

Execution order:
1. A executes (no dependencies)
2. B and C execute concurrently (both depend only on A)
3. D executes (waits for B and C)
```

### Concurrency Control

The `max_concurrent` parameter limits parallel execution:

```python
# Unlimited concurrency (default)
results = await flow(session, "main", graph, ipu)

# Max 5 concurrent operations
results = await flow(session, "main", graph, ipu, max_concurrent=5)

# Sequential execution
results = await flow(session, "main", graph, ipu, max_concurrent=1)
```

**Note**: Operations still wait for dependencies regardless of concurrency limit.

### Context Propagation

Predecessor results are passed to dependent operations:

```python
# Build graph
builder.add("fetch", "communicate", instruction="Fetch data", imodel="gpt-4o")
builder.add("process", "communicate",
    instruction="Process the data",
    imodel="gpt-4o",
    depends_on=["fetch"]
)

# Execute
results = await flow(session, "main", builder.build(), ipu)

# "process" operation receives context:
# {
#   "fetch_result": <result from fetch operation>
# }
```

### Aggregation

Aggregation operations collect results from multiple sources:

```python
builder.add("task1", "communicate", instruction="Task 1", imodel="gpt-4o")
builder.add("task2", "communicate", instruction="Task 2", imodel="gpt-4o")
builder.add_aggregation("combine", "communicate",
    instruction="Combine results",
    imodel="gpt-4o",
    source_names=["task1", "task2"]
)

results = await flow(session, "main", builder.build(), ipu)

# "combine" operation receives:
# {
#   "task1_result": <result from task1>,
#   "task2_result": <result from task2>
# }
```

## Usage Patterns

### Basic Execution

```python
from lionpride.operations import Builder, flow

builder = Builder()
builder.add("op1", "communicate", instruction="Op 1", imodel="gpt-4o")
builder.add("op2", "communicate", instruction="Op 2", imodel="gpt-4o", depends_on=["op1"])

graph = builder.build()
results = await flow(session, "main", graph, ipu)

print(results["op1"])
print(results["op2"])
```

### Streaming Progress

```python
from lionpride.operations import flow_stream

async for result in flow_stream(session, "main", graph, ipu):
    progress = f"[{result.completed}/{result.total}]"
    if result.success:
        print(f"{progress} ✓ {result.name}")
    else:
        print(f"{progress} ✗ {result.name}: {result.error}")
```

### Error Handling

```python
# Stop on first error (default)
try:
    results = await flow(session, "main", graph, ipu, stop_on_error=True)
except RuntimeError as e:
    print(f"Execution failed: {e}")

# Continue on errors
results = await flow(session, "main", graph, ipu, stop_on_error=False)

# Check for errors
for name, result in results.items():
    if isinstance(result, Exception):
        print(f"{name} failed: {result}")
```

### Shared Context

```python
# Provide context to all operations
context = {
    "dataset": "sales_2024",
    "region": "US",
}

results = await flow(
    session,
    "main",
    graph,
    ipu,
    context=context,
)

# Each operation receives context in parameters
```

### Verbose Execution

```python
results = await flow(
    session,
    "main",
    graph,
    ipu,
    verbose=True,
)

# Prints:
# Pre-allocated branches for 5 operations
# Operation 12345678 waiting for 2 dependencies
# Operation 12345678 prepared with 3 context items
# Executing operation: 12345678
# Completed operation: 12345678
# Operation 'op_name' completed
```

## Advanced Patterns

### Parallel Batch Processing

```python
builder = Builder()

# Create parallel tasks
for i in range(10):
    builder.add(
        f"task_{i}",
        "communicate",
        instruction=f"Process item {i}",
        imodel="gpt-4o",
    )

# Aggregate results
builder.add_aggregation(
    "combine",
    "communicate",
    instruction="Combine all results",
    imodel="gpt-4o",
)

# Execute with concurrency limit
results = await flow(
    session,
    "main",
    builder.build(),
    ipu,
    max_concurrent=5,  # Max 5 tasks at once
)
```

### Progressive Disclosure

```python
async def progressive_flow(session, graph, ipu):
    """Show results as they complete."""
    async for result in flow_stream(session, "main", graph, ipu):
        print(f"\n=== {result.name} completed ===")
        print(result.result)
        print(f"Progress: {result.completed}/{result.total}")

await progressive_flow(session, graph, ipu)
```

### Custom Executor

```python
executor = DependencyAwareExecutor(
    session=session,
    ipu=ipu,
    graph=graph,
    context={"mode": "production"},
    max_concurrent=10,
    stop_on_error=False,  # Continue on errors
    verbose=True,
)

# Access executor internals
print(f"Total operations: {len(executor.graph.nodes)}")

# Execute
results = await executor.execute()

# Check execution state
print(f"Errors: {len(executor.errors)}")
print(f"Skipped: {len(executor.skipped_operations)}")
```

### Dynamic Graph Execution

```python
# Build initial graph
builder = Builder()
builder.add("analyze", "communicate", instruction="Analyze", imodel="gpt-4o")

# Execute and get results
results1 = await flow(session, "main", builder.build(), ipu)

# Extend graph based on results
if results1["analyze"]["needs_more_data"]:
    builder.add("fetch_more", "communicate",
        instruction="Fetch additional data",
        imodel="gpt-4o",
        depends_on=["analyze"]
    )
    builder.add("reanalyze", "communicate",
        instruction="Reanalyze with new data",
        imodel="gpt-4o",
        depends_on=["fetch_more"]
    )

# Execute extended graph
results2 = await flow(session, "main", builder.build(), ipu)
```

## Performance Optimization

### Concurrency Tuning

```python
# CPU-bound operations (LLM calls are I/O-bound)
# Higher concurrency = better throughput
results = await flow(session, "main", graph, ipu, max_concurrent=20)

# Memory-constrained environments
# Lower concurrency = less memory usage
results = await flow(session, "main", graph, ipu, max_concurrent=3)

# Respect API rate limits
# Limit concurrent calls to avoid throttling
results = await flow(session, "main", graph, ipu, max_concurrent=5)
```

### Graph Structure Optimization

```python
# ❌ Bad: Sequential bottleneck
builder.sequence("fetch", "process1", "process2", "process3")

# ✓ Good: Parallel processing
builder.add("fetch", "communicate", instruction="Fetch", imodel="gpt-4o")
builder.add("process1", "communicate", instruction="Process 1", imodel="gpt-4o", depends_on=["fetch"])
builder.add("process2", "communicate", instruction="Process 2", imodel="gpt-4o", depends_on=["fetch"])
builder.add("process3", "communicate", instruction="Process 3", imodel="gpt-4o", depends_on=["fetch"])

# fetch → [process1, process2, process3] (parallel)
```

## Common Pitfalls

- **Cyclic graphs**: Creating circular dependencies
  - **Solution**: `build()` validates DAG and raises ValueError on cycles

- **Missing IPU**: Not providing IPU to flow()
  - **Solution**: Create and pass IPU instance from session

- **Unbounded concurrency**: Not setting max_concurrent for large graphs
  - **Solution**: Set reasonable limit based on resources and API limits

- **Error handling**: Not checking for errors in non-stop-on-error mode
  - **Solution**: Check result types or use try/except

- **Context confusion**: Expecting automatic context merging
  - **Solution**: Context propagation uses `{name}_result` keys for predecessor results

## Design Rationale

### Why Separate flow() from Builder?

Separating building from execution enables:

- **Reusability**: Execute same graph multiple times
- **Testing**: Build and validate without execution
- **Serialization**: Store graphs for later execution
- **Flexibility**: Same graph with different sessions/branches

### Why Dependency-Aware Execution?

Dependency coordination ensures:

- **Correctness**: Operations execute only when dependencies are satisfied
- **Efficiency**: Maximum parallelism within dependency constraints
- **Simplicity**: No manual coordination code

### Why IPU Validation?

IPU provides:

- **Consistency**: Uniform validation across all operations
- **Traceability**: Execution tracking and logging
- **Error Handling**: Standardized error reporting
- **Security**: Operation validation before execution

### Why Streaming Mode?

Streaming execution enables:

- **Progress Monitoring**: Real-time updates for long-running workflows
- **Early Results**: Access results as soon as available
- **Memory Efficiency**: Process results incrementally
- **User Feedback**: Show progress in UI applications

## See Also

- **Related Components**:
  - [`OperationGraphBuilder`](builder.md): Build operation graphs
  - [IPU](../ipu/overview.md): Input Processing Unit
  - [Graph](../base/graph.md): Underlying graph structure

- **Related Operations**:
  - [`communicate`](communicate.md): Chat operation
  - [`operate`](operate.md): Structured operation
  - [`react`](react.md): ReAct operation

- **User Guide**:
  - Workflow patterns (documentation pending)
  - Performance optimization (documentation pending)

## Examples

### Complete Example: Data Processing Pipeline

```python
from lionpride.operations import Builder, flow_stream
from pydantic import BaseModel

class DataSource(BaseModel):
    records: list[dict]
    count: int

class ProcessedData(BaseModel):
    cleaned: list[dict]
    errors: list[str]

class Analysis(BaseModel):
    summary: str
    insights: list[str]

# Build pipeline
builder = Builder()

# Parallel data sources
builder.add("source_db", "operate",
    instruction="Fetch from database",
    imodel="gpt-4o",
    response_model=DataSource,
)

builder.add("source_api", "operate",
    instruction="Fetch from API",
    imodel="gpt-4o",
    response_model=DataSource,
)

# Process each source
builder.add("process_db", "operate",
    instruction="Clean and validate DB data",
    imodel="gpt-4o",
    response_model=ProcessedData,
    depends_on=["source_db"],
)

builder.add("process_api", "operate",
    instruction="Clean and validate API data",
    imodel="gpt-4o",
    response_model=ProcessedData,
    depends_on=["source_api"],
)

# Combine and analyze
builder.add_aggregation("analyze", "operate",
    instruction="Analyze combined data",
    imodel="gpt-4o",
    response_model=Analysis,
    source_names=["process_db", "process_api"],
)

# Generate report
builder.add("report", "communicate",
    instruction="Generate executive report",
    imodel="gpt-4o",
    depends_on=["analyze"],
)

# Build graph
graph = builder.build()

# Execute with progress tracking
print("Starting pipeline...")
async for result in flow_stream(session, "main", graph, ipu, max_concurrent=3):
    print(f"[{result.completed}/{result.total}] {result.name}")
    if result.success:
        print(f"  ✓ Completed")
    else:
        print(f"  ✗ Failed: {result.error}")

print("\nPipeline complete!")

# Or execute without streaming
results = await flow(
    session,
    "main",
    graph,
    ipu,
    context={"timestamp": "2025-11-24"},
    max_concurrent=3,
    verbose=True,
)

# Access final report
print(results["report"])
```
