# react()

> Multi-step ReAct (Reasoning + Acting) loop with tool execution

## Overview

The `react()` function implements the ReAct (Reasoning + Acting) pattern for multi-step LLM workflows with tool calling. The LLM iteratively reasons about the task, calls tools to gather information, and uses the results to produce a final answer. This is ideal for complex tasks requiring multiple steps and external data.

**Key Capabilities:**

- **Multi-Step Reasoning**: Iterative reasoning loop up to max_steps
- **Tool Execution**: Automatic tool calling and result integration
- **Chain of Thought**: Explicit reasoning at each step
- **Final Answer**: Structured final response with optional validation
- **Execution Tracking**: Detailed step-by-step history

## Signature

```python
async def react(
    session: Session,
    branch: Branch | str,
    parameters: ReactParam | dict,
) -> ReactResult
```

## Parameters

- `session` (Session): Session for tool access and message persistence
- `branch` (Branch | str): Branch for conversation context
- `parameters` (ReactParam | dict): ReAct parameters

### ReactParam Fields

```python
@dataclass(slots=True, frozen=True, init=False)
class ReactParam(Params):
    instruction: str = None              # Task instruction (required)
    imodel: str | iModel = None          # Model name or instance (required)
    tools: list = None                   # Tool classes/instances (required)
    response_model: type[BaseModel] = None  # Final answer schema
    model_name: str = None               # Model name for invocation (required)
    context: Any = None                  # Additional context
    max_steps: int = 5                   # Maximum ReAct steps
    use_lndl: bool = False              # Use LNDL mode (vs JSON)
    lndl_threshold: float = 0.85         # LNDL similarity threshold
    verbose: bool = False                # Verbose output
```

## Returns

Returns a `ReactResult` object:

```python
class ReactResult(BaseModel):
    steps: list[ReactStep]               # Execution steps
    final_response: Any                  # Final answer (validated if response_model)
    total_steps: int                     # Total steps executed
    completed: bool                      # Whether execution completed normally
    reason_stopped: str                  # Why execution stopped
```

### ReactStep Structure

```python
class ReactStep(BaseModel):
    step: int                            # Step number (1-indexed)
    reasoning: str | None                # LLM reasoning
    actions_requested: list[ActionRequestModel]  # Tool calls requested
    actions_executed: list[ActionResponseModel]  # Tool execution results
    is_final: bool                       # Whether this is the final step
```

## Basic Usage

### Simple Tool-Using Agent

```python
from lionpride.services.types import Tool
from pydantic import BaseModel

class SearchTool(Tool):
    name = "search"
    description = "Search the web"

    async def invoke(self, query: str) -> dict:
        # Search implementation
        return {"results": [...]}

class Answer(BaseModel):
    answer: str
    sources: list[str]

result = await react(
    session,
    "main",
    ReactParam(
        instruction="What is the latest Python version?",
        imodel="gpt-4o",
        tools=[SearchTool],
        model_name="gpt-4o",
        response_model=Answer,
    )
)

print(result.total_steps)  # Number of steps taken
print(result.final_response)  # Answer(answer="Python 3.13", sources=[...])
```

## ReAct Flow

The ReAct loop follows this pattern:

```text
1. LLM receives instruction + tool descriptions
2. LLM reasons about the task
3. LLM either:
   a. Requests tool calls → Tools execute → Results added to context → Go to step 2
   b. Returns final answer → Loop ends
4. Repeat until final answer or max_steps reached
```

### Step-by-Step Example

```python
result = await react(
    session,
    "main",
    ReactParam(
        instruction="Find the current weather in Tokyo",
        imodel="gpt-4o",
        tools=[WeatherTool, SearchTool],
        model_name="gpt-4o",
        max_steps=5,
        verbose=True,
    )
)

# Console output:
# --- ReAct Step 1/5 ---
# Reasoning: I need to find Tokyo's current weather...
# Executing 1 action(s)...
#   Tool get_weather: {"temp": 18, "condition": "Sunny"}
#
# --- ReAct Step 2/5 ---
# Reasoning: I have the weather data. I can now provide the final answer.
# Task completed at step 2
# Final answer: The current weather in Tokyo is 18°C and Sunny.

# result.steps contains all steps
for step in result.steps:
    print(f"Step {step.step}: {step.reasoning[:50]}...")
    print(f"  Actions: {len(step.actions_requested)}")
```

## Tool Registration

### Tool Classes

```python
from lionpride.services.types import Tool
from pydantic import Field

class CalculatorTool(Tool):
    name = "calculator"
    description = "Perform calculations"

    # Define tool schema
    @property
    def tool_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate"
                }
            },
            "required": ["expression"]
        }

    async def invoke(self, expression: str) -> dict:
        try:
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

# Use in react()
result = await react(
    session,
    "main",
    ReactParam(
        instruction="What is 123 * 456?",
        imodel="gpt-4o",
        tools=[CalculatorTool],  # Pass class, not instance
        model_name="gpt-4o",
    )
)
```

### Multiple Tools

```python
result = await react(
    session,
    "main",
    ReactParam(
        instruction="Search for Python and calculate days since release",
        imodel="gpt-4o",
        tools=[SearchTool, CalculatorTool, DateTool],
        model_name="gpt-4o",
        max_steps=10,
    )
)

# LLM can use all tools as needed
# Example flow:
# 1. SearchTool("Python release date") → "Dec 3, 1991"
# 2. DateTool("current date") → "Nov 24, 2025"
# 3. CalculatorTool("days between dates") → 12410
# 4. Final answer: "Python was released 12,410 days ago"
```

## Response Validation

### Structured Final Answer

```python
from pydantic import BaseModel, Field

class ResearchResult(BaseModel):
    summary: str = Field(..., description="Research summary")
    key_findings: list[str] = Field(..., description="Key findings")
    sources: list[str] = Field(..., description="Source URLs")

result = await react(
    session,
    "main",
    ReactParam(
        instruction="Research lionpride framework",
        imodel="gpt-4o",
        tools=[SearchTool],
        model_name="gpt-4o",
        response_model=ResearchResult,
    )
)

# result.final_response is validated ResearchResult instance
print(result.final_response.summary)
print(result.final_response.key_findings)
```

## Advanced Patterns

### Max Steps and Early Termination

```python
result = await react(
    session,
    "main",
    ReactParam(
        instruction="Complex research task",
        imodel="gpt-4o",
        tools=[SearchTool, AnalysisTool],
        model_name="gpt-4o",
        max_steps=10,
    )
)

# Check why execution stopped
if result.completed:
    print("Task completed successfully")
else:
    print(f"Stopped: {result.reason_stopped}")
    # Possible reasons:
    # - "Max steps (10) reached"
    # - "Error at step 3: ..."
    # - "Validation failed: ..."
```

### Verbose Debugging

```python
result = await react(
    session,
    "main",
    ReactParam(
        instruction="Debug this task",
        imodel="gpt-4o",
        tools=[DebugTool],
        model_name="gpt-4o",
        verbose=True,  # Print detailed execution info
    )
)

# Prints:
# - Step number and progress
# - Reasoning text
# - Tool calls and arguments
# - Tool results
# - Final answer
```

### Custom Context

```python
result = await react(
    session,
    "main",
    ReactParam(
        instruction="Analyze using this data",
        imodel="gpt-4o",
        tools=[AnalysisTool],
        model_name="gpt-4o",
        context={
            "dataset": "sales_2024.csv",
            "focus": "quarterly trends",
        },
    )
)

# Context is included in LLM prompt
```

### LNDL Mode

```python
result = await react(
    session,
    "main",
    ReactParam(
        instruction="Extract structured data",
        imodel="gpt-4o",
        tools=[ScrapeTool],
        model_name="gpt-4o",
        response_model=DataModel,
        use_lndl=True,  # Use fuzzy LNDL parsing
        lndl_threshold=0.9,
    )
)
```

## Result Analysis

### Inspecting Steps

```python
result = await react(session, "main", params)

# Total steps
print(f"Completed in {result.total_steps} steps")

# Step details
for step in result.steps:
    print(f"\nStep {step.step}:")
    print(f"  Reasoning: {step.reasoning}")
    print(f"  Tools called: {len(step.actions_requested)}")
    for action in step.actions_executed:
        print(f"    {action.function}: {action.output}")

# Final step
final_step = result.steps[-1]
if final_step.is_final:
    print("Task completed successfully")
```

### Error Handling

```python
result = await react(session, "main", params)

if not result.completed:
    print(f"Failed: {result.reason_stopped}")

    # Check last step for errors
    last_step = result.steps[-1] if result.steps else None
    if last_step:
        for action in last_step.actions_executed:
            if action.error:
                print(f"Tool error: {action.function} - {action.error}")
```

## Common Pitfalls

- **Missing model_name**: Not providing `model_name` parameter
  - **Solution**: Always provide model name: `model_name="gpt-4o"`

- **Missing tools**: Not providing any tools
  - **Solution**: Provide at least one Tool class/instance: `tools=[SearchTool]`

- **Tool not callable**: Providing invalid tool type
  - **Solution**: Tools must be Tool subclass or instance

- **Infinite loops**: LLM never returns final answer
  - **Solution**: Set reasonable `max_steps` (default: 5)

- **Tool registration**: Tools fail to register in session
  - **Solution**: react() auto-registers tools, but ensure valid Tool implementations

- **Response model mismatch**: Final answer doesn't match response_model
  - **Solution**: Ensure response_model matches expected final answer structure

## Design Rationale

### Why ReAct Pattern?

ReAct combines reasoning (thinking) and acting (tool use) in an iterative loop. This enables:

- **Complex Tasks**: Multi-step workflows with data gathering
- **Explainability**: Explicit reasoning at each step
- **Flexibility**: LLM decides when to use tools vs reasoning
- **Error Recovery**: LLM can retry tools or change strategy

### Why Automatic Tool Execution?

Manual tool execution adds boilerplate. Automatic execution:

- **Simplifies Code**: No manual action parsing/execution
- **Ensures Consistency**: Standard tool calling pattern
- **Improves Reliability**: Built-in error handling

### Why Step Tracking?

Detailed step history enables:

- **Debugging**: Understand LLM decision-making
- **Optimization**: Analyze tool usage patterns
- **Auditing**: Track what data was accessed
- **Learning**: Improve prompts based on step analysis

## See Also

- **Related Functions**:
  - [`operate`](operate.md): Structured outputs with actions (single-step)
  - [`communicate`](communicate.md): Stateful chat (no tools)
  - [`generate`](generate.md): Stateless generation

- **Related Types**:
  - [Tool](../services/tool.md): Tool implementation interface
  - [ActionRequestModel](../types/action.md): Tool call representation
  - [ActionResponseModel](../types/action.md): Tool result representation

- **User Guide**:
  - Building tool-using agents (documentation pending)
  - ReAct pattern best practices (documentation pending)

## Examples

### Complete Example: Research Agent

```python
from lionpride.services.types import Tool
from pydantic import BaseModel, Field

# Define tools
class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information"

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }

    async def invoke(self, query: str) -> dict:
        # Search implementation
        results = ["Result 1", "Result 2", "Result 3"]
        return {"results": results, "query": query}

class SummarizeTool(Tool):
    name = "summarize"
    description = "Summarize text content"

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"}
            },
            "required": ["text"]
        }

    async def invoke(self, text: str) -> dict:
        # Summarization implementation
        summary = f"Summary of: {text[:50]}..."
        return {"summary": summary}

# Define response structure
class ResearchReport(BaseModel):
    topic: str = Field(..., description="Research topic")
    summary: str = Field(..., description="Research summary")
    key_points: list[str] = Field(..., description="Key points found")
    sources_used: list[str] = Field(..., description="Tools/sources used")

# Execute research task
result = await react(
    session,
    "main",
    ReactParam(
        instruction="""
        Research the lionpride framework:
        1. Search for documentation
        2. Summarize key features
        3. Identify main use cases
        """,
        imodel="gpt-4o",
        tools=[WebSearchTool, SummarizeTool],
        model_name="gpt-4o",
        response_model=ResearchReport,
        max_steps=8,
        verbose=True,
    )
)

# Print results
print(f"Completed in {result.total_steps} steps")
print(f"\nTopic: {result.final_response.topic}")
print(f"Summary: {result.final_response.summary}")
print(f"\nKey Points:")
for point in result.final_response.key_points:
    print(f"  - {point}")
print(f"\nSources: {', '.join(result.final_response.sources_used)}")

# Analyze execution
print(f"\nExecution Details:")
for step in result.steps:
    print(f"\nStep {step.step}:")
    print(f"  Reasoning: {step.reasoning[:100]}...")
    if step.actions_requested:
        print(f"  Tools called: {[a.function for a in step.actions_requested]}")
```

### Complete Example: Data Analysis Agent

```python
class DataLoaderTool(Tool):
    name = "load_data"
    description = "Load dataset from file"

    async def invoke(self, filename: str) -> dict:
        # Load data implementation
        return {"rows": 1000, "columns": ["A", "B", "C"]}

class StatsTool(Tool):
    name = "calculate_stats"
    description = "Calculate statistics"

    async def invoke(self, column: str) -> dict:
        # Statistics implementation
        return {"mean": 42.5, "std": 10.2, "min": 10, "max": 100}

class Analysis(BaseModel):
    dataset: str
    statistics: dict
    insights: list[str]
    recommendation: str

result = await react(
    session,
    "main",
    ReactParam(
        instruction="""
        Analyze sales_2024.csv:
        1. Load the data
        2. Calculate statistics for revenue column
        3. Identify trends
        4. Provide recommendations
        """,
        imodel="gpt-4o",
        tools=[DataLoaderTool, StatsTool],
        model_name="gpt-4o",
        response_model=Analysis,
        max_steps=6,
    )
)

print(result.final_response.recommendation)
```
