# Tool Calling with ReAct

## Overview

This recipe demonstrates tool calling using lionpride's `react()` operation - a multi-step reasoning loop where the LLM can call tools, observe results, and continue reasoning until it reaches a final answer.

## Prerequisites

```bash
pip install lionpride pydantic
```

## The Code

### Example 1: Simple Calculator Tools

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel, Tool
from lionpride.operations import react

def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

async def math_assistant():
    """LLM uses calculator tools to solve math problems"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    # Create tools from Python functions
    add_tool = Tool.from_function(add)
    multiply_tool = Tool.from_function(multiply)

    # Register tools
    session.services.register(add_tool)
    session.services.register(multiply_tool)

    branch = session.create_branch(name="math")

    # ReAct loop: LLM can call tools multiple times
    result = await react(
        session=session,
        branch=branch,
        parameters={
            "instruction": "What is (15 + 27) × 3?",
            "imodel": model.name,
            "tools": [add_tool.name, multiply_tool.name],
            "max_steps": 5,
        }
    )

    print(f"Final Answer: {result.final_response}")
    print(f"\nSteps taken: {result.total_steps}")
    print(f"Completed: {result.completed}")

    # Show reasoning trace
    for step in result.steps:
        print(f"\nStep {step.step}:")
        print(f"  Reasoning: {step.reasoning}")
        if step.actions_requested:
            for action in step.actions_requested:
                print(f"  Tool called: {action.function}")
                print(f"  Arguments: {action.arguments}")
        if step.actions_executed:
            for action in step.actions_executed:
                print(f"  Result: {action.output}")

    return result

asyncio.run(math_assistant())
```

### Example 2: Web Search Tool

```python
import asyncio
from datetime import datetime
from lionpride import Session
from lionpride.services import iModel, Tool
from lionpride.operations import react

def search_web(query: str, num_results: int = 3) -> list[dict]:
    """
    Search the web for information.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results with title, snippet, url
    """
    # Simulated search results
    return [
        {
            "title": f"Result {i+1} for '{query}'",
            "snippet": f"This is a snippet about {query}...",
            "url": f"https://example.com/result{i+1}"
        }
        for i in range(num_results)
    ]

def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")

async def research_assistant():
    """LLM uses search to answer questions"""
    session = Session()
    model = iModel(provider="anthropic", endpoint="messages",
                   model="claude-3-5-sonnet-20241022", temperature=0)
    session.services.register(model)

    # Create and register tools
    search_tool = Tool.from_function(search_web)
    date_tool = Tool.from_function(get_current_date)

    session.services.register(search_tool)
    session.services.register(date_tool)

    branch = session.create_branch(name="research")

    result = await react(
        session=session,
        branch=branch,
        parameters={
            "instruction": "What major tech events happened this year?",
            "imodel": model.name,
            "tools": [search_tool.name, date_tool.name],
            "max_steps": 3,
        }
    )

    print(f"Research Result:\n{result.final_response}")

    return result

asyncio.run(research_assistant())
```

### Example 3: Data Processing Pipeline

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel, Tool
from lionpride.operations import react

# Simulated database
DATABASE = {
    "users": [
        {"id": 1, "name": "Alice", "department": "Engineering", "salary": 120000},
        {"id": 2, "name": "Bob", "department": "Sales", "salary": 90000},
        {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 110000},
        {"id": 4, "name": "Diana", "department": "Sales", "salary": 95000},
    ]
}

def query_database(table: str, filter_by: str | None = None) -> list[dict]:
    """
    Query database table.

    Args:
        table: Table name
        filter_by: Optional filter (e.g., "department=Engineering")
    """
    data = DATABASE.get(table, [])

    if filter_by:
        key, value = filter_by.split("=")
        data = [row for row in data if str(row.get(key)) == value]

    return data

def calculate_average(numbers: list[float]) -> float:
    """Calculate average of a list of numbers"""
    return sum(numbers) / len(numbers) if numbers else 0

async def data_analyst():
    """LLM analyzes data using tools"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o", temperature=0)
    session.services.register(model)

    # Register data tools
    query_tool = Tool.from_function(query_database)
    avg_tool = Tool.from_function(calculate_average)

    session.services.register(query_tool)
    session.services.register(avg_tool)

    branch = session.create_branch(name="analysis")

    result = await react(
        session=session,
        branch=branch,
        parameters={
            "instruction": (
                "What is the average salary of employees in the Engineering department?"
            ),
            "imodel": model.name,
            "tools": [query_tool.name, avg_tool.name],
            "max_steps": 5,
        }
    )

    print(f"Analysis Result: {result.final_response}")

    # Show tool calls
    print("\nTool Execution Trace:")
    for step in result.steps:
        if step.actions_executed:
            for action in step.actions_executed:
                print(f"  {action.function}({action.arguments}) → {action.output}")

    return result

asyncio.run(data_analyst())
```

### Example 4: Structured Output from ReAct

Get a typed response after tool calling:

```python
import asyncio
from pydantic import BaseModel, Field
from lionpride import Session
from lionpride.services import iModel, Tool
from lionpride.operations import react

class WeatherReport(BaseModel):
    """Weather report with recommendations"""
    temperature: float = Field(..., description="Temperature in Fahrenheit")
    condition: str = Field(..., description="Weather condition")
    recommendation: str = Field(..., description="What to wear")

def get_weather(city: str) -> dict:
    """Get current weather for a city"""
    # Simulated weather API
    weather_data = {
        "San Francisco": {"temp": 65, "condition": "Foggy"},
        "New York": {"temp": 75, "condition": "Sunny"},
        "Seattle": {"temp": 55, "condition": "Rainy"},
    }
    return weather_data.get(city, {"temp": 70, "condition": "Unknown"})

async def weather_assistant():
    """Get weather and recommendations with structured output"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    weather_tool = Tool.from_function(get_weather)
    session.services.register(weather_tool)

    branch = session.create_branch(name="weather")

    result = await react(
        session=session,
        branch=branch,
        parameters={
            "instruction": "What's the weather in Seattle and what should I wear?",
            "imodel": model.name,
            "tools": [weather_tool.name],
            "response_model": WeatherReport,  # Structured final output
            "max_steps": 3,
        }
    )

    weather = result.final_response
    print(f"Temperature: {weather.temperature}°F")
    print(f"Condition: {weather.condition}")
    print(f"Recommendation: {weather.recommendation}")

    return result

asyncio.run(weather_assistant())
```

### Example 5: Parallel Tool Calling

Some models support calling multiple tools in parallel:

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel, Tool
from lionpride.operations import react

def get_stock_price(symbol: str) -> float:
    """Get current stock price"""
    # Simulated stock prices
    prices = {"AAPL": 178.50, "GOOGL": 142.30, "MSFT": 380.20}
    return prices.get(symbol.upper(), 0.0)

def get_company_info(symbol: str) -> dict:
    """Get company information"""
    info = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
        "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
    }
    return info.get(symbol.upper(), {})

async def stock_analyzer():
    """Analyze multiple stocks with parallel tool calls"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o", temperature=0)
    session.services.register(model)

    price_tool = Tool.from_function(get_stock_price)
    info_tool = Tool.from_function(get_company_info)

    session.services.register(price_tool)
    session.services.register(info_tool)

    branch = session.create_branch(name="stocks")

    result = await react(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Compare Apple (AAPL) and Microsoft (MSFT) stocks",
            "imodel": model.name,
            "tools": [price_tool.name, info_tool.name],
            "max_steps": 3,
        }
    )

    print(f"Stock Analysis:\n{result.final_response}")

    return result

asyncio.run(stock_analyzer())
```

## Expected Output

### Example 1 (Calculator)

```
Final Answer: 126

Steps taken: 2
Completed: True

Step 1:
  Reasoning: I need to first add 15 and 27
  Tool called: add
  Arguments: {'a': 15, 'b': 27}
  Result: 42

Step 2:
  Reasoning: Now multiply the result by 3
  Tool called: multiply
  Arguments: {'a': 42, 'b': 3}
  Result: 126
```

### Example 2 (Web Search)

```
Research Result:
Based on the search results, major tech events this year include...
[LLM synthesizes information from search results]
```

### Example 3 (Data Analysis)

```
Analysis Result: The average salary of employees in the Engineering department is $115,000.

Tool Execution Trace:
  query_database({'table': 'users', 'filter_by': 'department=Engineering'}) → [{'id': 1, 'name': 'Alice', ...}, ...]
  calculate_average({'numbers': [120000, 110000]}) → 115000.0
```

### Example 4 (Weather)

```
Temperature: 55.0°F
Condition: Rainy
Recommendation: Bring an umbrella and wear a waterproof jacket. Layers are recommended due to the cool temperature.
```

## Key Concepts

### ReAct Loop

```
1. LLM receives instruction
2. LLM reasons about what tool to call
3. Tools are executed
4. Results fed back to LLM
5. Repeat until LLM decides it has the final answer
```

### Tool Definition

```python
# From function (automatic schema generation)
tool = Tool.from_function(my_function)

# Manual definition
from lionpride.services import ToolConfig

tool = Tool(
    config=ToolConfig(
        name="my_tool",
        description="What the tool does",
        input_schema={
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param"]
        },
    ),
    executor=my_function,
)
```

### Max Steps

Control the maximum number of reasoning iterations:

```python
result = await react(
    ...,
    parameters={
        "max_steps": 5,  # Stop after 5 iterations
    }
)

# Check if completed
if not result.completed:
    print(f"Stopped early: {result.reason_stopped}")
```

## Variations

### Error Handling in Tools

```python
def safe_divide(a: float, b: float) -> float:
    """Divide two numbers with error handling"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Tool executor catches exceptions and returns error to LLM
divide_tool = Tool.from_function(safe_divide)

# LLM sees the error and can retry with different arguments
```

### Async Tools

```python
import httpx

async def fetch_url(url: str) -> str:
    """Fetch content from URL (async)"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

# Works seamlessly with async functions
fetch_tool = Tool.from_function(fetch_url)
```

### Tool Selection Hints

```python
# Guide the LLM with better descriptions
def search_arxiv(query: str) -> list[dict]:
    """
    Search academic papers on arXiv.

    Use this tool when the user asks about:
    - Scientific research
    - Academic papers
    - Technical publications

    Args:
        query: Search query (e.g., "quantum computing")
    """
    # Implementation
    pass
```

## Common Pitfalls

1. **Tools not registered**

   ```python
   # ❌ Wrong - tool created but not registered
   tool = Tool.from_function(my_func)
   result = await react(parameters={"tools": [tool.name]})  # Fails!

   # ✅ Right - register first
   session.services.register(tool)
   result = await react(parameters={"tools": [tool.name]})
   ```

2. **Missing type hints**

   ```python
   # ❌ Wrong - no type hints, poor schema generation
   def my_tool(x, y):
       return x + y

   # ✅ Right - explicit types for better schema
   def my_tool(x: float, y: float) -> float:
       """Add two numbers"""
       return x + y
   ```

3. **Infinite loops**

   ```python
   # ❌ Wrong - no max_steps, could loop forever
   result = await react(parameters={...})

   # ✅ Right - set reasonable limit
   result = await react(parameters={"max_steps": 10})
   ```

4. **Tool naming conflicts**

   ```python
   # ❌ Wrong - same name
   tool1 = Tool.from_function(func1)  # name="func"
   tool2 = Tool.from_function(func2)  # name="func" - conflict!

   # ✅ Right - explicit unique names
   tool1 = Tool.from_function(func1, name="func1")
   tool2 = Tool.from_function(func2, name="func2")
   ```

## Next Steps

- **Multi-agent workflows**: See [Multi-Agent](multi_agent.md)
- **Streaming tool calls**: See [Streaming](streaming.md)
- **Error handling strategies**: See [Error Handling](error_handling.md)

## See Also

- [API Reference: Tool](../api/services.md#tool)
- [API Reference: react()](../api/operations.md#react)
- [ReAct Pattern Deep Dive](../patterns/react.md)
