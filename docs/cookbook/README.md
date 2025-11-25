# lionpride Cookbook

**Real-world, production-ready examples for building with lionpride.**

Each recipe is self-contained, runnable, and demonstrates a specific pattern or use case. All examples use the lionpride v2 API (alpha).

## Getting Started

### Prerequisites

```bash
# Install lionpride
pip install lionpride

# Set API keys for providers you want to use
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

### Recipe Format

Each recipe follows this structure:

- **Overview**: What the recipe demonstrates
- **Prerequisites**: Required setup and dependencies
- **The Code**: Complete, runnable examples
- **Expected Output**: What you'll see when running
- **Variations**: Alternative approaches and extensions
- **Common Pitfalls**: Mistakes to avoid
- **Next Steps**: Related recipes and documentation

## Recipes

### 1. [Quick Start](quickstart.md) ⭐

**Get started in 5 minutes**

Learn the basics: Session, iModel, Branch, and your first chat interaction. This is your entry point to lionpride.

```python
session = Session()
model = iModel(provider="openai", model="gpt-4o-mini")
session.services.register(model)
branch = session.create_branch(name="main")
result = await session.invoke(operation="communicate", ...)
```

**Time**: 5 minutes
**Difficulty**: Beginner
**Topics**: Session, Branch, iModel, communicate

---

### 2. [Chat with Different Providers](chat.md)

**Provider flexibility with unified API**

Use the same API across OpenAI, Anthropic, Gemini, Groq, and more. Compare responses, implement fallback chains, and understand provider-specific features.

```python
# Same API, different providers
openai_model = iModel(provider="openai", model="gpt-4o-mini")
claude_model = iModel(provider="anthropic", endpoint="messages",
                      model="claude-3-5-sonnet-20241022")
gemini_model = iModel(provider="gemini", model="gemini-2.0-flash-exp")
```

**Time**: 10 minutes
**Difficulty**: Beginner
**Topics**: Multi-provider, iModel, configuration

---

### 3. [Structured Outputs](structured_outputs.md)

**Type-safe responses with Spec and Operable**

Extract structured data from LLM responses using LNDL (75% token reduction) or Pydantic models. Get typed, validated outputs instead of raw text.

```python
# LNDL approach - efficient
person_spec = Operable(specs=[
    Spec(str, name="name"),
    Spec(int, name="age"),
    Spec(list[str], name="skills"),
])
result = await communicate(operable=person_spec, return_as="model")

# Pydantic approach - strict validation
class Person(BaseModel):
    name: str
    age: int
    skills: list[str]

result = await communicate(response_model=Person, return_as="model")
```

**Time**: 15 minutes
**Difficulty**: Intermediate
**Topics**: Spec, Operable, LNDL, Pydantic, validation

---

### 4. [Tool Calling](tool_calling.md)

**ReAct pattern for agentic workflows**

Build agents that use tools to solve problems. Multi-step reasoning loops where LLMs call Python functions, observe results, and continue until reaching a solution.

```python
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

add_tool = Tool.from_function(add)
session.services.register(add_tool)

result = await react(
    instruction="What is 15 + 27?",
    tools=[add_tool.name],
    max_steps=5,
)
```

**Time**: 20 minutes
**Difficulty**: Intermediate
**Topics**: react(), Tool, ReAct pattern, function calling

---

### 5. [Multi-Turn Conversations](multi_turn.md)

**Context management with Branch**

Build conversations with memory. Manage context, inspect history, create multiple conversation threads, and persist/resume conversations.

```python
branch = session.create_branch(name="tutoring", system=system_msg)

# Turn 1
await communicate(instruction="What are Python lists?", ...)

# Turn 2 - has context from turn 1
await communicate(instruction="How do I add items?", ...)

# Inspect history
for msg_id in branch.order:
    message = session.messages[msg_id]
    print(f"{message.role}: {message.content}")
```

**Time**: 15 minutes
**Difficulty**: Beginner
**Topics**: Branch, Session, Message, conversation history

---

### 6. [Multi-Agent Workflows](multi_agent.md)

**Orchestrate agents with Builder and Flow**

Create operation graphs where multiple specialized agents collaborate. Sequential, parallel, and dependency-aware execution patterns.

```python
builder = Builder(session=session)

research = builder.communicate(instruction="Research quantum computing", ...)
write = builder.communicate(instruction="Write article", context_from=[research])
edit = builder.communicate(instruction="Edit for clarity", context_from=[write])

results = await flow(builder.operations)  # Auto-executes in order
```

**Time**: 25 minutes
**Difficulty**: Advanced
**Topics**: Builder, flow(), operation graphs, multi-agent

---

### 7. [Streaming Responses](streaming.md)

**Real-time token streaming**

Stream LLM responses token-by-token for better UX. Handle progressive rendering, parallel streaming, and streaming workflows.

```python
model = iModel(provider="openai", model="gpt-4o-mini", stream=True)

async for chunk in model.invoke(messages=[...]):
    print(chunk.data, end="", flush=True)
```

**Time**: 15 minutes
**Difficulty**: Intermediate
**Topics**: Streaming, async iterators, flow_stream()

---

### 8. [Error Handling](error_handling.md)

**Production resilience patterns**

Build robust applications with retry logic, fallback chains, circuit breakers, validation recovery, and timeout handling.

```python
# Retry with exponential backoff
@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=10))
async def resilient_call():
    return await communicate(...)

# Fallback chain
for provider in [primary, fallback1, fallback2]:
    try:
        return await communicate(imodel=provider)
    except:
        continue  # Try next provider
```

**Time**: 20 minutes
**Difficulty**: Advanced
**Topics**: Retry, fallback, circuit breaker, validation, timeouts

---

## Learning Paths

### For Beginners

1. [Quick Start](quickstart.md) - Learn the basics
2. [Chat with Different Providers](chat.md) - Provider flexibility
3. [Multi-Turn Conversations](multi_turn.md) - Context management

### For Building Applications

1. [Structured Outputs](structured_outputs.md) - Type-safe responses
2. [Tool Calling](tool_calling.md) - Agentic workflows
3. [Error Handling](error_handling.md) - Production resilience

### For Advanced Users

1. [Multi-Agent Workflows](multi_agent.md) - Complex orchestration
2. [Streaming Responses](streaming.md) - Real-time UX
3. [Error Handling](error_handling.md) - Advanced patterns

## Common Patterns Quick Reference

### Basic Chat

```python
session = Session()
model = iModel(provider="openai", model="gpt-4o-mini")
session.services.register(model)
branch = session.create_branch()
result = await communicate(session, branch, {"instruction": "...", "imodel": model.name})
```

### Structured Output (LNDL)

```python
spec = Operable(specs=[Spec(str, name="field")])
result = await communicate(..., parameters={"operable": spec, "return_as": "model"})
```

### Structured Output (Pydantic)

```python
class MyModel(BaseModel):
    field: str

result = await communicate(..., parameters={"response_model": MyModel, "return_as": "model"})
```

### Tool Calling

```python
tool = Tool.from_function(my_function)
session.services.register(tool)
result = await react(..., parameters={"tools": [tool.name], "max_steps": 5})
```

### Multi-Agent Workflow

```python
builder = Builder(session=session)
op1 = builder.communicate(...)
op2 = builder.communicate(context_from=[op1])
results = await flow(builder.operations)
```

### Streaming

```python
model = iModel(..., stream=True)
async for chunk in model.invoke(messages=[...]):
    print(chunk.data, end="", flush=True)
```

## Additional Resources

### Documentation

- [API Reference](../api/) - Complete API documentation
- [User Guide](../user_guide/) - Conceptual guides
- [Integration Guides](../integration/) - Third-party integrations
- [Patterns](../patterns/) - Design patterns and best practices

### Examples

- [Notebooks](../../notebooks/) - Interactive Jupyter examples
- [Tests](../../tests/) - Test suite as examples

### Community

- [GitHub](https://github.com/khive-ai/lionpride) - Issues, PRs, discussions
- [License](../../LICENSE) - Apache 2.0

## Contributing

Found a bug in a recipe? Have a suggestion for a new recipe?

1. Check [existing issues](https://github.com/khive-ai/lionpride/issues)
2. Open a new issue with details
3. Submit a PR with improvements

## Version

These recipes are for **lionpride v2 (alpha)**. API may change before stable release.

Last updated: 2025-01-15

---

**Ready to start?** → Begin with [Quick Start](quickstart.md)
