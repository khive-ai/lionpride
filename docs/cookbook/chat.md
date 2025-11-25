# Simple Chat with Different Providers

## Overview

This recipe shows how to use lionpride with different LLM providers. The API is identical - only the `provider` and model parameters change.

## Prerequisites

Install lionpride and set up API keys:

```bash
pip install lionpride

# Set API keys for providers you want to use
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export GROQ_API_KEY="gsk_..."
```

## The Code

### Example 1: OpenAI GPT-4o

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def chat_openai():
    """Chat with OpenAI GPT-4o"""
    session = Session()

    # OpenAI configuration
    model = iModel(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=500,
    )
    session.services.register(model)

    branch = session.create_branch(name="openai-chat")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Explain quantum computing in simple terms",
            "imodel": model.name,
        }
    )

    print(f"OpenAI Response:\n{result}\n")
    return result

asyncio.run(chat_openai())
```

### Example 2: Anthropic Claude

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def chat_anthropic():
    """Chat with Anthropic Claude"""
    session = Session()

    # Anthropic configuration - note the 'messages' endpoint
    model = iModel(
        provider="anthropic",
        endpoint="messages",  # Important: Anthropic uses 'messages' endpoint
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=500,
    )
    session.services.register(model)

    branch = session.create_branch(name="claude-chat")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Explain quantum computing in simple terms",
            "imodel": model.name,
        }
    )

    print(f"Claude Response:\n{result}\n")
    return result

asyncio.run(chat_anthropic())
```

### Example 3: Google Gemini

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def chat_gemini():
    """Chat with Google Gemini"""
    session = Session()

    # Gemini configuration
    model = iModel(
        provider="gemini",
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        max_tokens=500,
    )
    session.services.register(model)

    branch = session.create_branch(name="gemini-chat")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Explain quantum computing in simple terms",
            "imodel": model.name,
        }
    )

    print(f"Gemini Response:\n{result}\n")
    return result

asyncio.run(chat_gemini())
```

### Example 4: Groq (Fast Inference)

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def chat_groq():
    """Chat with Groq - extremely fast inference"""
    session = Session()

    # Groq configuration - blazing fast
    model = iModel(
        provider="groq",
        model="llama-3.1-70b-versatile",
        temperature=0.7,
        max_tokens=500,
    )
    session.services.register(model)

    branch = session.create_branch(name="groq-chat")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Explain quantum computing in simple terms",
            "imodel": model.name,
        }
    )

    print(f"Groq Response:\n{result}\n")
    return result

asyncio.run(chat_groq())
```

### Example 5: Multi-Provider Comparison

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def compare_providers():
    """Compare responses from different providers"""
    session = Session()

    # Register multiple models
    models = [
        iModel(provider="openai", model="gpt-4o-mini", temperature=0.7),
        iModel(provider="anthropic", endpoint="messages",
               model="claude-3-5-haiku-20241022", temperature=0.7),
        iModel(provider="gemini", model="gemini-2.0-flash-exp", temperature=0.7),
    ]

    for model in models:
        session.services.register(model)

    question = "What is the difference between AI and machine learning?"

    # Ask same question to all models
    for model in models:
        branch = session.create_branch(name=f"{model.name}-branch")

        result = await communicate(
            session=session,
            branch=branch,
            parameters={
                "instruction": question,
                "imodel": model.name,
            }
        )

        print(f"\n{'='*60}")
        print(f"Provider: {model.backend.config.provider}")
        print(f"Model: {model.backend.config.model}")
        print(f"{'='*60}")
        print(result)
        print()

asyncio.run(compare_providers())
```

## Expected Output

### Example 1 (OpenAI)

```
OpenAI Response:
Quantum computing is a revolutionary approach to computation that leverages
the principles of quantum mechanics. Unlike classical computers that use bits
(0s and 1s), quantum computers use quantum bits or "qubits" that can exist in
multiple states simultaneously (superposition)...
```

### Example 2 (Anthropic)

```
Claude Response:
I'd be happy to explain quantum computing in simple terms!

Think of a regular computer like a light switch - it's either ON (1) or OFF (0).
Every calculation happens one step at a time using these bits.

Quantum computing is fundamentally different...
```

### Example 5 (Comparison Output)

```
============================================================
Provider: openai
Model: gpt-4o-mini
============================================================
AI (Artificial Intelligence) is the broader concept of machines
being able to carry out tasks in a way that we would consider "smart."
Machine Learning is a specific subset of AI...

============================================================
Provider: anthropic
Model: claude-3-5-haiku-20241022
============================================================
Great question! Here's a clear breakdown:

**Artificial Intelligence (AI)** is the broader field...

[Each provider's response shown]
```

## Provider Comparison

| Provider | Endpoint | Best For | Speed | Cost |
|----------|----------|----------|-------|------|
| OpenAI | `chat/completions` (default) | General purpose, reasoning | Medium | Medium |
| Anthropic | `messages` | Long context, analysis | Medium | Medium |
| Gemini | Default | Multimodal, free tier | Fast | Low |
| Groq | Default | Speed-critical apps | Very Fast | Low |

## Variations

### Custom API Base URL (Self-hosted or Proxy)

```python
model = iModel(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-key",
    base_url="https://your-api-proxy.com/v1",  # Custom endpoint
)
```

### Model-Specific Parameters

Different providers support different parameters:

```python
# OpenAI: frequency_penalty, presence_penalty
openai_model = iModel(
    provider="openai",
    model="gpt-4o",
    frequency_penalty=0.5,
    presence_penalty=0.5,
)

# Anthropic: top_k
claude_model = iModel(
    provider="anthropic",
    endpoint="messages",
    model="claude-3-5-sonnet-20241022",
    top_k=40,
)

# Gemini: candidate_count
gemini_model = iModel(
    provider="gemini",
    model="gemini-2.0-flash-exp",
    candidate_count=1,
)
```

### Using Environment Variables for Configuration

```python
import os
from lionpride import Session
from lionpride.services import iModel

# API key automatically loaded from environment
model = iModel(
    provider="openai",
    model=os.getenv("PREFERRED_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("TEMPERATURE", "0.7")),
)
```

### Fallback Chain (Provider Resilience)

```python
async def chat_with_fallback(question: str):
    """Try multiple providers until one succeeds"""
    session = Session()

    providers = [
        ("openai", "gpt-4o-mini", {}),
        ("anthropic", "claude-3-5-haiku-20241022", {"endpoint": "messages"}),
        ("gemini", "gemini-2.0-flash-exp", {}),
    ]

    for provider, model_name, kwargs in providers:
        try:
            model = iModel(provider=provider, model=model_name, **kwargs)
            session.services.register(model)
            branch = session.create_branch(name=f"{provider}-branch")

            result = await communicate(
                session=session,
                branch=branch,
                parameters={
                    "instruction": question,
                    "imodel": model.name,
                }
            )

            print(f"✓ Success with {provider}")
            return result

        except Exception as e:
            print(f"✗ {provider} failed: {e}")
            continue

    raise RuntimeError("All providers failed")
```

## Common Pitfalls

1. **Wrong endpoint for Anthropic**

   ```python
   # ❌ Wrong - will fail
   iModel(provider="anthropic", model="claude-3-5-sonnet-20241022")

   # ✅ Correct - specify messages endpoint
   iModel(provider="anthropic", endpoint="messages", model="claude-3-5-sonnet-20241022")
   ```

2. **Missing API key**

   ```python
   # Ensure environment variable is set
   import os
   if not os.getenv("OPENAI_API_KEY"):
       raise ValueError("OPENAI_API_KEY not set")
   ```

3. **Provider-specific model names**

   ```python
   # Each provider has different model names
   # OpenAI: gpt-4o, gpt-4o-mini
   # Anthropic: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
   # Gemini: gemini-2.0-flash-exp, gemini-1.5-pro
   ```

## Next Steps

- **Structured outputs**: See [Structured Outputs](structured_outputs.md)
- **Multi-turn conversations**: See [Multi-Turn Chat](multi_turn.md)
- **Tool calling**: See [Tool Calling](tool_calling.md)

## See Also

- [API Reference: iModel](../api/services.md)
- [API Reference: Providers](../api/providers.md)
- [Streaming Responses](streaming.md)
