# Services

> Unified interface for LLM providers, tools, and service management

## Overview

The `services` module provides a unified abstraction layer for interacting with LLM
providers (OpenAI, Anthropic, Google Gemini, Claude Code) and local tools. It handles
the complexity of different API formats, rate limiting, and lifecycle hooks through a
consistent interface.

**Key Capabilities:**

- **iModel**: Unified service interface wrapping backends with rate limiting and hooks
- **Tool**: Wraps callable functions for LLM tool use with schema generation
- **ServiceRegistry**: Pile-based collection with O(1) name lookup for service
  management
- **ServiceBackend**: Base abstraction for Endpoint (HTTP APIs) and Tool (local
  functions)

The module follows lionagi v0's proven patterns for rate limiting and event-driven
processing. `iModel` provides the same `invoke()` interface regardless of whether it
wraps an HTTP endpoint or a local function, enabling seamless provider switching and
tool integration.

## Classes

### Core

| Class                          | Description                                            |
| ------------------------------ | ------------------------------------------------------ |
| [iModel](imodel.md)            | Unified service interface with rate limiting and hooks |
| [Tool](tool.md)                | Function wrapper for LLM tool use                      |
| [ServiceRegistry](registry.md) | Pile-based service collection with name lookup         |

### Backends

| Class                                               | Description                                   |
| --------------------------------------------------- | --------------------------------------------- |
| [ServiceBackend](backend.md)                        | Base class for all service backends           |
| [ServiceConfig](backend.md#serviceconfig)           | Configuration container for backends          |
| [NormalizedResponse](backend.md#normalizedresponse) | Generic normalized response                   |
| [Calling](backend.md#calling)                       | Base event for service invocations with hooks |
| `Endpoint`                                          | HTTP API backend for LLM providers            |
| `APICalling`                                        | API call event with request/response tracking |
| `ToolCalling`                                       | Tool execution event                          |

### Utilities

| Class                                             | Description                                                               |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| [`RateLimitConfig`](utilities.md#ratelimitconfig) | Token bucket rate limiting configuration                                  |
| [`TokenBucket`](utilities.md#tokenbucket)         | Async token bucket rate limiter                                           |
| [`CircuitBreaker`](utilities.md#circuitbreaker)   | Fail-fast circuit breaker for resilience                                  |
| [`RetryConfig`](utilities.md#retryconfig)         | Exponential backoff retry configuration                                   |
| [`TokenCalculator`](utilities.md#tokencalculator) | Tiktoken-based token counting                                             |
| `RateLimitedExecutor`                             | Event-driven rate limiting processor                                      |
| `HookRegistry`                                    | Lifecycle hook management (PreEventCreate, PreInvocation, PostInvocation) |

See [Service Utilities](utilities.md) for detailed API documentation.

### Provider Support

| Provider      | Endpoint           | Description                           |
| ------------- | ------------------ | ------------------------------------- |
| `openai`      | `chat/completions` | OpenAI GPT models                     |
| `anthropic`   | `messages`         | Anthropic Claude models               |
| `gemini`      | `generateContent`  | Google Gemini models                  |
| `claude_code` | CLI                | Claude Code with session continuation |

## Quick Start

```python
from lionpride import iModel, Tool, ServiceRegistry

# Create model with provider shorthand
model = iModel(
    provider="openai",
    model="gpt-4o-mini",
    limit_requests=60,     # Rate limit: 60 RPM
    limit_tokens=100000,   # Rate limit: 100k TPM
)

# Invoke the model
calling = await model.invoke(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(calling.response.data["choices"][0]["message"]["content"])

# Create and register a tool
def search(query: str, limit: int = 10) -> list[dict]:
    """Search for items matching query."""
    return [{"id": 1, "name": "Result"}]

tool = Tool(func_callable=search)

# Use ServiceRegistry for multi-model management
registry = ServiceRegistry()
registry.register(model)
registry.register(tool)

# Lookup by name
model = registry.get("gpt-4o-mini")
tool = registry.get("search")
```

## See Also

- [Session](../session/index.md) - Session uses ServiceRegistry for model management
- [Operations](../operations/index.md) - Operations use iModel for LLM calls
- [Base Classes](../base/element.md) - iModel inherits from Element
- [User Guide: Services](../../user_guide/services.md) - Tutorial and patterns
