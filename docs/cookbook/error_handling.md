# Error Handling and Resilience Patterns

## Overview

This recipe demonstrates robust error handling, retry strategies, fallback mechanisms, and resilience patterns in lionpride. Build production-ready applications that gracefully handle failures.

## Prerequisites

```bash
pip install lionpride pydantic tenacity
```

## The Code

### Example 1: Basic Error Handling

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def basic_error_handling():
    """Handle common errors in LLM operations"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    branch = session.create_branch(name="error-demo")

    try:
        result = await communicate(
            session=session,
            branch=branch,
            parameters={
                "instruction": "Write a poem about coding",
                "imodel": model.name,
            }
        )
        print(f"Success: {result}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        # Invalid parameters, missing required fields

    except RuntimeError as e:
        print(f"Execution error: {e}")
        # API call failed, timeout, rate limit

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Catch-all for other errors

asyncio.run(basic_error_handling())
```

### Example 2: Retry with Exponential Backoff

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RuntimeError),
)
async def communicate_with_retry(session, branch, parameters):
    """Retry operation with exponential backoff"""
    return await communicate(
        session=session,
        branch=branch,
        parameters=parameters,
    )

async def retry_pattern():
    """Use retry logic for transient failures"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    branch = session.create_branch(name="retry")

    try:
        result = await communicate_with_retry(
            session=session,
            branch=branch,
            parameters={
                "instruction": "Explain retry patterns",
                "imodel": model.name,
            }
        )
        print(f"Success after retries: {result}")

    except Exception as e:
        print(f"Failed after all retries: {e}")

asyncio.run(retry_pattern())
```

### Example 3: Fallback Chain

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def fallback_chain():
    """Try multiple providers until one succeeds"""
    session = Session()

    # Register multiple models as fallbacks
    providers = [
        iModel(provider="openai", model="gpt-4o-mini", temperature=0, name="primary"),
        iModel(provider="anthropic", endpoint="messages",
               model="claude-3-5-haiku-20241022", temperature=0, name="fallback1"),
        iModel(provider="gemini", model="gemini-2.0-flash-exp",
               temperature=0, name="fallback2"),
    ]

    for model in providers:
        session.services.register(model)

    branch = session.create_branch(name="fallback")
    question = "What is machine learning?"

    # Try each provider in order
    for i, model in enumerate(providers, 1):
        try:
            print(f"Attempt {i}: Trying {model.name}...")

            result = await communicate(
                session=session,
                branch=branch,
                parameters={
                    "instruction": question,
                    "imodel": model.name,
                }
            )

            print(f"✓ Success with {model.name}")
            print(f"Result: {result[:100]}...")
            return result

        except Exception as e:
            print(f"✗ {model.name} failed: {e}")
            if i == len(providers):
                print("All providers failed!")
                raise
            print(f"Falling back to next provider...\n")

asyncio.run(fallback_chain())
```

### Example 4: Validation and Recovery

```python
import asyncio
from pydantic import BaseModel, Field, ValidationError
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

class ExtractedData(BaseModel):
    """Structured data with validation"""
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

async def validate_and_recover():
    """Validate outputs and retry with corrections"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    branch = session.create_branch(name="validation")

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}:")

        try:
            result = await communicate(
                session=session,
                branch=branch,
                parameters={
                    "instruction": (
                        "Extract person info: 'John Doe, 32 years old, "
                        "email: john.doe@example.com'"
                    ),
                    "imodel": model.name,
                    "response_model": ExtractedData,
                    "return_as": "model",
                }
            )

            print(f"✓ Validation passed!")
            print(f"Name: {result.name}")
            print(f"Age: {result.age}")
            print(f"Email: {result.email}")
            return result

        except ValidationError as e:
            print(f"✗ Validation failed: {e}")

            if attempt < max_attempts:
                # Add corrective instruction
                error_details = str(e)
                await communicate(
                    session=session,
                    branch=branch,
                    parameters={
                        "instruction": (
                            f"The previous extraction had errors: {error_details}. "
                            "Please correct and extract again."
                        ),
                        "imodel": model.name,
                    }
                )
            else:
                print("Max attempts reached!")
                raise

asyncio.run(validate_and_recover())
```

### Example 5: Circuit Breaker Pattern

```python
import asyncio
from datetime import datetime, timedelta
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""

    def __init__(self, failure_threshold=3, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func):
        """Wrap function with circuit breaker logic"""
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                # Check if timeout has passed
                if (datetime.now() - self.last_failure_time).seconds > self.timeout:
                    print("Circuit breaker: Transitioning to half-open")
                    self.state = "half-open"
                else:
                    raise RuntimeError("Circuit breaker is OPEN - service unavailable")

            try:
                result = await func(*args, **kwargs)

                # Success - reset if half-open
                if self.state == "half-open":
                    print("Circuit breaker: Transitioning to closed")
                    self.state = "closed"
                    self.failures = 0

                return result

            except Exception as e:
                self.failures += 1
                self.last_failure_time = datetime.now()

                if self.failures >= self.failure_threshold:
                    print(f"Circuit breaker: OPENING after {self.failures} failures")
                    self.state = "open"

                raise

        return wrapper

async def circuit_breaker_demo():
    """Use circuit breaker to protect against repeated failures"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    breaker = CircuitBreaker(failure_threshold=3, timeout=5)

    @breaker.call
    async def protected_communicate(instruction):
        branch = session.create_branch(name=f"attempt-{datetime.now()}")
        return await communicate(
            session=session,
            branch=branch,
            parameters={
                "instruction": instruction,
                "imodel": model.name,
            }
        )

    # Simulate failures
    print("Testing circuit breaker:\n")

    for i in range(5):
        try:
            if i < 3:
                # Simulate failures by passing invalid model
                result = await protected_communicate("Test question")
            else:
                # After circuit opens, calls are rejected immediately
                result = await protected_communicate("Test question")

            print(f"Attempt {i+1}: ✓ Success")

        except Exception as e:
            print(f"Attempt {i+1}: ✗ {e}")

        await asyncio.sleep(1)

    # Wait for circuit to half-open
    print(f"\nWaiting {breaker.timeout} seconds for circuit to half-open...")
    await asyncio.sleep(breaker.timeout + 1)

    # Try again
    try:
        result = await protected_communicate("Test question after timeout")
        print(f"After timeout: ✓ Success")
    except Exception as e:
        print(f"After timeout: ✗ {e}")

asyncio.run(circuit_breaker_demo())
```

### Example 6: Timeout Handling

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def timeout_handling():
    """Handle operation timeouts"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0.7)
    session.services.register(model)

    branch = session.create_branch(name="timeout")

    try:
        # Set timeout of 5 seconds
        result = await asyncio.wait_for(
            communicate(
                session=session,
                branch=branch,
                parameters={
                    "instruction": "Write a very long essay about the history of computing",
                    "imodel": model.name,
                    "max_tokens": 4000,  # Long response
                }
            ),
            timeout=5.0  # 5 second timeout
        )
        print(f"Completed within timeout: {result[:100]}...")

    except asyncio.TimeoutError:
        print("Operation timed out after 5 seconds")

        # Option 1: Cancel and retry with shorter response
        print("Retrying with shorter max_tokens...")
        result = await asyncio.wait_for(
            communicate(
                session=session,
                branch=branch,
                parameters={
                    "instruction": "Write a brief summary about the history of computing",
                    "imodel": model.name,
                    "max_tokens": 200,  # Much shorter
                }
            ),
            timeout=10.0
        )
        print(f"Completed: {result}")

asyncio.run(timeout_handling())
```

### Example 7: Rate Limit Handling

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def rate_limit_handling():
    """Handle rate limits gracefully"""
    session = Session()

    # Use built-in rate limiting
    model = iModel(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0,
        limit_requests=10,      # Max 10 requests per minute
        limit_tokens=10000,     # Max 10k tokens per minute
        capacity_refresh_time=60,
    )
    session.services.register(model)

    # Send multiple requests
    tasks = []
    for i in range(20):  # More than rate limit
        branch = session.create_branch(name=f"request-{i}")
        task = communicate(
            session=session,
            branch=branch,
            parameters={
                "instruction": f"Question {i}: What is 2+2?",
                "imodel": model.name,
            }
        )
        tasks.append(task)

    # Execute with rate limiting (automatically throttled)
    print(f"Sending {len(tasks)} requests with rate limit of 10/minute...")

    results = []
    for i, task in enumerate(tasks):
        try:
            result = await task
            print(f"Request {i+1}: ✓ {result}")
            results.append(result)
        except Exception as e:
            print(f"Request {i+1}: ✗ {e}")

    print(f"\nCompleted {len(results)}/{len(tasks)} requests")

asyncio.run(rate_limit_handling())
```

## Expected Output

### Example 3 (Fallback)

```
Attempt 1: Trying primary...
✗ primary failed: Connection timeout

Falling back to next provider...

Attempt 2: Trying fallback1...
✓ Success with fallback1
Result: Machine learning is a subset of artificial intelligence that enables systems to learn...
```

### Example 5 (Circuit Breaker)

```
Testing circuit breaker:

Attempt 1: ✗ API Error
Attempt 2: ✗ API Error
Attempt 3: ✗ API Error
Circuit breaker: OPENING after 3 failures
Attempt 4: ✗ Circuit breaker is OPEN - service unavailable
Attempt 5: ✗ Circuit breaker is OPEN - service unavailable

Waiting 5 seconds for circuit to half-open...
Circuit breaker: Transitioning to half-open
After timeout: ✓ Success
Circuit breaker: Transitioning to closed
```

## Key Patterns

### Error Hierarchy

```
Exception
├── ValueError          # Configuration errors (invalid parameters)
├── RuntimeError        # Execution errors (API failures, timeouts)
├── ValidationError     # Pydantic validation failures
└── TimeoutError        # Operation timeouts
```

### Retry Decision Tree

```
Error occurs
├── Transient? (network, rate limit, timeout)
│   └── Retry with backoff
├── Configuration? (invalid params)
│   └── Fix and retry
├── Validation? (bad output)
│   └── Retry with correction
└── Permanent? (auth, quota)
    └── Fail fast
```

## Best Practices

### 1. Fail Fast for Configuration Errors

```python
# Validate early
if not session.services.has(model_name):
    raise ValueError(f"Model {model_name} not registered")

# Don't retry configuration errors
```

### 2. Retry Transient Errors

```python
# Network errors, rate limits, timeouts
retry_exceptions = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

@retry(retry=retry_if_exception_type(retry_exceptions))
async def resilient_call():
    ...
```

### 3. Log Failures

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = await communicate(...)
except Exception as e:
    logger.error(f"Communication failed: {e}", exc_info=True)
    raise
```

### 4. Use Timeouts

```python
# Always set reasonable timeouts
result = await asyncio.wait_for(
    long_operation(),
    timeout=30.0  # 30 seconds
)
```

### 5. Graceful Degradation

```python
# Provide fallback responses
try:
    result = await ai_analysis()
except Exception:
    result = "Analysis unavailable. Please try again later."
```

## Common Pitfalls

1. **Catching too broad**

   ```python
   # ❌ Wrong - masks real issues
   try:
       result = await communicate(...)
   except:  # Catches everything!
       pass

   # ✅ Right - specific exceptions
   try:
       result = await communicate(...)
   except (ValueError, RuntimeError) as e:
       handle_error(e)
   ```

2. **Infinite retries**

   ```python
   # ❌ Wrong - no limit
   while True:
       try:
           return await communicate(...)
       except:
           continue  # Forever!

   # ✅ Right - max attempts
   for attempt in range(max_retries):
       try:
           return await communicate(...)
       except:
           if attempt == max_retries - 1:
               raise
   ```

3. **Not handling async errors**

   ```python
   # ❌ Wrong - async error not awaited
   try:
       task = communicate(...)  # Not awaited!
   except Exception:
       pass  # Won't catch

   # ✅ Right - await first
   try:
       result = await communicate(...)
   except Exception:
       handle_error()
   ```

## Next Steps

- **Production deployment**: See [Deployment Guide](../user_guide/deployment.md)
- **Monitoring and observability**: See [Monitoring Guide](../user_guide/monitoring.md)
- **Testing strategies**: See [Testing Guide](../user_guide/testing.md)

## See Also

- [API Reference: Error Handling](../api/errors.md)
- [Resilience Patterns](../patterns/resilience.md)
- [Production Best Practices](../user_guide/production.md)
