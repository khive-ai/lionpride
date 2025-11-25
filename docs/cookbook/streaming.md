# Streaming Responses

## Overview

This recipe demonstrates streaming responses from LLMs in real-time using lionpride. Stream individual tokens as they're generated, providing better UX for long responses and enabling progressive rendering.

## Prerequisites

```bash
pip install lionpride
```

## The Code

### Example 1: Basic Streaming Chat

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel

async def stream_chat():
    """Stream tokens as they're generated"""
    session = Session()
    model = iModel(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        stream=True  # Enable streaming
    )
    session.services.register(model)

    from lionpride.session import InstructionContent, Message

    # Create message
    message = Message(
        content=InstructionContent(
            instruction="Write a short story about a robot learning to paint"
        )
    )

    # Stream response
    print("Assistant: ", end="", flush=True)

    async for chunk in model.invoke(messages=[message.chat_msg]):
        if hasattr(chunk, 'data'):
            # Extract text from chunk
            text = chunk.data
            print(text, end="", flush=True)

    print()  # Newline at end

asyncio.run(stream_chat())
```

### Example 2: Streaming with Branch Context

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def stream_with_context():
    """Stream responses while maintaining conversation context"""
    session = Session()
    model = iModel(
        provider="anthropic",
        endpoint="messages",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        stream=True
    )
    session.services.register(model)

    branch = session.create_branch(name="streaming-chat")

    # Turn 1: Regular response (builds context)
    await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Tell me about async programming in Python",
            "imodel": model.name,
        }
    )

    # Turn 2: Streaming response (uses context)
    print("\nUser: Can you give me a practical example?")
    print("Assistant: ", end="", flush=True)

    # Get conversation history for streaming
    from lionpride.session.messages.utils import prepare_messages_for_chat

    messages = [session.messages[msg_id] for msg_id in branch.order]
    chat_messages = prepare_messages_for_chat(messages)

    # Add new instruction
    from lionpride.session import InstructionContent, Message

    new_message = Message(
        content=InstructionContent(
            instruction="Can you give me a practical example?"
        )
    )
    chat_messages.append(new_message.chat_msg)

    # Stream
    full_response = []
    async for chunk in model.invoke(messages=chat_messages):
        if hasattr(chunk, 'data'):
            text = chunk.data
            full_response.append(text)
            print(text, end="", flush=True)

    print("\n")

    # Save streamed response to branch
    from lionpride.session import AssistantResponseContent

    response_message = Message(
        content=AssistantResponseContent(
            assistant_response="".join(full_response)
        )
    )
    session.conversations.add_item(response_message, progressions=[branch.name])

    return session, branch

asyncio.run(stream_with_context())
```

### Example 3: Streaming with Progress Indicators

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel

async def stream_with_progress():
    """Show progress while streaming"""
    session = Session()
    model = iModel(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        stream=True
    )
    session.services.register(model)

    from lionpride.session import InstructionContent, Message

    message = Message(
        content=InstructionContent(
            instruction="Explain quantum entanglement in detail"
        )
    )

    print("Generating response", end="", flush=True)

    tokens_received = 0
    response_chunks = []

    async for chunk in model.invoke(messages=[message.chat_msg]):
        if hasattr(chunk, 'data'):
            tokens_received += 1
            response_chunks.append(chunk.data)

            # Show progress every 10 tokens
            if tokens_received % 10 == 0:
                print(".", end="", flush=True)

    print(f"\n\nReceived {tokens_received} token chunks\n")
    print("Response:")
    print("".join(response_chunks))

    return "".join(response_chunks)

asyncio.run(stream_with_progress())
```

### Example 4: Streaming Multiple Responses in Parallel

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.session import InstructionContent, Message

async def stream_one_model(model, instruction: str, label: str):
    """Stream from one model"""
    message = Message(content=InstructionContent(instruction=instruction))

    print(f"\n[{label}] Starting...")
    response = []

    async for chunk in model.invoke(messages=[message.chat_msg]):
        if hasattr(chunk, 'data'):
            response.append(chunk.data)
            # Show first few chunks
            if len(response) <= 5:
                print(f"[{label}] {chunk.data}", end="", flush=True)

    print(f"\n[{label}] Complete ({len(response)} chunks)")
    return "".join(response)

async def parallel_streaming():
    """Stream from multiple models simultaneously"""
    session = Session()

    gpt = iModel(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        stream=True,
        name="gpt"
    )

    claude = iModel(
        provider="anthropic",
        endpoint="messages",
        model="claude-3-5-haiku-20241022",
        temperature=0.7,
        stream=True,
        name="claude"
    )

    session.services.register(gpt)
    session.services.register(claude)

    question = "What is the future of AI?"

    # Stream from both models in parallel
    results = await asyncio.gather(
        stream_one_model(gpt, question, "GPT"),
        stream_one_model(claude, question, "Claude"),
    )

    print("\n" + "="*60)
    print("FINAL RESPONSES")
    print("="*60)

    print(f"\nGPT Response ({len(results[0])} chars):")
    print(results[0][:200] + "...")

    print(f"\nClaude Response ({len(results[1])} chars):")
    print(results[1][:200] + "...")

    return results

asyncio.run(parallel_streaming())
```

### Example 5: Streaming Workflow Results

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import Builder, flow_stream

async def streaming_workflow():
    """Stream results from multi-step workflow"""
    session = Session()
    model = iModel(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        stream=True
    )
    session.services.register(model)

    builder = Builder(session=session)

    # Build workflow
    research = builder.communicate(
        instruction="List 3 key benefits of exercise",
        imodel=model.name,
        branch="research",
    )

    article = builder.communicate(
        instruction="Write a short article using these benefits",
        imodel=model.name,
        branch="article",
        context_from=[research],
    )

    # Stream workflow execution
    print("Executing workflow...\n")

    async for operation_id, chunk in flow_stream(builder.operations):
        # Get operation details
        operation = next(
            (op for op in builder.operations if op.id == operation_id),
            None
        )

        if operation:
            branch_name = operation.parameters.get("branch", "unknown")
            print(f"[{branch_name}] {chunk}", end="", flush=True)

    print("\n\nWorkflow complete!")

asyncio.run(streaming_workflow())
```

## Expected Output

### Example 1 (Basic)

```
Assistant: In a small workshop filled with the scent of oil and metal, a robot
named ARU-7 discovered something unexpected...
[Text streams word by word in real-time]
```

### Example 2 (With Context)

```
User: Can you give me a practical example?
Assistant: Certainly! Here's a practical example using asyncio and aiohttp
to fetch multiple URLs concurrently...
[Response streams in real-time]
```

### Example 3 (Progress)

```
Generating response..........

Received 87 token chunks

Response:
Quantum entanglement is one of the most fascinating and counterintuitive
phenomena in quantum mechanics...
```

### Example 4 (Parallel)

```
[GPT] Starting...
[GPT] The future of AI is...

[Claude] Starting...
[Claude] AI's future trajectory...

[GPT] Complete (245 chunks)
[Claude] Complete (298 chunks)

============================================================
FINAL RESPONSES
============================================================

GPT Response (1456 chars):
The future of AI is poised for remarkable transformation...

Claude Response (1789 chars):
AI's future trajectory appears to be heading toward...
```

## Key Concepts

### Streaming Architecture

```
LLM Provider → HTTP SSE/Streaming → lionpride iModel → async iterator → Your code
```

### Stream Chunk Structure

```python
# Chunk object contains:
chunk.data          # Text content
chunk.metadata      # Provider-specific metadata
chunk.raw_response  # Original response object
```

### Stream vs Non-Stream

```python
# Non-streaming: Wait for complete response
model = iModel(stream=False)
response = await model.invoke(...)  # Returns complete response

# Streaming: Iterate over chunks
model = iModel(stream=True)
async for chunk in model.invoke(...):  # Yields chunks
    process(chunk)
```

## Variations

### Custom Chunk Processing

```python
async def process_stream_with_formatting(model, message):
    """Process stream chunks with custom formatting"""
    chunks = []
    word_count = 0

    async for chunk in model.invoke(messages=[message]):
        if hasattr(chunk, 'data'):
            text = chunk.data
            chunks.append(text)

            # Count words (approximate)
            word_count += len(text.split())

            # Custom formatting
            if text.strip().endswith(('.', '!', '?')):
                print(text + " ", end="", flush=True)  # Add space after sentences
            else:
                print(text, end="", flush=True)

    print(f"\n\nTotal words: ~{word_count}")
    return "".join(chunks)
```

### Buffered Streaming

```python
async def buffered_stream(model, message, buffer_size=10):
    """Buffer chunks before displaying"""
    buffer = []

    async for chunk in model.invoke(messages=[message]):
        if hasattr(chunk, 'data'):
            buffer.append(chunk.data)

            if len(buffer) >= buffer_size:
                # Flush buffer
                print("".join(buffer), end="", flush=True)
                buffer = []

    # Flush remaining
    if buffer:
        print("".join(buffer), end="", flush=True)
```

### Stream to File

```python
async def stream_to_file(model, message, filename="output.txt"):
    """Stream response directly to file"""
    with open(filename, "w") as f:
        async for chunk in model.invoke(messages=[message]):
            if hasattr(chunk, 'data'):
                f.write(chunk.data)
                f.flush()  # Ensure immediate write

    print(f"Response saved to {filename}")
```

### Error Handling in Streams

```python
async def safe_stream(model, message):
    """Stream with error handling"""
    chunks = []

    try:
        async for chunk in model.invoke(messages=[message]):
            if hasattr(chunk, 'data'):
                chunks.append(chunk.data)
                print(chunk.data, end="", flush=True)

    except Exception as e:
        print(f"\n\nStreaming error: {e}")
        print(f"Received {len(chunks)} chunks before error")
        return "".join(chunks)  # Return partial response

    return "".join(chunks)
```

## Provider Differences

### Streaming Support by Provider

| Provider | Streaming Support | Chunk Format |
|----------|------------------|--------------|
| OpenAI | ✓ Full | SSE with delta |
| Anthropic | ✓ Full | SSE with delta |
| Gemini | ✓ Full | SSE |
| Groq | ✓ Full | SSE with delta |

### Provider-Specific Metadata

```python
async for chunk in model.invoke(...):
    # OpenAI
    if chunk.metadata.get("finish_reason"):
        print(f"Finished: {chunk.metadata['finish_reason']}")

    # Anthropic
    if chunk.metadata.get("stop_reason"):
        print(f"Stopped: {chunk.metadata['stop_reason']}")
```

## Common Pitfalls

1. **Forgetting to set stream=True**

   ```python
   # ❌ Wrong - returns complete response, not stream
   model = iModel(provider="openai", model="gpt-4o-mini")
   async for chunk in model.invoke(...):  # TypeError!

   # ✅ Right - enable streaming
   model = iModel(provider="openai", model="gpt-4o-mini", stream=True)
   async for chunk in model.invoke(...):  # Works
   ```

2. **Not flushing print buffer**

   ```python
   # ❌ Wrong - output may be buffered
   print(chunk.data, end="")

   # ✅ Right - flush immediately
   print(chunk.data, end="", flush=True)
   ```

3. **Blocking operations in stream loop**

   ```python
   # ❌ Wrong - blocks streaming
   async for chunk in model.invoke(...):
       time.sleep(0.1)  # Don't block!
       print(chunk.data)

   # ✅ Right - use async sleep if needed
   async for chunk in model.invoke(...):
       await asyncio.sleep(0.1)  # Non-blocking
       print(chunk.data)
   ```

4. **Not handling incomplete chunks**

   ```python
   # ❌ Wrong - assumes complete words
   async for chunk in model.invoke(...):
       words = chunk.data.split()  # May split mid-word!

   # ✅ Right - accumulate and process complete units
   buffer = ""
   async for chunk in model.invoke(...):
       buffer += chunk.data
       # Process complete sentences/paragraphs
   ```

## Performance Considerations

### Streaming Benefits

- **Perceived latency**: User sees output immediately
- **Memory efficiency**: Process chunks without buffering entire response
- **Progressive rendering**: Update UI as data arrives

### Trade-offs

- **Complexity**: More code than simple await
- **Error handling**: Must handle mid-stream failures
- **State management**: Track partial responses

### When to Use Streaming

**Use streaming for:**

- Long responses (>100 tokens)
- Interactive applications
- Real-time feedback requirements
- Large-scale generation

**Skip streaming for:**

- Short responses (<50 tokens)
- Batch processing
- Responses needing validation before display
- Structured output (wait for complete JSON)

## Next Steps

- **Error handling in streams**: See [Error Handling](error_handling.md)
- **Building streaming UIs**: See [UI Integration Guide](../integration/ui.md)
- **Production deployment**: See [Deployment Guide](../user_guide/deployment.md)

## See Also

- [API Reference: iModel streaming](../api/services.md#streaming)
- [API Reference: flow_stream()](../api/operations.md#flow_stream)
- [Streaming Patterns](../patterns/streaming.md)
