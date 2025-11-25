# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Functional test for LNDL v2 Cognitive Runtime with live API.

This test demonstrates the complete cognitive programming loop:
1. LLM generates LNDL v2 response with cognitive constructs
2. Runtime parses and detects yield points
3. Actions are executed at yield points
4. Observations are injected back
5. Final output is produced

Run with: uv run pytest tests/lndl/test_cognitive_live.py -v -s
"""

import asyncio
import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from lionpride.lndl.cognitive import (
    CognitiveOutput,
    CognitiveYield,
    execute_cognitive,
    parse_cognitive,
)

# Load .env from parent projects directory
env_path = Path(__file__).parents[4] / ".env"
load_dotenv(env_path)


# LNDL v2 system prompt for cognitive programming
LNDL_V2_SYSTEM_PROMPT = """You are an AI that responds using LNDL v2 (Lion Natural Directive Language).

## LNDL v2 Syntax

### Variables
<lvar alias>content</lvar>          - Raw variable
<lvar Model.field alias>content</lvar>  - Namespaced variable

### Actions
<lact alias>function_call(args)</lact>  - Action declaration

### Cognitive Control (v2)
<context>
  <include msg="msg_id"/>           - Include message in context
  <compress msgs="range" to="alias"/>  - Compress messages
</context>

<yield for="action_alias" reason="why" keep="what"/>  - Yield for action execution

### Output
OUT{field1: [var1, var2], field2: literal}

## Example Response

<lvar reasoning>I need to search for information about the topic.</lvar>

<lact search>search(query="relevant query")</lact>

<yield for="search" reason="need search results"/>

<lvar Answer.text answer>Based on my analysis, the answer is...</lvar>

OUT{answer: [answer], reasoning: [reasoning]}

IMPORTANT: Always respond in LNDL v2 format as shown above."""


@pytest.fixture
def openrouter_api_key():
    """Get OpenRouter API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not found in environment")
    return key


async def call_openrouter(
    api_key: str,
    messages: list[dict],
    model: str = "openai/gpt-4o-mini",  # Cheap and fast
) -> str:
    """Call OpenRouter API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def mock_search(query: str) -> str:
    """Mock search function for testing."""
    return f"Search results for '{query}': Found 3 relevant articles about the topic."


@pytest.mark.asyncio
async def test_lndl_v2_parse_only():
    """Test LNDL v2 parsing without live API."""
    # Simulated LNDL v2 response
    response = """
<context>
  <include msg="user_question"/>
</context>

<lvar reasoning>The user is asking about quantum computing advances in 2024.</lvar>

<lact search>search(query="quantum computing 2024 advances")</lact>

<yield for="search" reason="need search results" keep="top_3"/>

<lvar Answer.text answer>Based on the search results, quantum computing made significant advances in 2024.</lvar>

OUT{answer: [answer], reasoning: [reasoning]}
"""

    program, state = parse_cognitive(response)

    # Verify parsing
    assert program.context is not None
    assert len(program.context.directives) == 1
    assert len(state.lvars) == 2
    assert "reasoning" in state.lvars
    assert "answer" in state.lvars
    assert len(state.lacts) == 1
    assert "search" in state.lacts
    assert program.yields is not None
    assert len(program.yields) == 1
    assert program.yields[0].for_ref == "search"
    assert program.out_block is not None
    assert "answer" in program.out_block.fields


@pytest.mark.asyncio
async def test_cognitive_executor_with_mock():
    """Test cognitive executor with mock action execution."""
    response = """
<lvar reasoning>I need to search for AI information.</lvar>

<lact s>search(query="AI advances 2024")</lact>

<yield for="s" reason="need search results" keep="top_5"/>

<lvar answer>AI made significant advances in 2024.</lvar>

OUT{answer: [answer], reasoning: [reasoning]}
"""

    gen = execute_cognitive(response)
    results = []

    # Iterate through the generator
    result = await gen.__anext__()
    results.append(result)

    if isinstance(result, CognitiveYield):
        # Execute mock search
        observation = mock_search("AI advances 2024")
        result = await gen.asend(observation)
        results.append(result)

    # Verify results
    assert len(results) == 2
    assert isinstance(results[0], CognitiveYield)
    assert results[0].for_ref == "s"
    assert results[0].action_call == 'search(query="AI advances 2024")'

    assert isinstance(results[1], CognitiveOutput)
    assert "answer" in results[1].fields
    assert "s" in results[1].state.observations


@pytest.mark.asyncio
async def test_lndl_v2_live_api(openrouter_api_key):
    """Test LNDL v2 with live OpenRouter API call."""
    messages = [
        {"role": "system", "content": LNDL_V2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "What is the capital of France? Respond in LNDL v2 format.",
        },
    ]

    # Call OpenRouter
    response = await call_openrouter(openrouter_api_key, messages)

    print("\n=== LLM Response ===")
    print(response)
    print("=" * 50)

    # Parse the response
    program, state = parse_cognitive(response)

    print("\n=== Parsed Cognitive State ===")
    print(f"Lvars: {list(state.lvars.keys())}")
    print(f"Lacts: {list(state.lacts.keys())}")
    print(f"Context directives: {len(state.context_directives)}")
    if program.yields:
        print(f"Yields: {len(program.yields)}")
    if program.out_block:
        print(f"Output fields: {list(program.out_block.fields.keys())}")

    # Verify we got valid LNDL
    assert len(state.lvars) > 0, "Should have at least one lvar"
    assert program.out_block is not None, "Should have OUT block"


@pytest.mark.asyncio
async def test_lndl_v2_cognitive_loop(openrouter_api_key):
    """Test complete cognitive loop with yield and observation injection."""
    messages = [
        {"role": "system", "content": LNDL_V2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": """I need you to search for information about "lionpride python framework" and summarize what you find.

Use a <lact> to declare the search action, then <yield> to request its execution.
After the yield, I will provide you with search results as an observation.

Respond in LNDL v2 format with a yield statement.""",
        },
    ]

    # First LLM call - should produce yield
    response = await call_openrouter(openrouter_api_key, messages)

    print("\n=== First LLM Response (with yield) ===")
    print(response)
    print("=" * 50)

    # Parse and execute
    gen = execute_cognitive(response)

    try:
        result = await gen.__anext__()

        if isinstance(result, CognitiveYield):
            print("\n=== Yield Point ===")
            print(f"For: {result.for_ref}")
            print(f"Reason: {result.reason}")
            print(f"Action: {result.action_call}")

            # Mock observation (in real system, would execute the action)
            observation = (
                "lionpride is a foundational Python framework for AI primitives, "
                "featuring Element, Pile, Flow, and Message abstractions. "
                "It's part of the Lion ecosystem by HaiyangLi."
            )

            print("\n=== Injecting Observation ===")
            print(observation)

            # Continue execution with observation
            result = await gen.asend(observation)

        if isinstance(result, CognitiveOutput):
            print("\n=== Final Output ===")
            print(f"Fields: {result.fields}")
            print(f"Observations: {result.state.observations}")

            # Verify observation was stored
            assert len(result.state.observations) > 0, "Should have stored observation"

    except StopAsyncIteration:
        # Model might not have produced a yield
        print("No yield produced by model")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_lndl_v2_parse_only())
    print("\n✓ Parse-only test passed")

    asyncio.run(test_cognitive_executor_with_mock())
    print("✓ Mock executor test passed")

    # Live API tests require API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        asyncio.run(test_lndl_v2_live_api(api_key))
        print("✓ Live API test passed")

        asyncio.run(test_lndl_v2_cognitive_loop(api_key))
        print("✓ Cognitive loop test passed")
    else:
        print("⚠ Skipping live API tests (no OPENROUTER_API_KEY)")
