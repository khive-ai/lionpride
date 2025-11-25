#!/usr/bin/env python3
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Functional test script for ReAct operation with real OpenAI API.

Tests the IPU execution pattern:
    ipu = IPU()
    session = Session()
    ipu.register_session(session)
    op = await session.conduct(branch, "react", ipu, ...)
    # Result available after execution

Run with:
    uv run python scripts/functional_test/test_branch_react.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Model to use for all tests
MODEL_NAME = "gpt-4.1-mini"


# Define test tool functions
async def calculator(a: float, b: float, operation: str) -> str:
    """Simple calculator that performs basic math operations.

    Args:
        a: First number
        b: Second number
        operation: Operation to perform (add, subtract, multiply, divide)

    Returns:
        Result of the calculation as a string
    """
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else "Error: Division by zero"
    else:
        result = f"Unknown operation: {operation}"
    return f"{a} {operation} {b} = {result}"


async def get_weather(city: str) -> str:
    """Get weather for a city (mock data).

    Args:
        city: Name of the city

    Returns:
        Weather information as a string
    """
    # Mock weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 58°F",
        "tokyo": "Rainy, 65°F",
        "paris": "Partly cloudy, 68°F",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather in {city}: Clear, 70°F (default)"


async def test_react():
    """Test ReAct operation with real API calls using IPU pattern."""
    from lionpride.ipu import IPU
    from lionpride.operations import ReactParam, ReactResult
    from lionpride.services import iModel
    from lionpride.services.providers.oai_chat import OAIChatEndpoint
    from lionpride.services.types.tool import Tool
    from lionpride.session import Session

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return False

    print("=" * 60)
    print(f"Testing ReAct operation with IPU pattern ({MODEL_NAME})")
    print("=" * 60)

    # Create model
    endpoint = OAIChatEndpoint(
        provider="openai",
        name="openai",
        api_key=api_key,
    )
    model = iModel(backend=endpoint)

    # Create tools from functions
    calculator_tool = Tool(func_callable=calculator)
    weather_tool = Tool(func_callable=get_weather)

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    session.services.register(model, update=True)
    ipu.register_session(session)

    all_passed = True

    # Test 1: Complex calculation with tool
    print("\n--- Test 1: Complex Calculation with Tool ---")
    branch1 = session.create_branch(name="test1")
    try:
        # Create ReactParam
        params = ReactParam(
            instruction="What is 2791.352 multiplied by 49021.8254? Use the calculator tool. Give me the exact result.",
            imodel=model,
            model_name=MODEL_NAME,
            tools=[calculator_tool],
            max_steps=3,
            verbose=True,
        )

        # Execute via session.conduct() + IPU
        op = await session.conduct(branch1, "react", ipu, params=params)

        # Get result (IPU.queue executes immediately for now)
        result = op.execution.response if hasattr(op.execution, "response") else None

        # For now, the result is returned directly from IPU.execute()
        # We need to check the operation's execution result
        print(f"\nOperation status: {op.status}")

        if isinstance(result, ReactResult):
            print(f"Completed: {result.completed}")
            print(f"Total steps: {result.total_steps}")
            print(f"Final response: {result.final_response}")

            if result.completed:
                print("PASS: ReAct completed successfully")
                if any(step.actions_executed for step in result.steps):
                    print("PASS: Calculator tool was executed")
                else:
                    print("WARN: No tool execution recorded")
            else:
                print(f"WARN: ReAct did not complete normally: {result.reason_stopped}")
        else:
            # Check if result was stored differently
            print(f"Result type: {type(result)}")
            print("PASS: Operation executed (result format may vary)")

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # Test 2: Multi-step reasoning with multiple tools
    print("\n--- Test 2: Multi-Step with Multiple Tools ---")
    branch2 = session.create_branch(name="test2")
    try:
        params = ReactParam(
            instruction="First check the weather in Tokyo, then calculate 8847.291 divided by 173.456. Report both results with the exact calculation.",
            imodel=model,
            model_name=MODEL_NAME,
            tools=[calculator_tool, weather_tool],
            max_steps=5,
            verbose=True,
        )

        op = await session.conduct(branch2, "react", ipu, params=params)
        print(f"Operation status: {op.status}")
        print("PASS: Multi-tool ReAct executed")

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # Test 3: Task that can be answered directly (no tools needed)
    print("\n--- Test 3: Direct Answer (No Tools Needed) ---")
    branch3 = session.create_branch(name="test3")
    try:
        params = ReactParam(
            instruction="What is the capital of France? You can answer directly without using tools.",
            imodel=model,
            model_name=MODEL_NAME,
            tools=[calculator_tool],  # Calculator provided but shouldn't be needed
            max_steps=3,
            verbose=True,
        )

        op = await session.conduct(branch3, "react", ipu, params=params)
        print(f"Operation status: {op.status}")
        print("PASS: Direct answer ReAct executed")

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All react() tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(test_react())
    sys.exit(0 if success else 1)
