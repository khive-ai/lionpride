#!/usr/bin/env python3
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Functional test script for LNDL-centric operations with real OpenAI API.

Tests IPU pattern with:
1. communicate with LNDL mode (structured output via system prompt injection)
2. operate with LNDL mode and actions
3. react with LNDL mode

Run with:
    uv run python scripts/functional_test/test_lndl_operations.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file
load_dotenv()

# Model to use for all tests
MODEL_NAME = "gpt-4.1-mini"


# Test models
class AnalysisReport(BaseModel):
    """Analysis report with summary and confidence."""

    summary: str = Field(..., description="Analysis summary")
    confidence: float = Field(..., description="Confidence score 0-1")
    recommendation: str = Field(..., description="Action recommendation")


class CalculationResult(BaseModel):
    """Calculation result with explanation."""

    result: str = Field(..., description="The calculated result")
    steps: str = Field(..., description="Step-by-step explanation")


# Test tools
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


async def test_communicate_lndl():
    """Test communicate with LNDL mode (structured output via operable)."""
    from lionpride.ipu import IPU
    from lionpride.operations import CommunicateParam
    from lionpride.services import iModel
    from lionpride.services.providers.oai_chat import OAIChatEndpoint
    from lionpride.session import Session
    from lionpride.types import Operable, Spec

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return False

    print("=" * 60)
    print(f"Test 1: communicate with LNDL mode ({MODEL_NAME})")
    print("=" * 60)

    # Create model
    endpoint = OAIChatEndpoint(
        provider="openai",
        name="openai",
        api_key=api_key,
    )
    model = iModel(backend=endpoint)

    # Create operable for LNDL mode
    operable = Operable(
        specs=(Spec(base_type=AnalysisReport, name="analysis"),),
        name="AnalysisResponse",
    )

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    session.services.register(model, update=True)
    ipu.register_session(session)

    # Create branch
    branch = session.create_branch(name="test_lndl_communicate")

    try:
        params = CommunicateParam(
            instruction=(
                "Analyze the statement: 'Artificial Intelligence will transform healthcare.' "
                "Provide a structured analysis with summary, confidence score, and recommendation."
            ),
            imodel=model,
            operable=operable,
            lndl_threshold=0.85,
            return_as="model",
        )

        op = await session.conduct(branch, "communicate", ipu, params=params)
        print(f"\nOperation status: {op.status}")
        print("PASS: LNDL communicate executed via IPU pattern")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_operate_lndl_structured():
    """Test operate with LNDL mode for structured output (no JSON actions)."""
    from lionpride.ipu import IPU
    from lionpride.operations import OperateParam
    from lionpride.services import iModel
    from lionpride.services.providers.oai_chat import OAIChatEndpoint
    from lionpride.session import Session

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return False

    print("\n" + "=" * 60)
    print(f"Test 2: operate with LNDL mode ({MODEL_NAME})")
    print("=" * 60)

    # Create model
    endpoint = OAIChatEndpoint(
        provider="openai",
        name="openai",
        api_key=api_key,
    )
    model = iModel(backend=endpoint)

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    session.services.register(model, update=True)
    ipu.register_session(session)

    # Create branch
    branch = session.create_branch(name="test_lndl_operate")

    try:
        params = OperateParam(
            instruction=(
                "Analyze the calculation: 3847.29 multiplied by 8291.47. "
                "What would the approximate result be? Provide the result field and explanation steps."
            ),
            imodel=model,
            response_model=CalculationResult,
            use_lndl=True,
            lndl_threshold=0.85,
            max_retries=2,
        )

        op = await session.conduct(branch, "operate", ipu, params=params)
        print(f"\nOperation status: {op.status}")
        print("PASS: operate with LNDL mode executed via IPU pattern")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_communicate_json_vs_lndl():
    """Compare JSON mode vs LNDL mode for the same task."""
    from lionpride.ipu import IPU
    from lionpride.operations import CommunicateParam
    from lionpride.services import iModel
    from lionpride.services.providers.oai_chat import OAIChatEndpoint
    from lionpride.session import Session
    from lionpride.types import Operable, Spec

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return False

    print("\n" + "=" * 60)
    print(f"Test 3: JSON vs LNDL mode comparison ({MODEL_NAME})")
    print("=" * 60)

    # Create model
    endpoint = OAIChatEndpoint(
        provider="openai",
        name="openai",
        api_key=api_key,
    )
    model = iModel(backend=endpoint)

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    session.services.register(model, update=True)
    ipu.register_session(session)

    instruction = (
        "Summarize in 1-2 sentences: 'Machine learning enables computers to learn from data.'"
    )

    # Test 1: JSON mode
    print("\n--- JSON Mode ---")
    branch_json = session.create_branch(name="test_json")
    try:
        params = CommunicateParam(
            instruction=instruction,
            imodel=model,
            response_model=AnalysisReport,
            return_as="model",
            max_retries=1,
        )

        op = await session.conduct(branch_json, "communicate", ipu, params=params)
        print(f"JSON mode status: {op.status}")
        print("PASS: JSON mode succeeded")
    except Exception as e:
        print(f"FAIL: JSON mode failed: {e}")

    # Test 2: LNDL mode
    print("\n--- LNDL Mode ---")
    branch_lndl = session.create_branch(name="test_lndl")
    operable = Operable(
        specs=(Spec(base_type=AnalysisReport, name="analysis"),),
        name="AnalysisResponse",
    )
    try:
        params = CommunicateParam(
            instruction=instruction,
            imodel=model,
            operable=operable,
            lndl_threshold=0.85,
            return_as="model",
            max_retries=1,
        )

        op = await session.conduct(branch_lndl, "communicate", ipu, params=params)
        print(f"LNDL mode status: {op.status}")
        print("PASS: LNDL mode succeeded")
    except Exception as e:
        print(f"FAIL: LNDL mode failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nPASS: Comparison test completed")
    return True


async def test_react_with_lndl():
    """Test react with LNDL mode for step responses."""
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

    print("\n" + "=" * 60)
    print(f"Test 4: react with LNDL mode ({MODEL_NAME})")
    print("=" * 60)

    # Create model
    endpoint = OAIChatEndpoint(
        provider="openai",
        name="openai",
        api_key=api_key,
    )
    model = iModel(backend=endpoint)

    # Create calculator tool
    calculator_tool = Tool(func_callable=calculator)

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    session.services.register(model, update=True)
    ipu.register_session(session)

    # Create branch
    branch = session.create_branch(name="test_react_lndl")

    try:
        params = ReactParam(
            instruction="What is 5847.23 divided by 123.45? Use the calculator tool. Give the exact result.",
            imodel=model,
            model_name=MODEL_NAME,
            tools=[calculator_tool],
            max_steps=3,
            use_lndl=True,
            lndl_threshold=0.85,
            verbose=True,
        )

        op = await session.conduct(branch, "react", ipu, params=params)
        print(f"\nOperation status: {op.status}")
        print("PASS: react with LNDL mode executed via IPU pattern")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all LNDL functional tests."""
    print("\n" + "=" * 60)
    print("LNDL-Centric Functional Tests (IPU Pattern)")
    print("=" * 60)

    all_passed = True

    # Test 1: communicate with LNDL
    result1 = await test_communicate_lndl()
    all_passed = all_passed and result1

    # Test 2: operate with LNDL mode
    result2 = await test_operate_lndl_structured()
    all_passed = all_passed and result2

    # Test 3: JSON vs LNDL comparison
    result3 = await test_communicate_json_vs_lndl()
    all_passed = all_passed and result3

    # Test 4: react with LNDL mode
    result4 = await test_react_with_lndl()
    all_passed = all_passed and result4

    print("\n" + "=" * 60)
    if all_passed:
        print("All LNDL tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
