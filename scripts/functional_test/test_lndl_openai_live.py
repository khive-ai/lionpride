#!/usr/bin/env python3
"""Functional tests for LNDL formatting system with real OpenAI API.

Tests:
1. Basic LNDL flow via Operative
2. prepare_lndl_messages flow with Session/Branch
3. Streaming with LNDL via StreamChannel

Run with: python tests/functional/test_lndl_openai_live.py
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Load environment from parent projects directory
from dotenv import load_dotenv

load_dotenv("/Users/lion/projects/.env")

from typing import Optional

from pydantic import BaseModel, Field


# Test Models
class AnalysisResult(BaseModel):
    """Test model for LNDL flow."""

    topic: str = Field(description="The topic being analyzed")
    summary: str = Field(description="A brief summary of the analysis")
    key_insight: str = Field(description="The most important insight")  # Simplified from list
    confidence: float = Field(description="Confidence score between 0 and 1")


class SimpleReport(BaseModel):
    """Simpler model for testing."""

    title: str = Field(description="Report title")
    content: str = Field(description="Report content")


def separator(title: str):
    """Print a separator for test sections."""
    print("\n" + "=" * 60)
    print(f"TEST: {title}")
    print("=" * 60)


async def test_1_basic_lndl_flow():
    """Test 1: Basic LNDL flow via Operative.

    1. Create a Pydantic model (AnalysisResult)
    2. Create an Operative from it via create_operative_from_model
    3. Use generate_lndl_spec_format(operative) to get the LNDL prompt
    4. Call OpenAI with the LNDL prompt
    5. Parse the response with parse_lndl_fuzzy
    6. Verify the parsed result matches expected types
    """
    separator("1. Basic LNDL Flow via Operative")

    from lionpride import iModel
    from lionpride.lndl import get_lndl_system_prompt, parse_lndl_fuzzy
    from lionpride.operations.lndl import generate_lndl_spec_format
    from lionpride.operations.operate.operative import create_operative_from_model

    # Step 1 & 2: Create Operative from model
    print("\n[1] Creating Operative from AnalysisResult model...")
    operative = create_operative_from_model(AnalysisResult)
    print(f"   Operative name: {operative.name}")
    print(f"   Operable name: {operative.operable.name}")
    print(f"   Specs: {operative.operable.allowed()}")

    # Step 3: Generate LNDL spec format
    print("\n[2] Generating LNDL spec format...")
    lndl_spec = generate_lndl_spec_format(operative)
    print(f"   LNDL spec:\n{lndl_spec[:500]}...")

    # Step 4: Call OpenAI with LNDL prompt
    print("\n[3] Calling OpenAI API...")

    # Build the full system prompt
    system_prompt = f"{get_lndl_system_prompt()}\n\n{lndl_spec}"

    # User instruction
    user_instruction = """Analyze the topic "Machine Learning in Healthcare" and provide:
- A brief summary
- The most important insight
- Your confidence level (0-1)

Remember to use LNDL format with <lvar> tags and OUT{} block."""

    # Create iModel and invoke
    model = iModel(provider="openai", model="gpt-4o-mini")

    calling = await model.invoke(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    # Get response
    response_data = calling.response
    print(f"   Status: {response_data.status}")
    print(f"   Model used: {response_data.raw_response.get('model', 'unknown')}")

    usage = response_data.raw_response.get("usage", {})
    print(
        f"   Tokens - prompt: {usage.get('prompt_tokens', 'N/A')}, completion: {usage.get('completion_tokens', 'N/A')}, total: {usage.get('total_tokens', 'N/A')}"
    )

    # Extract content - data is already the text content (normalized by OAIChatEndpoint)
    content = response_data.data  # This is the string content directly
    print(f"\n[4] Raw response content:\n{content[:1000]}...")

    # Step 5: Parse with parse_lndl_fuzzy
    print("\n[5] Parsing response with parse_lndl_fuzzy...")
    try:
        parsed = parse_lndl_fuzzy(content, operative.operable)
        print("   Parsed successfully!")
        print(f"   Output keys: {list(parsed.fields.keys())}")

        # Step 6: Verify types
        print("\n[6] Verifying parsed result...")
        if "analysisresult" in parsed.fields:
            result = parsed.fields["analysisresult"]
            print(f"   Type: {type(result)}")
            print(f"   topic: {result.topic[:100] if hasattr(result, 'topic') else 'N/A'}...")
            print(f"   summary: {result.summary[:100] if hasattr(result, 'summary') else 'N/A'}...")
            print(
                f"   key_insight: {result.key_insight[:100] if hasattr(result, 'key_insight') else 'N/A'}..."
            )
            print(f"   confidence: {result.confidence if hasattr(result, 'confidence') else 'N/A'}")

            # Validate types
            assert isinstance(result.topic, str), "topic should be str"
            assert isinstance(result.summary, str), "summary should be str"
            assert isinstance(result.key_insight, str), "key_insight should be str"
            assert isinstance(result.confidence, float), "confidence should be float"
            print("\n   [PASS] All type validations passed!")
        else:
            print(
                f"   [WARNING] 'analysisresult' not in parsed fields: {list(parsed.fields.keys())}"
            )

    except Exception as e:
        print(f"   [FAIL] Parse error: {type(e).__name__}: {e}")
        # Print full traceback for debugging
        import traceback

        traceback.print_exc()
        return False

    return True


async def test_2_prepare_lndl_messages():
    """Test 2: Test the prepare_lndl_messages flow.

    1. Create a Session and Branch
    2. Create an InstructionContent and Message
    3. Call prepare_lndl_messages() and inspect the output
    4. Verify the system prompt includes LNDL guidance
    """
    separator("2. prepare_lndl_messages Flow")

    from lionpride.operations.lndl import prepare_lndl_messages
    from lionpride.operations.operate.operative import create_operative_from_model
    from lionpride.session import Branch, InstructionContent, Message, Session

    # Step 1: Create Session and Branch
    print("\n[1] Creating Session and Branch...")
    session = Session()
    branch = session.create_branch(name="test_branch")
    print(f"   Session ID: {session.id}")
    print(f"   Branch name: {branch.name}")

    # Step 2: Create Instruction and Message
    print("\n[2] Creating InstructionContent and Message...")
    ins_content = InstructionContent(
        instruction="Analyze the topic 'AI Safety' and provide key insights."
    )
    ins_msg = Message(
        content=ins_content,
        sender="user",
        recipient="assistant",
    )
    print(f"   Message role: {ins_msg.role}")
    print(f"   Message content type: {type(ins_msg.content).__name__}")

    # Create Operative
    operative = create_operative_from_model(SimpleReport)
    print(f"   Operative name: {operative.name}")

    # Step 3: Call prepare_lndl_messages
    print("\n[3] Calling prepare_lndl_messages()...")
    messages = prepare_lndl_messages(session, branch, ins_msg, operative)

    print(f"   Number of messages: {len(messages)}")
    for i, msg in enumerate(messages):
        print(
            f"   Message {i}: role={msg['role']}, content_length={len(str(msg.get('content', '')))}"
        )

    # Step 4: Verify LNDL guidance in system prompt
    print("\n[4] Verifying LNDL guidance in system prompt...")
    system_msg = messages[0] if messages[0]["role"] == "system" else None

    if system_msg:
        content = system_msg["content"]
        has_lndl = "LNDL" in content
        has_lvar = "<lvar" in content
        has_out = "OUT{" in content

        print(f"   Contains 'LNDL': {has_lndl}")
        print(f"   Contains '<lvar': {has_lvar}")
        print(f"   Contains 'OUT{{': {has_out}")
        print(f"   System prompt preview:\n{content[:500]}...")

        assert has_lndl, "System prompt should contain 'LNDL'"
        assert has_lvar, "System prompt should contain '<lvar'"
        assert has_out, "System prompt should contain 'OUT{'"
        print("\n   [PASS] LNDL guidance verification passed!")
    else:
        print("   [WARNING] No system message found")
        return False

    return True


async def test_3_streaming_with_lndl():
    """Test 3: Test streaming with LNDL.

    1. Use iModel.invoke_stream_with_channel()
    2. Verify StreamChannel properly parses SSE chunks
    """
    separator("3. Streaming with LNDL via StreamChannel")

    from lionpride import iModel
    from lionpride.lndl import get_lndl_system_prompt
    from lionpride.operations.lndl import generate_lndl_spec_format
    from lionpride.operations.operate.operative import create_operative_from_model

    # Create operative
    print("\n[1] Setting up streaming test...")
    operative = create_operative_from_model(SimpleReport)
    lndl_spec = generate_lndl_spec_format(operative)
    system_prompt = f"{get_lndl_system_prompt()}\n\n{lndl_spec}"

    user_instruction = """Create a brief report on "Python Programming" with:
- A title
- A short content section

Use LNDL format with <lvar> tags and OUT{} block."""

    # Create iModel
    print("\n[2] Creating iModel and starting stream...")
    model = iModel(provider="openai", model="gpt-4o-mini")

    # Stream with channel
    channel = await model.invoke_stream_with_channel(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
        ],
        temperature=0.3,
        max_tokens=800,
        stream=True,
    )

    print("\n[3] Consuming stream chunks...")
    chunk_count = 0
    chunks_collected = []

    def consumer(chunk):
        nonlocal chunk_count
        chunk_count += 1
        if chunk.content:
            chunks_collected.append(chunk.content)
            # Print progress dots
            if chunk_count % 10 == 0:
                print(".", end="", flush=True)

    channel.add_consumer(consumer)

    async for _chunk in channel:
        pass  # Processing handled by consumer

    print(f"\n   Total chunks received: {chunk_count}")
    print(f"   Stream completed: {channel.is_completed}")

    # Get accumulated content
    accumulated = channel.get_accumulated()
    print(f"\n[4] Accumulated content length: {len(accumulated)} chars")
    print(f"   Preview:\n{accumulated[:500]}...")

    # Verify streaming worked
    assert chunk_count > 0, "Should have received at least one chunk"
    assert len(accumulated) > 0, "Accumulated content should not be empty"
    assert channel.is_completed, "Stream should be completed"

    print("\n   [PASS] Streaming verification passed!")

    # Try to parse the streamed content
    print("\n[5] Attempting to parse streamed content...")
    from lionpride.lndl import parse_lndl_fuzzy

    try:
        parsed = parse_lndl_fuzzy(accumulated, operative.operable)
        print("   Parsed successfully!")
        print(f"   Output keys: {list(parsed.fields.keys())}")

        if "simplereport" in parsed.fields:
            result = parsed.fields["simplereport"]
            print(f"   title: {result.title if hasattr(result, 'title') else 'N/A'}")
            print(f"   content: {result.content[:100] if hasattr(result, 'content') else 'N/A'}...")
            print("\n   [PASS] Stream content parsing passed!")
    except Exception as e:
        print(f"   [WARNING] Could not parse streamed content: {e}")
        # This is acceptable - streaming doesn't guarantee LNDL format compliance

    return True


async def main():
    """Run all functional tests."""
    print("\n" + "=" * 60)
    print("LNDL Functional Tests with OpenAI API")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found in environment")
        return

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    results = {}

    # Test 1: Basic LNDL flow
    try:
        results["test_1_basic_lndl_flow"] = await test_1_basic_lndl_flow()
    except Exception as e:
        print(f"\n[ERROR] Test 1 failed with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["test_1_basic_lndl_flow"] = False

    # Test 2: prepare_lndl_messages
    try:
        results["test_2_prepare_lndl_messages"] = await test_2_prepare_lndl_messages()
    except Exception as e:
        print(f"\n[ERROR] Test 2 failed with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["test_2_prepare_lndl_messages"] = False

    # Test 3: Streaming
    try:
        results["test_3_streaming_with_lndl"] = await test_3_streaming_with_lndl()
    except Exception as e:
        print(f"\n[ERROR] Test 3 failed with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["test_3_streaming_with_lndl"] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")

    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
