# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Test Claude Code integration via iModel using IPU pattern.

Run manually with: uv run pytest scripts/functional_test/test_claude_code.py -v -s
"""

import shutil

import pytest

from lionpride.ipu import IPU
from lionpride.operations import GenerateParam
from lionpride.services import iModel
from lionpride.session import Session

pytestmark = pytest.mark.skipif(
    not shutil.which("claude"),
    reason="Claude CLI not installed (npm i -g @anthropic-ai/claude-code)",
)


@pytest.mark.asyncio
async def test_claude_code_basic_invoke():
    """Test basic Claude Code invocation via iModel (direct, not via IPU)."""
    model = iModel(
        provider="claude_code",
        endpoint="query_cli",
        model="sonnet",
        ws=".claude_test",  # Relative to cwd
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    assert model.name == "claude_code_cli"

    # Invoke directly (not via IPU - just testing iModel works)
    calling = await model.invoke(
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    )

    assert calling.execution.status.value == "completed"
    assert calling.execution.response is not None
    assert calling.execution.response.status == "success"

    # Check response data
    data = calling.execution.response.data
    assert data is not None
    assert "4" in data

    print(f"\nResponse: {data}")
    print(f"Model: {calling.execution.response.metadata.get('model')}")
    print(f"Cost: ${calling.execution.response.metadata.get('total_cost_usd', 0):.4f}")


@pytest.mark.asyncio
async def test_claude_code_with_ipu_pattern():
    """Test Claude Code with IPU + Session pattern."""
    model = iModel(
        provider="claude_code",
        endpoint="query_cli",
        model="sonnet",
        ws=".claude_test",
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    session.services.register(model)
    ipu.register_session(session)

    branch = session.create_branch(name="test")

    # Use generate via session.conduct() + IPU
    params = GenerateParam(
        imodel=model,
        messages=[{"role": "user", "content": "What is 3*3? Reply with just the number."}],
        return_as="text",
    )

    op = await session.conduct(branch, "generate", ipu, params=params)
    print(f"\nOperation status: {op.status}")
    print("PASS: Generate operation executed via IPU pattern")


@pytest.mark.asyncio
async def test_claude_code_session_continuation():
    """Test Claude Code session continuation via provider_metadata."""
    model = iModel(
        provider="claude_code",
        endpoint="query_cli",
        model="sonnet",
        ws=".claude_test",
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    # First call - no session_id
    assert model.provider_metadata.get("session_id") is None

    await model.invoke(
        messages=[{"role": "user", "content": "Remember the number 42. Just say OK."}],
    )

    # After first call - session_id should be stored
    session_id = model.provider_metadata.get("session_id")
    print(f"\nFirst call session_id: {session_id}")
    assert session_id is not None, "session_id should be stored after first call"

    # Second call - should auto-resume the session
    calling2 = await model.invoke(
        messages=[{"role": "user", "content": "What number did I tell you to remember?"}],
    )

    # Verify session continuity
    print(f"Second call response: {calling2.execution.response.data}")
    assert "42" in calling2.execution.response.data, "Session should have context from first call"

    # Session ID should remain the same
    assert model.provider_metadata.get("session_id") == session_id


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_claude_code_basic_invoke())
