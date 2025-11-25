# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for LNDL v2 cognitive operations using lionpride primitives."""

import pytest

from lionpride.lndl.operations import (
    CognitivePermission,
    transform_observation,
)


class TestCognitivePermission:
    """Tests for CognitivePermission dataclass."""

    def test_default_permissions(self):
        """Default permissions should allow most operations."""
        perm = CognitivePermission()
        assert perm.can_include is True
        assert perm.can_compress is True
        assert perm.can_drop is True
        assert perm.can_notice is True
        assert perm.can_yield is True
        assert perm.max_yields == 10
        assert perm.can_send is False  # Multi-agent disabled by default
        assert perm.allowed_actions == set()  # Empty = all allowed

    def test_restricted_permissions(self):
        """Can create restricted permissions."""
        perm = CognitivePermission(
            can_yield=False,
            can_compress=False,
            max_yields=3,
            allowed_actions={"search", "calculate"},
        )
        assert perm.can_yield is False
        assert perm.can_compress is False
        assert perm.max_yields == 3
        assert perm.allowed_actions == {"search", "calculate"}


class TestTransformObservation:
    """Tests for transform_observation function."""

    def test_none_observation(self):
        """None observation returns None."""
        assert transform_observation(None) is None

    def test_no_transformation(self):
        """No transformation params returns original."""
        data = {"key": "value"}
        assert transform_observation(data) == data

    # keep parameter tests
    def test_keep_top_n_list(self):
        """keep='top_3' on list returns first 3 items."""
        data = [1, 2, 3, 4, 5, 6, 7]
        result = transform_observation(data, keep="top_3")
        assert result == [1, 2, 3]

    def test_keep_top_n_string(self):
        """keep='top_3' on string returns first 3 lines."""
        data = "line1\nline2\nline3\nline4\nline5"
        result = transform_observation(data, keep="top_3")
        assert result == "line1\nline2\nline3"

    def test_keep_summary(self):
        """keep='summary' truncates long strings."""
        data = "x" * 1000
        result = transform_observation(data, keep="summary")
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_keep_first(self):
        """keep='first' returns first item of list."""
        data = ["first", "second", "third"]
        result = transform_observation(data, keep="first")
        assert result == "first"

    def test_keep_first_empty_list(self):
        """keep='first' on empty list returns original."""
        data: list = []
        result = transform_observation(data, keep="first")
        assert result == []

    # drop_full parameter tests
    def test_drop_full_string(self):
        """drop_full truncates long strings."""
        data = "x" * 500
        result = transform_observation(data, drop_full=True)
        assert len(result) == 215  # 200 + "... [truncated]" (15 chars)
        assert "[truncated]" in result

    def test_drop_full_list(self):
        """drop_full converts list to count."""
        data = [1, 2, 3, 4, 5]
        result = transform_observation(data, drop_full=True)
        assert result == "[5 items]"

    def test_drop_full_dict(self):
        """drop_full converts dict to keys."""
        data = {"a": 1, "b": 2, "c": 3}
        result = transform_observation(data, drop_full=True)
        assert "keys:" in result
        assert "a" in result
        assert "b" in result

    # transform parameter tests
    def test_transform_extract_key(self):
        """transform='extract_key' extracts known keys from dict."""
        data = {"result": "the_result", "other": "ignored"}
        result = transform_observation(data, transform="extract_key")
        assert result == "the_result"

    def test_transform_extract_key_data(self):
        """transform='extract_key' tries 'data' key too."""
        data = {"data": "the_data", "other": "ignored"}
        result = transform_observation(data, transform="extract_key")
        assert result == "the_data"

    def test_transform_extract_key_missing(self):
        """transform='extract_key' returns original if no known key."""
        data = {"foo": "bar"}
        result = transform_observation(data, transform="extract_key")
        assert result == {"foo": "bar"}

    # Combined tests
    def test_keep_and_drop_full(self):
        """Combined keep and drop_full."""
        data = [{"x": i} for i in range(10)]
        result = transform_observation(data, keep="top_3", drop_full=True)
        # First keep top_3, then drop_full
        assert result == "[3 items]"


class TestCompressMessages:
    """Tests for compress_messages function - requires Session mock."""

    @pytest.mark.asyncio
    async def test_compress_messages_placeholder(self):
        """Placeholder - full integration test requires Session."""
        # This would require mocking Session, Branch, and iModel
        # For now, we just verify the function signature is correct
        # Verify the function exists and has correct signature
        import inspect

        from lionpride.lndl.operations import compress_messages

        sig = inspect.signature(compress_messages)
        params = list(sig.parameters.keys())
        assert "session" in params
        assert "branch" in params
        assert "start_idx" in params
        assert "end_idx" in params
        assert "imodel" in params
        assert "max_tokens" in params


class TestCognitiveReact:
    """Tests for cognitive_react function - requires Session mock."""

    @pytest.mark.asyncio
    async def test_cognitive_react_placeholder(self):
        """Placeholder - full integration test requires Session."""
        # Verify the function exists and has correct signature
        import inspect

        from lionpride.lndl.operations import cognitive_react

        sig = inspect.signature(cognitive_react)
        params = list(sig.parameters.keys())
        assert "session" in params
        assert "branch" in params
        assert "instruction" in params
        assert "imodel" in params
        assert "action_executor" in params
        assert "permissions" in params
        assert "max_iterations" in params
        assert "verbose" in params
