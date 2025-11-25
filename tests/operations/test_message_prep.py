# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for message_prep module.

Covers lines 17-24 of message_prep.py:
- prepare_tool_schemas with tools=True
- prepare_tool_schemas with tools as list
- prepare_tool_schemas with tools=False
- prepare_tool_schemas with invalid tools type
"""

from unittest.mock import MagicMock

from lionpride.operations.operate.message_prep import prepare_tool_schemas
from lionpride.session import Session


class TestMessagePrepCoverage:
    """Test message_prep.py uncovered lines."""

    def test_prepare_tool_schemas_true(self):
        """Test line 21: tools=True returns all tool schemas."""
        session = Session()

        # Mock get_tool_schemas
        session.services.get_tool_schemas = MagicMock(return_value=[{"name": "tool1"}])

        result = prepare_tool_schemas(session, tools=True)

        assert result == [{"name": "tool1"}]
        session.services.get_tool_schemas.assert_called_once_with()

    def test_prepare_tool_schemas_list(self):
        """Test lines 22-23: tools=list returns specific tool schemas."""
        session = Session()

        # Mock get_tool_schemas
        session.services.get_tool_schemas = MagicMock(return_value=[{"name": "tool1"}])

        result = prepare_tool_schemas(session, tools=["tool1", "tool2"])

        assert result == [{"name": "tool1"}]
        session.services.get_tool_schemas.assert_called_once_with(tool_names=["tool1", "tool2"])

    def test_prepare_tool_schemas_false(self):
        """Test line 17-18: tools=False returns None."""
        session = Session()

        result = prepare_tool_schemas(session, tools=False)

        assert result is None

    def test_prepare_tool_schemas_none_return(self):
        """Test line 24: Invalid tools type returns None."""
        session = Session()

        # Pass something that's not bool or list
        result = prepare_tool_schemas(session, tools="invalid")  # type: ignore

        assert result is None
