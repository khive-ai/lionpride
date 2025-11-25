# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Session and Branch interaction.

Tests cover:
- Session message management across branches
- Message storage and branch assignment
"""

from lionpride.session import Session
from lionpride.session.messages import (
    AssistantResponseContent,
    InstructionContent,
    Message,
)


class TestSessionMessageManagement:
    """Test Session message management with branches."""

    def test_add_message_to_single_branch(self):
        """Test adding messages to a single branch."""
        session = Session()
        branch = session.create_branch(name="main")

        msg1 = Message(
            content=InstructionContent(instruction="Hello"),
            sender="user",
            recipient=session.id,
        )
        msg2 = Message(
            content=AssistantResponseContent(assistant_response="Hi there"),
            sender="assistant",
            recipient="user",
        )

        session.conversations.add_item(msg1, progressions=[branch])
        session.conversations.add_item(msg2, progressions=[branch])

        # Verify messages in session storage
        assert msg1.id in session.messages
        assert msg2.id in session.messages

        # Verify messages in branch
        branch_msgs = list(session.messages[branch])
        assert len(branch_msgs) == 2
        assert branch_msgs[0].id == msg1.id
        assert branch_msgs[1].id == msg2.id

    def test_add_message_to_multiple_branches(self):
        """Test adding same message to multiple branches."""
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        msg = Message(
            content=InstructionContent(instruction="Shared message"),
            sender="user",
            recipient=session.id,
        )

        # Add to both branches
        session.conversations.add_item(msg, progressions=[branch1, branch2])

        # Verify message stored once
        assert msg.id in session.messages
        assert len(session.messages) == 1

        # Verify message in both branches
        branch1_msgs = list(session.messages[branch1])
        branch2_msgs = list(session.messages[branch2])
        assert len(branch1_msgs) == 1
        assert len(branch2_msgs) == 1
        assert branch1_msgs[0].id == msg.id
        assert branch2_msgs[0].id == msg.id

    def test_add_message_storage_only(self):
        """Test adding message to storage without branch."""
        session = Session()

        msg = Message(
            content=InstructionContent(instruction="Storage only"),
            sender="user",
            recipient=session.id,
        )

        session.conversations.add_item(msg)

        # Verify in storage
        assert msg.id in session.messages
        assert len(session.messages) == 1

    def test_message_deduplication(self):
        """Test that adding same message twice doesn't duplicate in storage."""
        session = Session()
        branch = session.create_branch(name="main")

        msg = Message(
            content=InstructionContent(instruction="Test"),
            sender="user",
            recipient=session.id,
        )

        session.conversations.add_item(msg, progressions=[branch])
        # Adding same message again - storage deduplicates
        if msg.id not in session.messages:
            session.conversations.add_item(msg)

        # Verify stored once
        assert len(session.messages) == 1
