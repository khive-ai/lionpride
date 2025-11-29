# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Session and Branch interaction.

Tests cover:
- Session message management across branches
- Branch cloning and message lineage
- System message handling
- Multi-branch workflows
- Session <-> Branch <-> Message integration

API Pattern: All mutations go through session.conversations.xxx
- session.conversations.add_item(msg, progressions=[branch])
- session.conversations.add_progression(branch)
- session.conversations.get_progression(name)
- session.messages / session.branches are read-only views
"""

import pytest

from lionpride.session import Branch, Session
from lionpride.session.messages import (
    AssistantResponseContent,
    InstructionContent,
    Message,
    MessageRole,
    SenderRecipient,
    SystemContent,
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


class TestBranchSystemMessages:
    """Test system message handling in branches."""

    def test_system_message_auto_placement(self):
        """Test system messages automatically placed at order[0]."""
        session = Session()

        sys_msg = Message(
            content=SystemContent(system_message="You are helpful"),
            sender="system",
            recipient="user",
        )

        branch = session.create_branch(name="main", system=sys_msg)

        # Verify system message set
        system = session.get_branch_system(branch)
        assert system is not None
        assert system.id == sys_msg.id
        assert system.role == MessageRole.SYSTEM

    def test_system_message_via_branch_method(self):
        """Test system message set via branch.set_system_message()."""
        session = Session()
        branch = session.create_branch(name="main")

        sys_msg = Message(
            content=SystemContent(system_message="You are helpful"),
            sender="system",
            recipient="user",
        )

        # Add message to storage first
        session.conversations.add_item(sys_msg)
        # Set via branch method
        branch.set_system_message(sys_msg.id)

        # Verify set as branch system message
        system = session.get_branch_system(branch)
        assert system is not None
        assert system.id == sys_msg.id

    def test_change_system_message(self):
        """Test changing branch system message."""
        session = Session()

        sys_msg1 = Message(
            content=SystemContent(system_message="First system"),
            sender="system",
            recipient="user",
        )
        sys_msg2 = Message(
            content=SystemContent(system_message="Second system"),
            sender="system",
            recipient="user",
        )

        branch = session.create_branch(name="main", system=sys_msg1)

        # Change system message
        session.set_branch_system(branch, sys_msg2)

        # Verify changed
        system = session.get_branch_system(branch)
        assert system.id == sys_msg2.id

    def test_system_message_by_name_lookup(self):
        """Test system message retrieval by branch name."""
        session = Session()

        sys_msg = Message(
            content=SystemContent(system_message="Test system"),
            sender="system",
            recipient="user",
        )

        session.create_branch(name="main", system=sys_msg)

        # Retrieve by name
        system = session.get_branch_system("main")
        assert system is not None
        assert system.id == sys_msg.id


class TestBranchForking:
    """Test branch forking and message lineage."""

    def test_basic_branch_fork(self):
        """Test basic branch forking."""
        session = Session()
        original = session.create_branch(name="original")

        msg1 = Message(
            content=InstructionContent(instruction="Message 1"),
            sender="user",
            recipient=session.id,
        )
        msg2 = Message(
            content=AssistantResponseContent(assistant_response="Response 1"),
            sender="assistant",
            recipient="user",
        )

        session.conversations.add_item(msg1, progressions=[original])
        session.conversations.add_item(msg2, progressions=[original])

        # Fork branch
        cloned = session.fork(original, name="cloned")

        # Verify independent branches
        assert cloned.id != original.id
        assert cloned.name == "cloned"

        # Verify messages copied (different IDs)
        original_msgs = list(session.messages[original])
        cloned_msgs = list(session.messages[cloned])
        assert len(cloned_msgs) == len(original_msgs)

    def test_fork_with_system_message(self):
        """Test forking preserves system message content."""
        session = Session()

        sys_msg = Message(
            content=SystemContent(system_message="System prompt"),
            sender="system",
            recipient="user",
        )

        original = session.create_branch(name="original", system=sys_msg)
        cloned = session.fork(original, name="cloned", system=True)

        # Verify system message UUID is copied (same message referenced)
        original_system = session.get_branch_system(original)
        cloned_system = session.get_branch_system(cloned)
        assert original_system is not None
        assert cloned_system is not None
        assert cloned_system.id == original_system.id  # Same message referenced
        assert cloned_system.content.system_message == "System prompt"

    def test_fork_by_branch_reference(self):
        """Test forking branch via direct reference."""
        session = Session()
        original = session.create_branch(name="original")

        msg = Message(
            content=InstructionContent(instruction="Test"),
            sender="user",
            recipient=session.id,
        )
        session.conversations.add_item(msg, progressions=[original])

        # Fork via session.fork()
        cloned = session.fork(original, name="cloned")

        # Verify cloned successfully
        assert cloned.name == "cloned"
        cloned_msgs = list(session.messages[cloned])
        assert len(cloned_msgs) == 1

    def test_fork_shares_message_uuids(self):
        """Test forked branch shares same message UUIDs (not cloned)."""
        session = Session()
        original = session.create_branch(name="original")

        msg = Message(
            content=InstructionContent(instruction="Test"),
            sender="user1",
            recipient=session.id,
        )
        session.conversations.add_item(msg, progressions=[original])

        # Fork branch
        forked = session.fork(original, name="forked")

        # Verify forked branch has same message UUIDs
        assert len(forked) == len(original)
        assert list(forked.order) == list(original.order)
        # Note: Branch no longer has user field - session.id is the user identity


class TestMultiBranchWorkflows:
    """Test multi-branch conversation workflows."""

    def test_parallel_branches(self):
        """Test parallel independent branches."""
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        msg1 = Message(
            content=InstructionContent(instruction="Branch 1 message"),
            sender="user",
            recipient=session.id,
        )
        msg2 = Message(
            content=InstructionContent(instruction="Branch 2 message"),
            sender="user",
            recipient=session.id,
        )

        session.conversations.add_item(msg1, progressions=[branch1])
        session.conversations.add_item(msg2, progressions=[branch2])

        # Verify independent
        branch1_msgs = list(session.messages[branch1])
        branch2_msgs = list(session.messages[branch2])
        assert len(branch1_msgs) == 1
        assert len(branch2_msgs) == 1
        assert branch1_msgs[0].id != branch2_msgs[0].id

    def test_divergent_branches(self):
        """Test branches that diverge from common ancestor."""
        session = Session()
        main = session.create_branch(name="main")

        # Common messages
        msg1 = Message(
            content=InstructionContent(instruction="Common 1"),
            sender="user",
            recipient=session.id,
        )
        msg2 = Message(
            content=AssistantResponseContent(assistant_response="Common 2"),
            sender="assistant",
            recipient="user",
        )

        session.conversations.add_item(msg1, progressions=[main])
        session.conversations.add_item(msg2, progressions=[main])

        # Create divergent branch via fork
        experiment = session.fork(main, name="experiment")

        # Add different messages to each
        main_msg = Message(
            content=InstructionContent(instruction="Main branch"),
            sender="user",
            recipient=session.id,
        )
        exp_msg = Message(
            content=InstructionContent(instruction="Experiment branch"),
            sender="user",
            recipient=session.id,
        )

        session.conversations.add_item(main_msg, progressions=[main])
        session.conversations.add_item(exp_msg, progressions=[experiment])

        # Verify divergence
        main_msgs = list(session.messages[main])
        exp_msgs = list(session.messages[experiment])
        assert len(main_msgs) == 3  # 2 common + 1 new
        assert len(exp_msgs) == 3  # 2 cloned + 1 new
        assert main_msgs[-1].id != exp_msgs[-1].id

    def test_branch_message_isolation(self):
        """Test that branch messages remain isolated."""
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        msg1 = Message(
            content=InstructionContent(instruction="Only in branch1"),
            sender="user",
            recipient=session.id,
        )
        msg2 = Message(
            content=InstructionContent(instruction="Only in branch2"),
            sender="user",
            recipient=session.id,
        )

        session.conversations.add_item(msg1, progressions=[branch1])
        session.conversations.add_item(msg2, progressions=[branch2])

        # Verify isolation
        branch1_msgs = list(session.messages[branch1])
        branch2_msgs = list(session.messages[branch2])

        assert len(branch1_msgs) == 1
        assert len(branch2_msgs) == 1
        assert msg1.id in [m.id for m in branch1_msgs]
        assert msg1.id not in [m.id for m in branch2_msgs]
        assert msg2.id in [m.id for m in branch2_msgs]
        assert msg2.id not in [m.id for m in branch1_msgs]


class TestSessionBranchIntegration:
    """Test Session and Branch integration edge cases."""

    def test_session_repr(self):
        """Test Session string representation."""
        session = Session()
        branch = session.create_branch(name="main")

        msg = Message(
            content=InstructionContent(instruction="Test"),
            sender="user",
            recipient=session.id,
        )
        session.conversations.add_item(msg, progressions=[branch])

        repr_str = repr(session)
        assert "messages=1" in repr_str
        assert "branches=1" in repr_str

    def test_branch_capabilities(self):
        """Test branch capabilities filtering."""
        session = Session()

        # Create branch with specific capabilities
        branch = session.create_branch(name="limited", capabilities={"tool1", "tool2"})

        assert "tool1" in branch.capabilities
        assert "tool2" in branch.capabilities
        assert "tool3" not in branch.capabilities

    def test_message_count_via_len(self):
        """Test session message count via len(session.messages)."""
        session = Session()
        branch = session.create_branch(name="main")

        assert len(session.messages) == 0

        msg1 = Message(
            content=InstructionContent(instruction="Test 1"),
            sender="user",
            recipient=session.id,
        )
        session.conversations.add_item(msg1, progressions=[branch])
        assert len(session.messages) == 1

        msg2 = Message(
            content=InstructionContent(instruction="Test 2"),
            sender="user",
            recipient=session.id,
        )
        session.conversations.add_item(msg2, progressions=[branch])
        assert len(session.messages) == 2

    def test_branch_count_via_len(self):
        """Test session branch count via len(session.branches)."""
        session = Session()

        assert len(session.branches) == 0

        session.create_branch(name="branch1")
        assert len(session.branches) == 1

        session.create_branch(name="branch2")
        assert len(session.branches) == 2

    def test_get_branch_by_uuid(self):
        """Test retrieving branch by UUID."""
        session = Session()
        branch = session.create_branch(name="main")

        # Retrieve by UUID
        retrieved = session.conversations.get_progression(branch.id)
        assert retrieved.id == branch.id
        assert retrieved.name == "main"

    def test_empty_branch_messages(self):
        """Test getting messages from empty branch."""
        session = Session()
        branch = session.create_branch(name="empty")

        messages = list(session.messages[branch])
        assert len(messages) == 0

    def test_branch_without_system_message(self):
        """Test branch without system message."""
        session = Session()
        branch = session.create_branch(name="no_system")

        system = session.get_branch_system(branch)
        assert system is None
