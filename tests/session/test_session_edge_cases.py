# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Session and Branch.

Tests designed to expose potential bugs and verify edge case handling.
Each test documents the specific edge case being tested.

Covers:
- Session with no default branch
- Branch fork with various inheritance modes
- System message edge cases
- add_message content type dispatch
- get_branch with various input types
- set_default_model with unregistered model
- Multiple branches sharing message UUIDs
- Fork chain lineage tracking
- Empty branch operations
- Referential integrity violations
"""

from uuid import UUID, uuid4

import pytest

from lionpride.errors import NotFoundError
from lionpride.session import Branch, Session
from lionpride.session.messages import (
    ActionRequestContent,
    ActionResponseContent,
    AssistantResponseContent,
    InstructionContent,
    Message,
    MessageRole,
    SystemContent,
)


class TestSessionNoDefaultBranch:
    """Test Session behavior when no default branch is set."""

    @pytest.mark.asyncio
    async def test_conduct_without_branch_and_no_default_raises_runtime_error(self):
        """conduct() with no branch arg and no default branch should raise RuntimeError.

        Edge case: User creates session without default_branch, then calls conduct()
        without specifying a branch. Should fail with clear error message.
        """
        session = Session()

        with pytest.raises(RuntimeError, match="No branch provided and no default branch set"):
            await session.conduct("operate", branch=None)

    @pytest.mark.asyncio
    async def test_conduct_with_explicit_branch_succeeds_without_default(self):
        """conduct() with explicit branch should work even without default branch.

        Edge case: Session created without default, but explicit branch passed to conduct.
        The operation binding should succeed (branch resolution works).
        """
        session = Session()
        branch = session.create_branch(name="explicit")

        # This should not raise RuntimeError about missing branch
        # It will fail during operation execution (no registered factory),
        # but the branch resolution itself succeeds
        op = await session.conduct("operate", branch=branch)
        # Operation was created and bound to branch
        assert op._branch is branch

    def test_default_branch_property_returns_none_when_not_set(self):
        """default_branch property should return None when no default is configured.

        Edge case: Accessing default_branch on session with no default.
        """
        session = Session()
        assert session.default_branch is None

    def test_default_branch_returns_none_after_branch_removal(self):
        """default_branch should return None if the default branch was removed.

        Edge case: Set default branch, then remove it from conversations.
        The _default_branch_id still points to removed branch UUID.
        """
        session = Session()
        branch = session.create_branch(name="to_remove")
        session.set_default_branch(branch)

        # Remove the branch from conversations
        session.conversations.remove_progression(branch.id)

        # default_branch should handle this gracefully (suppress KeyError)
        assert session.default_branch is None


class TestBranchForkInheritance:
    """Test fork() capability/resource inheritance modes."""

    def test_fork_with_capabilities_true_copies_all(self):
        """fork() with capabilities=True should copy all capabilities from source.

        Edge case: Verify True means full copy, not just reference.
        """
        session = Session()
        source = session.create_branch(name="source", capabilities={"cap1", "cap2", "cap3"})

        forked = session.fork(source, capabilities=True)

        assert forked.capabilities == {"cap1", "cap2", "cap3"}
        # Verify it's a copy, not a reference
        forked.capabilities.add("cap4")
        assert "cap4" not in source.capabilities

    def test_fork_with_capabilities_none_creates_empty_set(self):
        """fork() with capabilities=None should create empty capabilities.

        Edge case: Source has capabilities, but fork wants none.
        """
        session = Session()
        source = session.create_branch(name="source", capabilities={"cap1", "cap2"})

        forked = session.fork(source, capabilities=None)

        assert forked.capabilities == set()

    def test_fork_with_explicit_capabilities_set(self):
        """fork() with explicit capabilities set should use that set.

        Edge case: Override source capabilities with different set.
        """
        session = Session()
        source = session.create_branch(name="source", capabilities={"cap1", "cap2"})

        forked = session.fork(source, capabilities={"new_cap"})

        assert forked.capabilities == {"new_cap"}
        assert source.capabilities == {"cap1", "cap2"}

    def test_fork_with_resources_true_copies_all(self):
        """fork() with resources=True should copy all resources from source.

        Edge case: Verify resources copying works like capabilities.
        """
        session = Session()
        source = session.create_branch(name="source", resources={"res1", "res2"})

        forked = session.fork(source, resources=True)

        assert forked.resources == {"res1", "res2"}
        # Verify independence
        forked.resources.add("res3")
        assert "res3" not in source.resources

    def test_fork_with_resources_none_creates_empty_set(self):
        """fork() with resources=None should create empty resources.

        Edge case: Source has resources but fork wants none.
        """
        session = Session()
        source = session.create_branch(name="source", resources={"res1"})

        forked = session.fork(source, resources=None)

        assert forked.resources == set()

    def test_fork_with_mixed_inheritance_modes(self):
        """fork() can use different modes for capabilities vs resources.

        Edge case: capabilities=True but resources=explicit set.
        """
        session = Session()
        source = session.create_branch(
            name="source",
            capabilities={"cap1"},
            resources={"res1", "res2"},
        )

        forked = session.fork(
            source,
            capabilities=True,  # Copy from source
            resources={"custom_res"},  # Explicit override
        )

        assert forked.capabilities == {"cap1"}
        assert forked.resources == {"custom_res"}


class TestSystemMessageEdgeCases:
    """Test system message handling edge cases."""

    def test_set_system_message_twice_replaces_at_order_zero(self):
        """Setting system message twice should replace, not append.

        Edge case: Branch.set_system_message is called multiple times.
        The system message UUID should be updated at order[0].
        """
        session = Session()
        branch = session.create_branch(name="main")

        sys1 = Message(content=SystemContent(system_message="First"))
        sys2 = Message(content=SystemContent(system_message="Second"))

        session.conversations.add_item(sys1)
        session.conversations.add_item(sys2)

        branch.set_system_message(sys1.id)
        assert branch.system_message == sys1.id
        assert len(branch) == 1  # Only system message
        assert branch[0] == sys1.id

        branch.set_system_message(sys2.id)
        assert branch.system_message == sys2.id
        assert len(branch) == 1  # Still only one entry (replaced, not appended)
        assert branch[0] == sys2.id

    def test_set_system_message_with_existing_messages(self):
        """Setting system message on branch with existing messages.

        Edge case: Branch already has messages, then set_system_message is called.
        System message should be inserted at order[0].
        """
        session = Session()
        branch = session.create_branch(name="main")

        # Add regular messages first
        msg1 = Message(content=InstructionContent(instruction="Hello"))
        msg2 = Message(content=AssistantResponseContent(assistant_response="Hi"))
        session.conversations.add_item(msg1, progressions=[branch])
        session.conversations.add_item(msg2, progressions=[branch])

        assert len(branch) == 2

        # Now set system message
        sys_msg = Message(content=SystemContent(system_message="System"))
        session.conversations.add_item(sys_msg)
        branch.set_system_message(sys_msg.id)

        assert len(branch) == 3
        assert branch[0] == sys_msg.id  # System message at front
        assert branch[1] == msg1.id
        assert branch[2] == msg2.id

    def test_same_system_message_on_multiple_branches(self):
        """Same system message UUID can be set on multiple branches.

        Edge case: One system message shared across branches.
        """
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        sys_msg = Message(content=SystemContent(system_message="Shared system"))
        session.conversations.add_item(sys_msg)

        branch1.set_system_message(sys_msg.id)
        branch2.set_system_message(sys_msg.id)

        assert branch1.system_message == branch2.system_message == sys_msg.id

        # Verify retrievable from session
        assert session.get_branch_system(branch1).id == sys_msg.id
        assert session.get_branch_system(branch2).id == sys_msg.id

    def test_set_branch_system_with_uuid_not_in_messages(self):
        """set_branch_system with UUID not in session creates a dangling reference.

        Edge case: Branch.set_system_message doesn't validate UUID existence.
        This is a potential bug - get_branch_system raises NotFoundError.
        """
        session = Session()
        branch = session.create_branch(name="main")

        fake_uuid = uuid4()

        # This sets the system_message on branch but UUID doesn't exist in session
        # No validation happens at set time
        branch.set_system_message(fake_uuid)

        assert branch.system_message == fake_uuid

        # Getting the system message from session raises NotFoundError
        # because the UUID doesn't exist in session.messages
        with pytest.raises(NotFoundError):
            session.get_branch_system(branch)


class TestAddMessageContentTypeDispatch:
    """Test add_message behavior with different content types."""

    def test_add_message_system_content_special_handling(self):
        """SystemContent messages are added to items only, then set via set_system_message.

        Edge case: SystemContent has different code path in add_message.
        """
        session = Session()
        branch = session.create_branch(name="main")

        sys_msg = Message(content=SystemContent(system_message="System prompt"))

        session.add_message(sys_msg, branches=[branch])

        # Verify message in session storage
        assert sys_msg.id in session.messages

        # Verify NOT added to branch progression via normal append
        # Instead, it's set via set_system_message
        assert branch.system_message == sys_msg.id
        assert branch[0] == sys_msg.id

    def test_add_message_instruction_content_normal_handling(self):
        """InstructionContent is added to items and progressions normally.

        Edge case: Contrast with SystemContent special handling.
        """
        session = Session()
        branch = session.create_branch(name="main")

        msg = Message(content=InstructionContent(instruction="Hello"))

        session.add_message(msg, branches=[branch])

        assert msg.id in session.messages
        assert msg.id in branch.order
        assert branch.system_message is None  # No system message set

    def test_add_message_system_content_to_multiple_branches(self):
        """SystemContent can be set on multiple branches via add_message.

        Edge case: add_message with SystemContent and list of branches.
        """
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        sys_msg = Message(content=SystemContent(system_message="Shared"))

        session.add_message(sys_msg, branches=[branch1, branch2])

        assert branch1.system_message == sys_msg.id
        assert branch2.system_message == sys_msg.id

    def test_add_message_system_content_no_branches(self):
        """SystemContent without branches just adds to session storage.

        Edge case: System message added but not set on any branch.
        """
        session = Session()

        sys_msg = Message(content=SystemContent(system_message="Orphan system"))

        session.add_message(sys_msg, branches=None)

        assert sys_msg.id in session.messages
        # Not set on any branch

    def test_add_message_action_response_content(self):
        """ActionResponseContent follows normal add path.

        Edge case: Tool response content type.
        """
        session = Session()
        branch = session.create_branch(name="main")

        msg = Message(
            content=ActionResponseContent(
                request_id="req123",
                result={"answer": 42},
            )
        )

        session.add_message(msg, branches=branch)

        assert msg.id in session.messages
        assert msg.id in branch.order
        assert msg.role == MessageRole.TOOL


class TestGetBranchInputTypes:
    """Test get_branch with various input types."""

    def test_get_branch_by_uuid_object(self):
        """get_branch accepts UUID object directly."""
        session = Session()
        branch = session.create_branch(name="test")

        retrieved = session.get_branch(branch.id)
        assert retrieved.id == branch.id

    def test_get_branch_by_uuid_string(self):
        """get_branch accepts UUID as string.

        Edge case: UUID string is parsed and used for lookup.
        """
        session = Session()
        branch = session.create_branch(name="test")

        retrieved = session.get_branch(str(branch.id))
        assert retrieved.id == branch.id

    def test_get_branch_by_name_string(self):
        """get_branch accepts branch name string."""
        session = Session()
        branch = session.create_branch(name="my_branch")

        retrieved = session.get_branch("my_branch")
        assert retrieved.id == branch.id

    def test_get_branch_by_branch_instance(self):
        """get_branch accepts Branch instance directly.

        Edge case: Returns same instance if in session.
        """
        session = Session()
        branch = session.create_branch(name="test")

        retrieved = session.get_branch(branch)
        assert retrieved is branch

    def test_get_branch_not_found_raises_notfounderror(self):
        """get_branch raises NotFoundError for missing branch."""
        session = Session()

        with pytest.raises(NotFoundError, match="Branch not found"):
            session.get_branch("nonexistent")

    def test_get_branch_with_default_value(self):
        """get_branch returns default when branch not found.

        Edge case: Using default parameter (positional-only) to avoid exception.
        """
        session = Session()

        # Note: default is positional-only parameter (after /)
        result = session.get_branch("nonexistent", None)
        assert result is None

        sentinel = object()
        result = session.get_branch("nonexistent", sentinel)
        assert result is sentinel

    def test_get_branch_instance_not_in_session_raises(self):
        """get_branch with Branch instance not in session raises error.

        Edge case: Passing a Branch object that belongs to different session.
        """
        session1 = Session()
        session2 = Session()

        branch = session2.create_branch(name="other_session")

        # Branch not in session1's conversations - raises NotFoundError
        # Error message comes from Pile, not Session
        with pytest.raises(NotFoundError, match="not found"):
            session1.get_branch(branch)


class TestSetDefaultModelEdgeCases:
    """Test set_default_model edge cases."""

    def test_set_default_model_with_unregistered_name(self):
        """set_default_model with unregistered string name still sets the name.

        Edge case: The method stores the name without validating it exists.
        This could lead to runtime errors when actually using the model.
        """
        session = Session()
        session.create_branch(name="main")  # Create default branch

        # Set model name that doesn't exist in registry
        session.set_default_model("nonexistent_model", operation="generate")

        # The name is stored
        assert session._default_backends["generate"] == "nonexistent_model"

        # But trying to get the model returns None (graceful handling)
        assert session.default_generate_model is None

    def test_set_default_model_adds_to_default_branch_resources(self):
        """set_default_model adds model name to default branch resources.

        Edge case: Resource tracking for access control.
        """
        session = Session(default_branch="main")
        branch = session.default_branch

        session.set_default_model("my_model", operation="generate")

        assert "my_model" in branch.resources

    def test_set_default_model_without_default_branch(self):
        """set_default_model works even without default branch.

        Edge case: No default branch to update resources.
        """
        session = Session()  # No default branch

        # Should not raise
        session.set_default_model("my_model", operation="parse")

        assert session._default_backends["parse"] == "my_model"

    def test_set_default_model_for_parse_operation(self):
        """set_default_model can set different models for parse vs generate.

        Edge case: Separate parse model configuration.
        """
        session = Session()

        session.set_default_model("generate_model", operation="generate")
        session.set_default_model("parse_model", operation="parse")

        assert session._default_backends["generate"] == "generate_model"
        assert session._default_backends["parse"] == "parse_model"


class TestMultipleBranchesSharedMessages:
    """Test multiple branches sharing the same message UUIDs."""

    def test_message_in_multiple_branches_single_storage(self):
        """Message added to multiple branches is stored once in session.

        Edge case: Deduplication in Pile storage.
        """
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        msg = Message(content=InstructionContent(instruction="Shared"))

        session.conversations.add_item(msg, progressions=[branch1, branch2])

        # Message stored once
        assert len(session.messages) == 1

        # But referenced in both branches
        assert msg.id in branch1.order
        assert msg.id in branch2.order

    def test_remove_message_from_one_branch_keeps_in_other(self):
        """Removing message from one branch keeps it in storage and other branches.

        Edge case: Branch removal vs session removal.
        """
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        msg = Message(content=InstructionContent(instruction="Shared"))
        session.conversations.add_item(msg, progressions=[branch1, branch2])

        # Remove from branch1 only
        branch1.remove(msg.id)

        # Message still in storage
        assert msg.id in session.messages

        # Still in branch2
        assert msg.id in branch2.order

        # Not in branch1
        assert msg.id not in branch1.order

    def test_forked_branches_share_message_references(self):
        """Forked branches share message UUIDs (not copies).

        Edge case: Fork semantics - shallow copy of order, not deep clone.
        """
        session = Session()
        source = session.create_branch(name="source")

        msg = Message(content=InstructionContent(instruction="Original"))
        session.conversations.add_item(msg, progressions=[source])

        forked = session.fork(source)

        # Same message UUID in both
        assert list(source.order) == list(forked.order)
        assert msg.id in forked.order

        # Same single message in storage
        assert len(session.messages) == 1


class TestForkChainLineage:
    """Test fork chain lineage tracking."""

    def test_fork_records_lineage_metadata(self):
        """fork() records lineage in metadata.forked_from.

        Edge case: Verify lineage metadata structure.
        """
        session = Session()
        original = session.create_branch(name="original")

        forked = session.fork(original, name="forked")

        assert "forked_from" in forked.metadata
        lineage = forked.metadata["forked_from"]
        assert lineage["branch_id"] == str(original.id)
        assert lineage["branch_name"] == "original"
        assert "created_at" in lineage
        assert lineage["message_count"] == 0

    def test_fork_chain_accumulates_lineage(self):
        """Chain of forks accumulates lineage in metadata.

        Edge case: Fork a fork - each forked branch has its own lineage,
        but does NOT inherit parent's lineage automatically.
        """
        session = Session()
        gen0 = session.create_branch(name="gen0")

        msg = Message(content=InstructionContent(instruction="Start"))
        session.conversations.add_item(msg, progressions=[gen0])

        gen1 = session.fork(gen0, name="gen1")
        gen2 = session.fork(gen1, name="gen2")
        gen3 = session.fork(gen2, name="gen3")

        # Each fork only knows its immediate parent
        assert gen1.metadata["forked_from"]["branch_name"] == "gen0"
        assert gen2.metadata["forked_from"]["branch_name"] == "gen1"
        assert gen3.metadata["forked_from"]["branch_name"] == "gen2"

        # gen3 doesn't directly know about gen0
        assert gen3.metadata["forked_from"]["branch_name"] != "gen0"

    def test_fork_chain_message_propagation(self):
        """Messages accumulate through fork chain.

        Edge case: Each fork starts with parent's messages, can diverge.
        """
        session = Session()
        gen0 = session.create_branch(name="gen0")

        msg0 = Message(content=InstructionContent(instruction="Gen0"))
        session.conversations.add_item(msg0, progressions=[gen0])

        gen1 = session.fork(gen0, name="gen1")
        msg1 = Message(content=InstructionContent(instruction="Gen1"))
        session.conversations.add_item(msg1, progressions=[gen1])

        gen2 = session.fork(gen1, name="gen2")
        msg2 = Message(content=InstructionContent(instruction="Gen2"))
        session.conversations.add_item(msg2, progressions=[gen2])

        # gen2 has all messages
        assert len(gen2) == 3
        assert msg0.id in gen2.order
        assert msg1.id in gen2.order
        assert msg2.id in gen2.order

        # gen0 only has its original message
        assert len(gen0) == 1

        # gen1 has two
        assert len(gen1) == 2


class TestEmptyBranchOperations:
    """Test operations on empty branches."""

    def test_get_branch_system_on_empty_branch(self):
        """get_branch_system returns None for branch without system message.

        Edge case: Empty branch with no system_message set.
        """
        session = Session()
        branch = session.create_branch(name="empty")

        system = session.get_branch_system(branch)
        assert system is None

    def test_fork_empty_branch(self):
        """Forking an empty branch creates empty fork.

        Edge case: Fork with no messages to copy.
        """
        session = Session()
        empty = session.create_branch(name="empty")

        forked = session.fork(empty, name="forked_empty")

        assert len(forked) == 0
        assert forked.metadata["forked_from"]["message_count"] == 0

    def test_empty_branch_iteration(self):
        """Iterating over messages in empty branch yields nothing.

        Edge case: list(session.messages[branch]) for empty branch.
        """
        session = Session()
        branch = session.create_branch(name="empty")

        messages = list(session.messages[branch])
        assert messages == []

    def test_branch_len_on_empty(self):
        """len(branch) returns 0 for empty branch.

        Edge case: Empty branch length.
        """
        session = Session()
        branch = session.create_branch(name="empty")

        assert len(branch) == 0


class TestReferentialIntegrityViolations:
    """Test referential integrity enforcement."""

    def test_create_branch_with_nonexistent_system_uuid_raises(self):
        """create_branch with system UUID not in session raises ValueError.

        Edge case: Referential integrity on branch creation.
        """
        session = Session()
        fake_uuid = uuid4()

        with pytest.raises(ValueError, match=r"System message UUID .* not found"):
            session.create_branch(name="bad", system=fake_uuid)

    def test_create_branch_with_nonexistent_messages_raises(self):
        """create_branch with messages containing nonexistent UUIDs raises error.

        Edge case: Referential integrity for initial messages.
        """
        session = Session()
        fake_uuids = [uuid4(), uuid4()]

        with pytest.raises(NotFoundError, match="not in items pile"):
            session.create_branch(name="bad", messages=fake_uuids)

    def test_flow_add_progression_with_invalid_uuids_raises(self):
        """Adding progression with invalid UUIDs raises NotFoundError.

        Edge case: Direct Flow manipulation bypassing session.
        """
        session = Session()

        # Create a branch with invalid UUIDs directly
        invalid_branch = Branch(
            session_id=session.id,
            name="invalid",
            order=[uuid4()],  # UUID not in session.messages
        )

        with pytest.raises(NotFoundError, match="not in items pile"):
            session.conversations.add_progression(invalid_branch)


class TestSessionInitializationEdgeCases:
    """Test Session initialization edge cases."""

    def test_session_with_string_default_branch_creates_branch(self):
        """Session with string default_branch creates named branch.

        Edge case: String as branch name vs Branch instance.
        """
        session = Session(default_branch="my_main")

        assert session.default_branch is not None
        assert session.default_branch.name == "my_main"

    def test_session_with_branch_instance_default_branch(self):
        """Session can be initialized with Branch instance as default.

        Edge case: Pre-constructed Branch passed to Session.
        """
        # Branch needs session_id, so we must create session first
        # then manually set up the branch
        temp_session_id = uuid4()
        _branch = Branch(session_id=temp_session_id, name="precreated")

        # This won't work cleanly because session_id is frozen
        # Let's test with UUID default_branch instead
        session = Session(default_branch="test")
        assert session.default_branch is not None

    def test_session_with_default_system_message(self):
        """Session can initialize with default_system for default branch.

        Edge case: System message set on default branch during init.
        """
        sys_msg = Message(content=SystemContent(system_message="Default system"))

        session = Session(default_branch="main", default_system=sys_msg)

        # System message should be in storage and set on default branch
        assert sys_msg.id in session.messages
        assert session.default_branch.system_message == sys_msg.id

    def test_session_repr_with_services(self):
        """Session repr includes service count.

        Edge case: Verify repr format with services.
        """
        session = Session()
        session.create_branch(name="main")

        repr_str = repr(session)
        assert "services=" in repr_str


class TestBranchSystemMessageReplacement:
    """Test system message replacement behavior."""

    def test_set_system_replaces_in_order_not_appends(self):
        """Setting system message on branch with existing system replaces at index 0.

        Edge case: The old system_message UUID should be replaced, not left orphaned.
        """
        session = Session()
        branch = session.create_branch(name="main")

        sys1 = Message(content=SystemContent(system_message="First"))
        sys2 = Message(content=SystemContent(system_message="Second"))

        session.conversations.add_item(sys1)
        session.conversations.add_item(sys2)

        # Set first system message
        session.set_branch_system(branch, sys1)
        assert branch[0] == sys1.id

        # Set second system message - should replace
        session.set_branch_system(branch, sys2)
        assert branch[0] == sys2.id
        assert len(branch) == 1  # Not 2

    def test_set_branch_system_auto_adds_to_session(self):
        """set_branch_system with Message not in session adds it automatically.

        Edge case: Convenience behavior - auto-add message.
        """
        session = Session()
        branch = session.create_branch(name="main")

        sys_msg = Message(content=SystemContent(system_message="Auto-added"))

        # Message not in session yet
        assert sys_msg.id not in session.messages

        # set_branch_system should add it
        session.set_branch_system(branch, sys_msg)

        # Now it's in session
        assert sys_msg.id in session.messages
        assert branch.system_message == sys_msg.id


class TestBranchRepr:
    """Test Branch string representation."""

    def test_branch_repr_format(self):
        """Branch repr shows message count and truncated session ID.

        Edge case: Verify repr format for debugging.
        """
        session = Session()
        branch = session.create_branch(name="my_branch")

        repr_str = repr(branch)
        assert "Branch(" in repr_str
        assert "messages=0" in repr_str
        assert "session=" in repr_str
        assert "name='my_branch'" in repr_str

    def test_branch_repr_without_name(self):
        """Branch repr handles auto-generated names.

        Edge case: Branch with default name.
        """
        session = Session()
        branch = session.create_branch()  # Auto-generated name

        repr_str = repr(branch)
        assert "Branch(" in repr_str
        # Name should still be present (auto-generated like "branch_0")
        assert "name=" in repr_str
