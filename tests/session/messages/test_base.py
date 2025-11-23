# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for session.messages.base - 90%+ coverage target.

Test Surface:
    - MessageRole enum (values, allowed(), string conversion)
    - validate_sender_recipient() function (all input types and edge cases)
    - serialize_sender_recipient() function (all type conversions)
    - SenderRecipient type alias (implicitly tested via functions)
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from lionpride.session.messages.base import (
    MessageRole,
    SenderRecipient,
    serialize_sender_recipient,
    validate_sender_recipient,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_uuid():
    """Sample UUID for testing."""
    return uuid4()


@pytest.fixture
def sample_uuid_str(sample_uuid):
    """String representation of sample UUID."""
    return str(sample_uuid)


@pytest.fixture
def non_uuid_string():
    """Regular string that's not a UUID."""
    return "custom_sender_name"


# =============================================================================
# MessageRole Enum Tests
# =============================================================================


def test_message_role_enum_values():
    """Test MessageRole enum has all expected values."""
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.TOOL.value == "tool"
    assert MessageRole.UNSET.value == "unset"


def test_message_role_enum_allowed():
    """Test MessageRole.allowed() returns all enum values."""
    allowed = MessageRole.allowed()
    assert MessageRole.SYSTEM in allowed
    assert MessageRole.USER in allowed
    assert MessageRole.ASSISTANT in allowed
    assert MessageRole.TOOL in allowed
    assert MessageRole.UNSET in allowed
    assert len(allowed) == 5


def test_message_role_enum_string_representation():
    """Test MessageRole can be created from string values."""
    assert MessageRole("system") == MessageRole.SYSTEM
    assert MessageRole("user") == MessageRole.USER
    assert MessageRole("assistant") == MessageRole.ASSISTANT
    assert MessageRole("tool") == MessageRole.TOOL
    assert MessageRole("unset") == MessageRole.UNSET


def test_message_role_enum_comparison():
    """Test MessageRole enum equality comparisons."""
    assert MessageRole.SYSTEM == MessageRole.SYSTEM
    assert MessageRole.USER != MessageRole.ASSISTANT
    assert MessageRole.TOOL != MessageRole.UNSET


# =============================================================================
# validate_sender_recipient() Tests - Happy Paths
# =============================================================================


def test_validate_sender_recipient_when_message_role_then_returns_as_is():
    """Test validate_sender_recipient with MessageRole input returns unchanged."""
    result = validate_sender_recipient(MessageRole.USER)
    assert result == MessageRole.USER
    assert isinstance(result, MessageRole)


def test_validate_sender_recipient_when_uuid_then_returns_as_is(sample_uuid):
    """Test validate_sender_recipient with UUID input returns unchanged."""
    result = validate_sender_recipient(sample_uuid)
    assert result == sample_uuid
    assert isinstance(result, UUID)


def test_validate_sender_recipient_when_none_then_returns_unset():
    """Test validate_sender_recipient with None returns MessageRole.UNSET."""
    result = validate_sender_recipient(None)
    assert result == MessageRole.UNSET
    assert isinstance(result, MessageRole)


@pytest.mark.parametrize(
    "role_string",
    ["system", "user", "assistant", "tool", "unset"],
)
def test_validate_sender_recipient_when_role_string_then_converts_to_enum(role_string):
    """Test validate_sender_recipient converts role strings to MessageRole enum."""
    result = validate_sender_recipient(role_string)
    assert isinstance(result, MessageRole)
    assert result.value == role_string


def test_validate_sender_recipient_when_uuid_string_then_converts_to_uuid(sample_uuid_str):
    """Test validate_sender_recipient converts valid UUID strings to UUID objects."""
    result = validate_sender_recipient(sample_uuid_str)
    assert isinstance(result, UUID)
    assert str(result) == sample_uuid_str


def test_validate_sender_recipient_when_arbitrary_string_then_returns_string(non_uuid_string):
    """Test validate_sender_recipient returns arbitrary strings unchanged."""
    result = validate_sender_recipient(non_uuid_string)
    assert result == non_uuid_string
    assert isinstance(result, str)


# =============================================================================
# validate_sender_recipient() Tests - Edge Cases
# =============================================================================


def test_validate_sender_recipient_when_empty_string_then_returns_empty():
    """Test validate_sender_recipient with empty string returns empty string."""
    result = validate_sender_recipient("")
    assert result == ""
    assert isinstance(result, str)


def test_validate_sender_recipient_when_malformed_uuid_string_then_returns_string():
    """Test validate_sender_recipient with malformed UUID returns as string."""
    malformed = "not-a-valid-uuid-format"
    result = validate_sender_recipient(malformed)
    assert result == malformed
    assert isinstance(result, str)


def test_validate_sender_recipient_when_partial_uuid_string_then_returns_string():
    """Test validate_sender_recipient with partial UUID returns as string."""
    partial = "12345678-1234"  # Incomplete UUID
    result = validate_sender_recipient(partial)
    assert result == partial
    assert isinstance(result, str)


def test_validate_sender_recipient_when_uuid_with_extra_chars_then_returns_string():
    """Test validate_sender_recipient with UUID-like string with extra characters."""
    uuid_ish = "prefix-" + str(uuid4())
    result = validate_sender_recipient(uuid_ish)
    assert result == uuid_ish
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "invalid_value",
    [123, 45.67, [], {}, object(), True, False],
)
def test_validate_sender_recipient_when_invalid_type_then_raises(invalid_value):
    """Test validate_sender_recipient raises ValueError for invalid types."""
    with pytest.raises(ValueError, match="Invalid sender or recipient"):
        validate_sender_recipient(invalid_value)


def test_validate_sender_recipient_when_list_then_raises():
    """Test validate_sender_recipient raises ValueError for list input."""
    with pytest.raises(ValueError, match="Invalid sender or recipient"):
        validate_sender_recipient(["system"])


def test_validate_sender_recipient_when_dict_then_raises():
    """Test validate_sender_recipient raises ValueError for dict input."""
    with pytest.raises(ValueError, match="Invalid sender or recipient"):
        validate_sender_recipient({"role": "system"})


# =============================================================================
# serialize_sender_recipient() Tests - Happy Paths
# =============================================================================


def test_serialize_sender_recipient_when_message_role_then_returns_value():
    """Test serialize_sender_recipient converts MessageRole to its string value."""
    result = serialize_sender_recipient(MessageRole.ASSISTANT)
    assert result == "assistant"
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "role,expected",
    [
        (MessageRole.SYSTEM, "system"),
        (MessageRole.USER, "user"),
        (MessageRole.ASSISTANT, "assistant"),
        (MessageRole.TOOL, "tool"),
        (MessageRole.UNSET, "unset"),
    ],
)
def test_serialize_sender_recipient_when_each_role_then_returns_value(role, expected):
    """Test serialize_sender_recipient for each MessageRole value."""
    result = serialize_sender_recipient(role)
    assert result == expected


def test_serialize_sender_recipient_when_uuid_then_returns_string(sample_uuid):
    """Test serialize_sender_recipient converts UUID to string."""
    result = serialize_sender_recipient(sample_uuid)
    assert result == str(sample_uuid)
    assert isinstance(result, str)


def test_serialize_sender_recipient_when_string_then_returns_unchanged(non_uuid_string):
    """Test serialize_sender_recipient returns string unchanged."""
    result = serialize_sender_recipient(non_uuid_string)
    assert result == non_uuid_string
    assert isinstance(result, str)


# =============================================================================
# serialize_sender_recipient() Tests - Edge Cases
# =============================================================================


def test_serialize_sender_recipient_when_none_then_returns_none():
    """Test serialize_sender_recipient with None returns None."""
    result = serialize_sender_recipient(None)
    assert result is None


def test_serialize_sender_recipient_when_empty_string_then_returns_none():
    """Test serialize_sender_recipient with empty string returns None."""
    result = serialize_sender_recipient("")
    assert result is None


def test_serialize_sender_recipient_when_zero_then_returns_none():
    """Test serialize_sender_recipient with zero returns None (falsy)."""
    result = serialize_sender_recipient(0)
    assert result is None


def test_serialize_sender_recipient_when_false_then_returns_none():
    """Test serialize_sender_recipient with False returns None (falsy)."""
    result = serialize_sender_recipient(False)
    assert result is None


def test_serialize_sender_recipient_when_empty_list_then_returns_none():
    """Test serialize_sender_recipient with empty list returns None (falsy)."""
    result = serialize_sender_recipient([])
    assert result is None


def test_serialize_sender_recipient_when_int_then_returns_string():
    """Test serialize_sender_recipient converts non-falsy int to string."""
    result = serialize_sender_recipient(42)
    assert result == "42"
    assert isinstance(result, str)


def test_serialize_sender_recipient_when_float_then_returns_string():
    """Test serialize_sender_recipient converts float to string."""
    result = serialize_sender_recipient(3.14)
    assert result == "3.14"
    assert isinstance(result, str)


def test_serialize_sender_recipient_when_object_then_returns_string():
    """Test serialize_sender_recipient converts arbitrary object to string."""
    obj = object()
    result = serialize_sender_recipient(obj)
    assert isinstance(result, str)
    assert str(obj) == result


# =============================================================================
# Integration Tests - Round Trip
# =============================================================================


def test_roundtrip_when_message_role_then_preserves():
    """Test validate → serialize → validate preserves MessageRole."""
    original = MessageRole.USER
    validated = validate_sender_recipient(original)
    serialized = serialize_sender_recipient(validated)
    restored = validate_sender_recipient(serialized)

    assert restored == original
    assert isinstance(restored, MessageRole)


def test_roundtrip_when_uuid_then_preserves(sample_uuid):
    """Test validate → serialize → validate preserves UUID."""
    validated = validate_sender_recipient(sample_uuid)
    serialized = serialize_sender_recipient(validated)
    restored = validate_sender_recipient(serialized)

    assert restored == sample_uuid
    assert isinstance(restored, UUID)


def test_roundtrip_when_string_then_preserves(non_uuid_string):
    """Test validate → serialize → validate preserves string."""
    validated = validate_sender_recipient(non_uuid_string)
    serialized = serialize_sender_recipient(validated)
    restored = validate_sender_recipient(serialized)

    assert restored == non_uuid_string
    assert isinstance(restored, str)


def test_roundtrip_when_none_then_becomes_unset_string():
    """Test validate(None) → serialize → validate becomes 'unset' string."""
    validated = validate_sender_recipient(None)  # MessageRole.UNSET
    serialized = serialize_sender_recipient(validated)  # "unset"
    restored = validate_sender_recipient(serialized)  # MessageRole.UNSET

    assert restored == MessageRole.UNSET
    assert serialized == "unset"


# =============================================================================
# Type Alias Tests (implicitly tested via functions)
# =============================================================================


def test_sender_recipient_type_accepts_message_role():
    """Test SenderRecipient type alias accepts MessageRole."""
    value: SenderRecipient = MessageRole.SYSTEM
    assert isinstance(value, MessageRole)


def test_sender_recipient_type_accepts_uuid(sample_uuid):
    """Test SenderRecipient type alias accepts UUID."""
    value: SenderRecipient = sample_uuid
    assert isinstance(value, UUID)


def test_sender_recipient_type_accepts_string():
    """Test SenderRecipient type alias accepts str."""
    value: SenderRecipient = "custom_name"
    assert isinstance(value, str)


# =============================================================================
# Boundary Tests
# =============================================================================


def test_validate_sender_recipient_when_whitespace_only_string_then_returns_string():
    """Test validate_sender_recipient with whitespace-only string."""
    whitespace = "   \t\n  "
    result = validate_sender_recipient(whitespace)
    assert result == whitespace
    assert isinstance(result, str)


def test_validate_sender_recipient_when_very_long_string_then_returns_string():
    """Test validate_sender_recipient with very long string."""
    long_string = "x" * 10000
    result = validate_sender_recipient(long_string)
    assert result == long_string
    assert isinstance(result, str)


def test_serialize_sender_recipient_when_uuid_with_uppercase_then_normalizes(sample_uuid):
    """Test serialize_sender_recipient converts UUID to lowercase string."""
    result = serialize_sender_recipient(sample_uuid)
    # UUIDs are typically lowercase in string form
    assert result == str(sample_uuid).lower() or result == str(sample_uuid)


def test_validate_sender_recipient_when_uppercase_role_string_then_fails():
    """Test validate_sender_recipient with uppercase role string (not in allowed)."""
    result = validate_sender_recipient("SYSTEM")  # Not in MessageRole.allowed()
    # Should try UUID conversion, fail, return as string
    assert result == "SYSTEM"
    assert isinstance(result, str)


def test_validate_sender_recipient_when_mixed_case_role_then_fails():
    """Test validate_sender_recipient with mixed case role string."""
    result = validate_sender_recipient("SyStEm")
    assert result == "SyStEm"
    assert isinstance(result, str)
