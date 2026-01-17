# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Mail primitive."""

from uuid import uuid4

import pytest

from lionpride.session.mail import Mail


class TestMailBasic:
    """Basic Mail functionality."""

    def test_create_direct_mail(self):
        """Mail with explicit recipient."""
        sender = uuid4()
        recipient = uuid4()
        mail = Mail(sender=sender, recipient=recipient, content="hello")

        assert mail.sender == sender
        assert mail.recipient == recipient
        assert mail.content == "hello"
        assert mail.channel is None
        assert mail.is_direct
        assert not mail.is_broadcast

    def test_create_broadcast_mail(self):
        """Mail without recipient is broadcast."""
        sender = uuid4()
        mail = Mail(sender=sender, content="broadcast message")

        assert mail.sender == sender
        assert mail.recipient is None
        assert mail.is_broadcast
        assert not mail.is_direct

    def test_mail_with_channel(self):
        """Mail with optional channel namespace."""
        sender = uuid4()
        mail = Mail(sender=sender, content="msg", channel="updates")

        assert mail.channel == "updates"

    def test_mail_has_element_identity(self):
        """Mail inherits Element identity."""
        mail = Mail(sender=uuid4(), content="test")

        assert mail.id is not None
        assert mail.created_at is not None

    def test_mail_content_any_type(self):
        """Mail content can be any type."""
        sender = uuid4()

        # String
        mail1 = Mail(sender=sender, content="string")
        assert mail1.content == "string"

        # Dict
        mail2 = Mail(sender=sender, content={"key": "value"})
        assert mail2.content == {"key": "value"}

        # List
        mail3 = Mail(sender=sender, content=[1, 2, 3])
        assert mail3.content == [1, 2, 3]

        # None
        mail4 = Mail(sender=sender, content=None)
        assert mail4.content is None


class TestMailCoercion:
    """UUID coercion from strings."""

    def test_sender_coerced_from_string(self):
        """Sender UUID can be passed as string."""
        sender_str = str(uuid4())
        mail = Mail(sender=sender_str, content="test")

        assert isinstance(mail.sender, type(uuid4()))
        assert str(mail.sender) == sender_str

    def test_recipient_coerced_from_string(self):
        """Recipient UUID can be passed as string."""
        sender = uuid4()
        recipient_str = str(uuid4())
        mail = Mail(sender=sender, recipient=recipient_str, content="test")

        assert isinstance(mail.recipient, type(uuid4()))
        assert str(mail.recipient) == recipient_str

    def test_recipient_none_stays_none(self):
        """None recipient is not coerced."""
        mail = Mail(sender=uuid4(), recipient=None, content="test")
        assert mail.recipient is None


class TestMailRepr:
    """Mail string representation."""

    def test_repr_direct(self):
        """Repr shows sender->recipient."""
        sender = uuid4()
        recipient = uuid4()
        mail = Mail(sender=sender, recipient=recipient, content="test")
        repr_str = repr(mail)

        assert str(sender)[:8] in repr_str
        assert str(recipient)[:8] in repr_str
        assert "Mail(" in repr_str

    def test_repr_broadcast(self):
        """Repr shows broadcast for no recipient."""
        mail = Mail(sender=uuid4(), content="test")
        assert "broadcast" in repr(mail)

    def test_repr_with_channel(self):
        """Repr includes channel if present."""
        mail = Mail(sender=uuid4(), content="test", channel="alerts")
        assert "channel=alerts" in repr(mail)
