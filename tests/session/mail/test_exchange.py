# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Exchange routing system."""

from uuid import uuid4

import pytest

from lionpride.libs import concurrency
from lionpride.session.mail import OUTBOX, Exchange, Mail


class TestExchangeRegistration:
    """Entity registration in Exchange."""

    def test_register_entity(self):
        """Register entity creates Flow mailbox."""
        exchange = Exchange()
        owner_id = uuid4()

        flow = exchange.register(owner_id)

        assert flow is not None
        assert flow.name == str(owner_id)
        assert exchange.has(owner_id)
        assert owner_id in exchange
        assert len(exchange) == 1

    def test_register_multiple_entities(self):
        """Multiple entities can be registered."""
        exchange = Exchange()
        ids = [uuid4() for _ in range(5)]

        for owner_id in ids:
            exchange.register(owner_id)

        assert len(exchange) == 5
        for owner_id in ids:
            assert exchange.has(owner_id)

    def test_register_duplicate_raises(self):
        """Cannot register same owner twice."""
        exchange = Exchange()
        owner_id = uuid4()

        exchange.register(owner_id)
        with pytest.raises(ValueError, match="already registered"):
            exchange.register(owner_id)

    def test_unregister_entity(self):
        """Unregister removes entity Flow."""
        exchange = Exchange()
        owner_id = uuid4()

        exchange.register(owner_id)
        flow = exchange.unregister(owner_id)

        assert flow is not None
        assert not exchange.has(owner_id)
        assert owner_id not in exchange
        assert len(exchange) == 0

    def test_unregister_unknown_returns_none(self):
        """Unregister unknown owner returns None."""
        exchange = Exchange()
        result = exchange.unregister(uuid4())
        assert result is None

    def test_get_flow(self):
        """Get returns entity's Flow."""
        exchange = Exchange()
        owner_id = uuid4()
        registered_flow = exchange.register(owner_id)

        flow = exchange.get(owner_id)

        assert flow is registered_flow
        assert flow.name == str(owner_id)

    def test_get_unknown_returns_none(self):
        """Get unknown owner returns None."""
        exchange = Exchange()
        assert exchange.get(uuid4()) is None

    def test_owner_ids(self):
        """owner_ids returns list of registered UUIDs."""
        exchange = Exchange()
        ids = [uuid4() for _ in range(3)]

        for owner_id in ids:
            exchange.register(owner_id)

        assert set(exchange.owner_ids) == set(ids)


class TestExchangeSend:
    """Sending mail through Exchange."""

    def test_send_creates_mail(self):
        """send() creates Mail and queues to outbox."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        mail = exchange.send(alice, bob, content="hello")

        assert isinstance(mail, Mail)
        assert mail.sender == alice
        assert mail.recipient == bob
        assert mail.content == "hello"

    def test_send_with_channel(self):
        """send() with channel namespace."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        mail = exchange.send(alice, bob, content="msg", channel="alerts")
        assert mail.channel == "alerts"

    def test_send_broadcast(self):
        """send() with None recipient is broadcast."""
        exchange = Exchange()
        alice = uuid4()
        exchange.register(alice)

        mail = exchange.send(alice, None, content="broadcast")

        assert mail.is_broadcast
        assert mail.recipient is None

    def test_send_unregistered_sender_raises(self):
        """send() from unregistered sender raises."""
        exchange = Exchange()
        with pytest.raises(ValueError, match="not registered"):
            exchange.send(uuid4(), uuid4(), content="test")


class TestExchangeSync:
    """Mail routing via sync."""

    @pytest.mark.asyncio
    async def test_sync_routes_direct_mail(self):
        """sync() routes mail to recipient inbox."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        exchange.send(alice, bob, content="hello bob")
        await exchange.sync()

        messages = exchange.receive(bob, sender=alice)
        assert len(messages) == 1
        assert messages[0].content == "hello bob"

    @pytest.mark.asyncio
    async def test_sync_routes_broadcast(self):
        """sync() routes broadcast to all except sender."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()
        carol = uuid4()

        exchange.register(alice)
        exchange.register(bob)
        exchange.register(carol)

        exchange.send(alice, None, content="broadcast")
        await exchange.sync()

        # Bob and Carol receive
        bob_msgs = exchange.receive(bob, sender=alice)
        carol_msgs = exchange.receive(carol, sender=alice)
        assert len(bob_msgs) == 1
        assert len(carol_msgs) == 1
        assert bob_msgs[0].content == "broadcast"

        # Alice doesn't receive own broadcast
        alice_msgs = exchange.receive(alice)
        assert len(alice_msgs) == 0

    @pytest.mark.asyncio
    async def test_sync_drops_unregistered_recipient(self):
        """Mail to unregistered recipient is dropped."""
        exchange = Exchange()
        alice = uuid4()
        unknown = uuid4()

        exchange.register(alice)

        # Send to unknown recipient - doesn't raise
        exchange.send(alice, unknown, content="lost mail")
        count = await exchange.sync()

        assert count == 0  # No deliveries

    @pytest.mark.asyncio
    async def test_multiple_syncs(self):
        """Multiple sync cycles work correctly."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        # First round
        exchange.send(alice, bob, content="msg1")
        await exchange.sync()

        # Second round
        exchange.send(alice, bob, content="msg2")
        await exchange.sync()

        messages = exchange.receive(bob, sender=alice)
        assert len(messages) == 2
        contents = {m.content for m in messages}
        assert contents == {"msg1", "msg2"}


class TestExchangeReceive:
    """Receiving mail from Exchange."""

    @pytest.mark.asyncio
    async def test_receive_all(self):
        """receive() without sender filter gets all mail."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()
        carol = uuid4()

        exchange.register(alice)
        exchange.register(bob)
        exchange.register(carol)

        exchange.send(bob, alice, content="from bob")
        exchange.send(carol, alice, content="from carol")
        await exchange.sync()

        all_msgs = exchange.receive(alice)
        assert len(all_msgs) == 2

    @pytest.mark.asyncio
    async def test_receive_filtered_by_sender(self):
        """receive() with sender only gets that sender's mail."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()
        carol = uuid4()

        exchange.register(alice)
        exchange.register(bob)
        exchange.register(carol)

        exchange.send(bob, alice, content="from bob")
        exchange.send(carol, alice, content="from carol")
        await exchange.sync()

        bob_msgs = exchange.receive(alice, sender=bob)
        assert len(bob_msgs) == 1
        assert bob_msgs[0].content == "from bob"

    def test_receive_unregistered_returns_empty(self):
        """receive() for unregistered owner returns empty list."""
        exchange = Exchange()
        assert exchange.receive(uuid4()) == []


class TestExchangePopMail:
    """Popping mail from inbox."""

    @pytest.mark.asyncio
    async def test_pop_mail(self):
        """pop_mail() returns and removes next mail."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        exchange.send(alice, bob, content="msg1")
        exchange.send(alice, bob, content="msg2")
        await exchange.sync()

        mail1 = exchange.pop_mail(bob, alice)
        assert mail1 is not None
        assert mail1.content == "msg1"

        mail2 = exchange.pop_mail(bob, alice)
        assert mail2 is not None
        assert mail2.content == "msg2"

        mail3 = exchange.pop_mail(bob, alice)
        assert mail3 is None  # Empty

    @pytest.mark.asyncio
    async def test_pop_mail_fifo_order(self):
        """pop_mail() returns in FIFO order."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        for i in range(5):
            exchange.send(alice, bob, content=f"msg{i}")
        await exchange.sync()

        for i in range(5):
            mail = exchange.pop_mail(bob, alice)
            assert mail.content == f"msg{i}"

    def test_pop_mail_unregistered_returns_none(self):
        """pop_mail() for unregistered returns None."""
        exchange = Exchange()
        assert exchange.pop_mail(uuid4(), uuid4()) is None


class TestExchangeCollect:
    """Collect operation details."""

    @pytest.mark.asyncio
    async def test_collect_returns_count(self):
        """collect() returns number of unique mails routed."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()
        carol = uuid4()

        exchange.register(alice)
        exchange.register(bob)
        exchange.register(carol)

        exchange.send(alice, bob, content="direct")
        exchange.send(alice, None, content="broadcast")  # Goes to bob and carol

        count = await exchange.collect(alice)

        # 1 direct + 1 broadcast = 2 unique mails
        assert count == 2

    @pytest.mark.asyncio
    async def test_collect_unregistered_raises(self):
        """collect() for unregistered owner raises."""
        exchange = Exchange()
        with pytest.raises(ValueError, match="not registered"):
            await exchange.collect(uuid4())


class TestExchangeConcurrency:
    """Concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_sends(self):
        """Multiple concurrent send operations."""
        exchange = Exchange()
        entities = [uuid4() for _ in range(10)]

        for eid in entities:
            exchange.register(eid)

        # All entities send to first entity concurrently
        target = entities[0]
        for sender in entities[1:]:
            exchange.send(sender, target, content=f"from {sender}")

        await exchange.sync()

        messages = exchange.receive(target)
        assert len(messages) == 9

    @pytest.mark.asyncio
    async def test_concurrent_syncs(self):
        """Multiple concurrent sync operations (should be safe)."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        for i in range(10):
            exchange.send(alice, bob, content=f"msg{i}")

        # Multiple concurrent syncs
        results = await concurrency.gather(
            exchange.sync(),
            exchange.sync(),
            exchange.sync(),
        )

        # Mail should be routed exactly once (first sync gets them)
        total = sum(results)
        assert total == 10

        messages = exchange.receive(bob)
        assert len(messages) == 10


class TestExchangeRunLoop:
    """Continuous run loop."""

    @pytest.mark.asyncio
    async def test_run_and_stop(self):
        """run() can be stopped with stop()."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)

        async def send_and_stop():
            await concurrency.sleep(0.05)
            exchange.send(alice, bob, content="hello")
            await concurrency.sleep(0.15)
            exchange.stop()

        async with concurrency.create_task_group() as tg:
            tg.start_soon(send_and_stop)
            await exchange.run(interval=0.05)

        # Mail should have been routed
        messages = exchange.receive(bob)
        assert len(messages) == 1


class TestExchangeRepr:
    """Exchange string representation."""

    def test_repr(self):
        """Repr shows entity count and pending."""
        exchange = Exchange()
        alice = uuid4()
        bob = uuid4()

        exchange.register(alice)
        exchange.register(bob)
        exchange.send(alice, bob, content="pending")

        repr_str = repr(exchange)

        assert "Exchange(" in repr_str
        assert "entities=2" in repr_str
        assert "pending_out=1" in repr_str
