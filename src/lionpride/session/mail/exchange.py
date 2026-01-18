# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Exchange: Routes mail between entity Flows.

Exchange wraps a Pile of Flows, where each Flow represents one entity's
mailbox. It handles registration, routing, and synchronization.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import PrivateAttr

from lionpride.core import Element, Flow, Pile, Progression
from lionpride.errors import ExistsError
from lionpride.libs import concurrency

from .mail import Mail

__all__ = ("OUTBOX", "Exchange")

OUTBOX = "outbox"


def _inbox_name(sender: UUID) -> str:
    """Get inbox progression name for a sender."""
    return f"inbox_{sender}"


class Exchange(Element):
    """Routes mail between entity Flows.

    Exchange manages a Pile of Flows, where each Flow is an entity's mailbox:
    - Flow.name = str(owner_id) for routing
    - Flow.items = Mail pile
    - Flow.progressions = "outbox" + "inbox_{sender}" queues

    Provides async-safe sync operations with proper locking.
    """

    flows: Pile[Flow[Mail, Progression]] = None  # type: ignore
    _owner_index: dict[UUID, UUID] = PrivateAttr(default_factory=dict)  # owner_id -> flow.id
    _stop: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize exchange with empty flows pile."""
        super().__init__(**kwargs)
        if self.flows is None:
            self.flows = Pile()

    def register(self, owner_id: UUID) -> Flow[Mail, Progression]:
        """Create and register a Flow for an entity.

        Args:
            owner_id: UUID of the entity

        Returns:
            The created Flow (entity's mailbox)

        Raises:
            ValueError: If owner already registered
        """
        if owner_id in self._owner_index:
            raise ValueError(f"Owner {owner_id} already registered")

        flow: Flow[Mail, Progression] = Flow(
            name=str(owner_id),
            item_type={Mail},
            strict_type=True,
        )
        flow.add_progression(Progression(name=OUTBOX))

        self.flows.add(flow)
        self._owner_index[owner_id] = flow.id

        return flow

    def unregister(self, owner_id: UUID) -> Flow[Mail, Progression] | None:
        """Unregister an entity's Flow.

        Args:
            owner_id: UUID of the entity

        Returns:
            The removed Flow, or None if not found
        """
        flow_id = self._owner_index.pop(owner_id, None)
        if flow_id is None:
            return None
        return self.flows.pop(flow_id, None)

    def get(self, owner_id: UUID) -> Flow[Mail, Progression] | None:
        """Get an entity's Flow by owner ID."""
        flow_id = self._owner_index.get(owner_id)
        if flow_id is None:
            return None
        return self.flows.get(flow_id, None)

    def has(self, owner_id: UUID) -> bool:
        """Check if an entity is registered."""
        return owner_id in self._owner_index

    @property
    def owner_ids(self) -> list[UUID]:
        """List of registered owner UUIDs."""
        return list(self._owner_index.keys())

    async def collect(self, owner_id: UUID) -> int:
        """Collect outbound mail from an entity's Flow and route to recipients.

        Uses concurrent delivery - collects mail under lock, then delivers
        in parallel to recipient Flows (each Flow has its own lock).

        Args:
            owner_id: Entity whose outbox to collect from

        Returns:
            Number of mail items collected and routed

        Raises:
            ValueError: If owner not registered
        """
        # Phase 1: Collect mail and determine deliveries (under lock)
        deliveries: list[tuple[UUID, Mail]] = []

        async with self.flows:
            flow = self.get(owner_id)
            if flow is None:
                raise ValueError(f"Owner {owner_id} not registered")

            outbox = flow.get_progression(OUTBOX)

            while len(outbox) > 0:
                mail_id = outbox.popleft()
                mail = flow.items.pop(mail_id, None)
                if mail is None:
                    continue

                if mail.is_broadcast:
                    # Queue delivery to all except sender
                    # Copy mail for each recipient to prevent shared mutable state
                    for other_id in self._owner_index:
                        if other_id != owner_id:
                            try:
                                mail_copy = mail.model_copy(deep=True)
                            except Exception:
                                # Content not deep-copyable, use shallow copy
                                # Caller should ensure broadcast content is immutable
                                mail_copy = mail.model_copy()
                            deliveries.append((other_id, mail_copy))
                elif mail.recipient is not None and mail.recipient in self._owner_index:
                    # Queue direct delivery
                    deliveries.append((mail.recipient, mail))
                # If recipient not registered, mail is dropped

        # Phase 2: Deliver concurrently (lock released, each Flow has own lock)
        # Use return_exceptions=True so one failure doesn't cancel others
        if deliveries:
            results = await concurrency.gather(
                *[self._deliver_to(recipient_id, mail) for recipient_id, mail in deliveries],
                return_exceptions=True,
            )
            # Log failures but don't raise - mail delivery is best-effort
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    _recipient_id, _mail = deliveries[i]
                    # Silently drop failed deliveries (recipient may have unregistered)

        # Count unique mail items (broadcast counts as 1, not N)
        unique_mails = {mail.id for _, mail in deliveries}
        return len(unique_mails)

    async def _deliver_to(self, recipient_id: UUID, mail: Mail) -> None:
        """Deliver mail to a recipient's inbox.

        Safe to call concurrently - each Flow has its own lock.
        Handles gracefully if recipient was unregistered.
        """
        recipient_flow = self.get(recipient_id)
        if recipient_flow is None:
            return  # Recipient unregistered, drop mail

        # Ensure inbox for this sender exists (idempotent creation)
        inbox_name = _inbox_name(mail.sender)
        try:
            recipient_flow.add_progression(Progression(name=inbox_name))
        except ExistsError:
            pass  # Already exists, that's fine

        # Add mail to recipient's flow
        recipient_flow.add_item(mail, progressions=inbox_name)

    async def collect_all(self) -> int:
        """Collect and route mail from all entities.

        Gracefully handles entities that unregister during iteration.

        Returns:
            Total number of mail items routed
        """
        total = 0
        for owner_id in list(self._owner_index.keys()):
            try:
                total += await self.collect(owner_id)
            except ValueError:
                # Owner unregistered mid-iteration, skip
                continue
        return total

    async def sync(self) -> int:
        """Synchronize all mailboxes - collect and route all pending mail.

        Returns:
            Number of mail items routed
        """
        return await self.collect_all()

    async def run(self, interval: float = 1.0) -> None:
        """Run continuous sync loop.

        Args:
            interval: Seconds between sync cycles
        """
        self._stop = False
        while not self._stop:
            await self.sync()
            await concurrency.sleep(interval)

    def stop(self) -> None:
        """Stop the continuous run loop."""
        self._stop = True

    # Convenience methods for working with entity Flows

    def send(
        self,
        sender: UUID,
        recipient: UUID | None,
        content: Any,
        channel: str | None = None,
    ) -> Mail:
        """Create and queue mail from sender to recipient.

        Args:
            sender: Sending entity's UUID
            recipient: Receiving entity's UUID (None for broadcast)
            content: Mail content
            channel: Optional channel/topic

        Returns:
            The created Mail

        Raises:
            ValueError: If sender not registered
        """
        flow = self.get(sender)
        if flow is None:
            raise ValueError(f"Sender {sender} not registered")

        mail = Mail(sender=sender, recipient=recipient, content=content, channel=channel)
        flow.add_item(mail, progressions=OUTBOX)
        return mail

    def receive(self, owner_id: UUID, sender: UUID | None = None) -> list[Mail]:
        """Get pending inbound mail for an entity.

        Args:
            owner_id: Entity to get mail for
            sender: If provided, only return mail from this sender

        Returns:
            List of pending mail (does not remove them)
        """
        flow = self.get(owner_id)
        if flow is None:
            return []

        result = []
        # Iterate over progressions Pile (public API) instead of _progression_names
        for progression in flow.progressions:
            prog_name = progression.name
            if prog_name and prog_name.startswith("inbox_"):
                if sender is not None:
                    expected_name = _inbox_name(sender)
                    if prog_name != expected_name:
                        continue
                for mail_id in progression:
                    mail = flow.items.get(mail_id, None)
                    if mail is not None:
                        result.append(mail)
        return result

    def pop_mail(self, owner_id: UUID, sender: UUID) -> Mail | None:
        """Pop next mail from sender's inbox (FIFO).

        Args:
            owner_id: Receiving entity
            sender: Sender whose mail to pop

        Returns:
            Next mail or None
        """
        flow = self.get(owner_id)
        if flow is None:
            return None

        inbox_name = _inbox_name(sender)
        try:
            inbox = flow.get_progression(inbox_name)
        except KeyError:
            return None

        if len(inbox) == 0:
            return None

        mail_id = inbox.popleft()
        return flow.items.pop(mail_id, None)

    def __len__(self) -> int:
        """Number of registered entities."""
        return len(self._owner_index)

    def __contains__(self, owner_id: UUID) -> bool:
        """Check if entity is registered."""
        return self.has(owner_id)

    def __repr__(self) -> str:
        pending = 0
        for flow in self.flows:
            try:
                outbox = flow.get_progression(OUTBOX)
                pending += len(outbox)
            except KeyError:
                pass  # No outbox progression
        return f"Exchange(entities={len(self)}, pending_out={pending})"
