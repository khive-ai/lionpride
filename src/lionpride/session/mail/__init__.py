# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Mail: Communication primitives for multi-entity coordination.

This module provides infrastructure for groupchat and gossip patterns:
- Mail: A message between entities (sender, recipient, content, channel)
- Exchange: Routes mail between registered entities (wraps Pile[Flow])

Each registered entity gets a Flow[Mail, Progression] as their mailbox,
with "outbox" and "inbox_{sender}" progressions for routing.

Example:
    ```python
    import asyncio
    from lionpride.session.mail import Mail, Exchange
    from uuid import uuid4


    async def main():
        alice_id, bob_id = uuid4(), uuid4()

        # Create exchange and register entities
        exchange = Exchange()
        exchange.register(alice_id)
        exchange.register(bob_id)

        # Alice sends to Bob
        exchange.send(alice_id, bob_id, content="Hello!")

        # Exchange syncs (routes mail)
        await exchange.sync()

        # Bob receives
        messages = exchange.receive(bob_id, sender=alice_id)
        print(messages[0].content)  # "Hello!"


    asyncio.run(main())
    ```
"""

from .exchange import OUTBOX, Exchange
from .mail import Mail

__all__ = (
    "OUTBOX",
    "Exchange",
    "Mail",
)
