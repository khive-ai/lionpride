# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for operations tests.

Operations are now registered per-Session via Session._register_default_operations().
No global dispatcher needed.
"""

import pytest


@pytest.fixture
def session():
    """Create a fresh Session with default operations registered."""
    from lionpride.session import Session

    return Session()


@pytest.fixture
def ipu():
    """Create a fresh IPU."""
    from lionpride.ipu import IPU

    return IPU()


@pytest.fixture
def session_with_ipu(session, ipu):
    """Create Session registered with IPU."""
    ipu.register_session(session)
    return session, ipu
