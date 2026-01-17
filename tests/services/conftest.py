# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for services tests.

This conftest provides reusable fixtures for testing service-related
functionality including hooks, backends, and registries.
"""

from __future__ import annotations

import pytest

from lionpride.services.types.hook import HookRegistry


@pytest.fixture
def hook_registry():
    """Create empty HookRegistry for testing."""
    return HookRegistry()
