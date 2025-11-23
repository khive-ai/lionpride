# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for operations tests."""

import pytest

from lionpride.operations.dispatcher import get_dispatcher, register_operation
from lionpride.operations.operate import generate, operate, react


def _register_default_operations():
    """Re-register default operations (generate, operate, react)."""
    dispatcher = get_dispatcher()
    if not dispatcher.is_registered("generate"):
        dispatcher.register("generate", generate)
    if not dispatcher.is_registered("operate"):
        dispatcher.register("operate", operate)
    if not dispatcher.is_registered("react"):
        dispatcher.register("react", react)


@pytest.fixture(autouse=True)
def cleanup_dispatcher():
    """Clear global dispatcher after each test to prevent state pollution.

    This fixture runs automatically for all tests in this directory and subdirectories.
    It ensures that operation registrations from one test don't affect other tests.
    The default operations (generate, operate, react) are re-registered before each test.
    """
    # Re-register defaults before test
    _register_default_operations()
    yield  # Run test
    # Cleanup after test
    dispatcher = get_dispatcher()
    dispatcher.clear()
