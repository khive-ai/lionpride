# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for functional tests.

Loads environment variables before tests run.
"""

import os

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Load environment variables before tests run."""
    # Load from project root .env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(project_root, ".env")

    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)


@pytest.fixture(scope="session", autouse=True)
def ensure_env_loaded():
    """Ensure environment is loaded for all tests."""
    # Already loaded in pytest_configure, but this is a safety net
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(project_root, ".env")

    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)
