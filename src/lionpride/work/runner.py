# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Report execution - delegates to executor.

This module provides backward-compatible aliases to the new executor-based
implementation.
"""

from .executor import execute_report, stream_report

# Backward-compatible alias
flow_report = execute_report
stream_flow_report = stream_report

__all__ = ("flow_report", "stream_flow_report")
