# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""IPU (Intelligence Processing Unit) - Validated execution context for operations.

The IPU pattern from lionagi v0.2.2: validation → structure → usefulness
"""

from .ipu import IPU, get_current_ipu
from .operation_spec import OperationSpec

__all__ = ("IPU", "OperationSpec", "get_current_ipu")
