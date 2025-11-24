# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operation specification defining input/output Operables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lionpride.types import Operable

__all__ = ("OperationSpec",)


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Specification for operation input/output validation.

    Defines:
    - input_operable: What inputs are valid
    - output_operable: What outputs are valid

    Used by IPU to validate operations before/after execution.
    """

    input_operable: Operable
    """Operable defining valid input fields and types"""

    output_operable: Operable
    """Operable defining valid output fields and types"""

    def validate_input_names(self, data: dict[str, Any]) -> bool:
        """Check if input field names are allowed.

        Args:
            data: Input data dict

        Returns:
            True if all names allowed

        Raises:
            ValueError: If disallowed field names present
        """
        return self.input_operable.check_allowed(*data.keys())

    def validate_output_names(self, data: dict[str, Any]) -> bool:
        """Check if output field names are allowed.

        Args:
            data: Output data dict

        Returns:
            True if all names allowed

        Raises:
            ValueError: If disallowed field names present
        """
        return self.output_operable.check_allowed(*data.keys())

    def __repr__(self) -> str:
        """String representation."""
        input_fields = self.input_operable.allowed()
        output_fields = self.output_operable.allowed()
        return f"OperationSpec(inputs={input_fields}, outputs={output_fields})"
