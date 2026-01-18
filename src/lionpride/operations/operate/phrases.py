# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.core.event import EventStatus
from lionpride.errors import (
    AccessError,
    ConfigurationError,
    ExecutionError,
    NotFoundError,
    ValidationError,
)
from lionpride.services.types import NormalizedResponse
from lionpride.session.session import Branch
from lionpride.types import is_sentinel

from .types import GenerateParams

if TYPE_CHECKING:
    from lionpride.services.types import Calling, iModel
    from lionpride.session import Session
    from lionpride.types import Operable


def genai_model_must_be_configured(
    session: Session, params: GenerateParams, *, operation: str = "operation"
) -> None:
    """Raises ConfigurationError if no imodel in params or session default."""
    if is_sentinel(params.imodel) and session.default_generate_model is None:
        raise ConfigurationError(
            f"{operation} requires 'imodel' in generate params or session.default_generate_model"
        )


def resource_must_exist_in_session(session: Session, name: str) -> None:
    """Raises NotFoundError if service not registered in session."""
    if not session.services.has(name):
        raise NotFoundError(
            f"Service '{name}' not found in session services",
            details={"available": session.services.list_names()},
        )


def resource_must_be_accessible_by_branch(branch: Branch, name: str) -> None:
    """Raises AccessError if branch lacks access to named resource."""
    if name not in branch.resources:
        raise AccessError(
            f"Branch '{branch.name}' has no access to resource '{name}'",
            details={
                "branch": branch.name,
                "resource": name,
                "available": list(branch.resources),
            },
        )


def capabilities_must_be_subset_of_branch(branch: Branch, capabilities: set[str]) -> None:
    """Raises AccessError if branch missing required capabilities."""
    if not capabilities.issubset(branch.capabilities):
        missing = capabilities - branch.capabilities
        raise AccessError(
            f"Branch '{branch.name}' missing capabilities: {missing}",
            details={
                "requested": sorted(capabilities),
                "available": sorted(branch.capabilities),
            },
        )


def capabilities_must_be_subset_of_operable(operable: Operable, capabilities: set[str]) -> None:
    """Raises ValidationError if capabilities exceed operable's allowed set."""
    allowed = operable.allowed()
    if not capabilities.issubset(allowed):
        missing = capabilities - allowed
        raise ValidationError(
            f"Requested capabilities not in operable: {missing}",
            details={
                "requested": sorted(capabilities),
                "available": sorted(allowed),
            },
        )


def response_must_be_completed(calling: Calling) -> None:
    """Raises ExecutionError if calling status is not COMPLETED."""
    if calling.execution.status != EventStatus.COMPLETED:
        raise ExecutionError(
            "Generation did not complete successfully",
            details=calling.execution.to_dict(),
            retryable=True,
        )


def resolve_branch_exists_in_session(session: Session, branch: Branch | str) -> Branch:
    """Return Branch or raise NotFoundError if not in session."""
    if (b_ := session.get_branch(branch, None)) is None:
        raise NotFoundError(f"Branch '{branch}' does not exist in session")
    return b_


def resolve_genai_model_exists_in_session(
    session: Session, params: GenerateParams
) -> tuple[iModel, dict[str, Any]]:
    """Return (iModel, kwargs) or raise ConfigurationError/ValidationError."""
    genai_model_must_be_configured(session, params, operation="generate")

    imodel_kw = params.imodel_kwargs or {}
    if not isinstance(imodel_kw, dict):
        raise ValidationError("'imodel_kwargs' must be a dict if provided")

    imodel = session.services.get(params.imodel or session.default_generate_model, None)
    if imodel is None:
        raise ConfigurationError("Provided generative model not found in session services")

    return imodel, imodel_kw


def resolve_response_is_normalized(calling: Calling) -> NormalizedResponse:
    """Return NormalizedResponse or raise ExecutionError if coercion fails."""
    from lionpride.ln import to_dict

    response = calling.response

    if is_sentinel(response):
        raise ExecutionError(
            "Generation completed but no response was returned",
            retryable=False,
        )

    # Already a NormalizedResponse
    if isinstance(response, NormalizedResponse):
        return response

    # Try to normalize via model_validate
    try:
        return NormalizedResponse.model_validate(to_dict(response))
    except Exception as e:
        raise ExecutionError(
            f"Response cannot be normalized: {e}",
            retryable=False,
        ) from e


def resolve_generate_params(params: Any) -> GenerateParams:
    """Extract GenerateParams from composite or raise ValidationError."""
    if not hasattr(params, "generate"):
        raise ValidationError("Params object missing 'generate'")
    if not isinstance(params.generate, GenerateParams):
        raise ValidationError("'generate' field is not of type GenerateParams")
    return params.generate
