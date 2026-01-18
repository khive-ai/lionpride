# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.core.event import EventStatus
from lionpride.errors import ConfigurationError, ExecutionError, ValidationError
from lionpride.services.types import NormalizedResponse
from lionpride.session.session import (
    capabilities_must_be_subset_of_branch,
    resolve_branch_exists_in_session,
    resource_must_be_accessible_by_branch,
    resource_must_exist_in_session,
)
from lionpride.types import is_sentinel

from .types import GenerateParams, ParseParams

if TYPE_CHECKING:
    from lionpride.services.types import Calling, iModel
    from lionpride.session import Session
    from lionpride.types import Operable


__all__ = (
    "capabilities_must_be_subset_of_branch",
    "capabilities_must_be_subset_of_operable",
    "genai_model_must_be_configured",
    "resolve_branch_exists_in_session",
    "resolve_genai_model_exists_in_session",
    "resolve_generate_params",
    "resolve_parse_params",
    "resolve_response_is_normalized",
    "resource_must_be_accessible_by_branch",
    "resource_must_exist_in_session",
    "response_must_be_completed",
)


def genai_model_must_be_configured(
    session: Session, params: GenerateParams, *, operation: str = "operation"
) -> None:
    """Raises ConfigurationError if no imodel in params or session default."""
    if is_sentinel(params.imodel) and session.default_generate_model is None:
        raise ConfigurationError(
            f"{operation} requires 'imodel' in generate params or session.default_generate_model"
        )


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


def resolve_generate_params(params: Any) -> GenerateParams:
    """Extract GenerateParams from composite or raise ValidationError."""
    if not hasattr(params, "generate"):
        raise ValidationError("Params object missing 'generate'")
    if not isinstance(params.generate, GenerateParams):
        raise ValidationError("'generate' field is not of type GenerateParams")
    return params.generate


def resolve_parse_params(params: Any) -> ParseParams:
    """Extract ParseParams from composite or raise ValidationError."""
    if not hasattr(params, "parse"):
        raise ValidationError("Params object missing 'parse'")
    if not isinstance(params.parse, ParseParams):
        raise ValidationError("'parse' field is not of type ParseParams")
    return params.parse


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


def resolve_response_is_normalized(calling: Calling) -> NormalizedResponse:
    """Return NormalizedResponse or raise ExecutionError if coercion fails."""
    from lionpride.ln import to_dict

    response = calling.response

    if is_sentinel(response):
        raise ExecutionError(
            "Generation completed but no response was returned",
            retryable=False,
        )

    if isinstance(response, NormalizedResponse):
        return response

    try:
        return NormalizedResponse.model_validate(to_dict(response))
    except Exception as e:
        raise ExecutionError(
            f"Response cannot be normalized: {e}",
            retryable=False,
        ) from e
