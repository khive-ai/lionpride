# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.types import Operable

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = (
    "Operative",
    "create_action_operative",
    "create_operative_from_model",
)


class Operative:
    """Spec/Operable-based validator with two-tier strategy (strict → fuzzy).

    Attributes:
        name: Operation name
        adapter: Framework adapter ("pydantic")
        strict: Strict validation (raise on error)
        auto_retry_parse: Enable fuzzy fallback
        max_retries: Max validation attempts
        operable: Single Operable with all specs
        request_exclude: Fields excluded from request model
    """

    def __init__(
        self,
        operable: Operable,
        *,
        name: str | None = None,
        adapter: str = "pydantic",
        strict: bool = False,
        auto_retry_parse: bool = True,
        max_retries: int = 3,
        request_exclude: set[str] | None = None,
    ):
        """Initialize Operative with Operable."""
        self.name = name or operable.name or "Operative"
        self.adapter = adapter
        self.strict = strict
        self.auto_retry_parse = auto_retry_parse
        self.max_retries = max_retries
        self.operable = operable
        self.request_exclude = request_exclude or set()

        # Cached models
        self._request_model_cls: type[BaseModel] | None = None
        self._response_model_cls: type[BaseModel] | None = None

        # Response state
        self.response_model: BaseModel | None = None
        self.response_str_dict: str | None = None
        self._should_retry: bool | None = None

    def create_request_model(self) -> type[BaseModel]:
        """Materialize request model (excludes runtime fields)."""
        if self._request_model_cls:
            return self._request_model_cls

        self._request_model_cls = self.operable.create_model(
            adapter="pydantic",  # type: ignore[arg-type]
            model_name=f"{self.name}Request",
            exclude=self.request_exclude,
        )
        return self._request_model_cls

    def create_response_model(self) -> type[BaseModel]:
        """Materialize response model.

        For single model-based spec (from create_operative_from_model),
        returns the original Pydantic model directly.

        For multi-spec operables, creates a composite model.
        """
        if self._response_model_cls:
            return self._response_model_cls

        # Check for single model-based spec - return original model directly
        specs = self.operable.get_specs()
        if len(specs) == 1:
            base_type = specs[0].base_type
            if hasattr(base_type, "model_fields"):
                # Single Pydantic model spec - use it directly
                self._response_model_cls = base_type
                return self._response_model_cls

        # Multi-spec or non-model specs: create composite model
        # Ensure request exists first
        if not self._request_model_cls:
            self.create_request_model()

        # Response inherits from request, adds excluded fields
        self._response_model_cls = self.operable.create_model(
            adapter="pydantic",  # type: ignore[arg-type]
            model_name=f"{self.name}Response",
            base_type=self._request_model_cls,
        )
        return self._response_model_cls

    def validate_response(self, text: str, strict: bool | None = None) -> Any:
        """Validate response with two-tier strategy (strict → fuzzy).

        Args:
            text: Raw response text
            strict: Override strict setting

        Returns:
            Validated model instance or None if validation fails
        """
        strict = self.strict if strict is None else strict

        if not self._response_model_cls:
            self.create_response_model()

        assert self._response_model_cls is not None  # Satisfy mypy

        # Get adapter
        from lionpride.types.spec_adapters.pydantic_field import PydanticSpecAdapter

        adapter_cls = PydanticSpecAdapter

        # First attempt: strict validation
        try:
            self.response_model = adapter_cls.validate_response(
                text,
                self._response_model_cls,
                strict=strict,
                fuzzy_parse=True,
            )
            self._should_retry = False
            return self.response_model

        except Exception as e:
            self.response_str_dict = text
            self._should_retry = strict

            if strict:
                raise e

            # Second attempt: fuzzy fallback
            if self.auto_retry_parse and not strict:
                try:
                    self.response_model = adapter_cls.validate_response(
                        text,
                        self._response_model_cls,
                        strict=False,
                        fuzzy_parse=True,
                    )
                    self._should_retry = False
                    return self.response_model
                except Exception:
                    pass

            return None

    @property
    def request_type(self) -> type[BaseModel]:
        """Get request model type."""
        if not self._request_model_cls:
            self.create_request_model()
        assert self._request_model_cls is not None  # Satisfy mypy
        return self._request_model_cls

    @property
    def response_type(self) -> type[BaseModel]:
        """Get response model type."""
        if not self._response_model_cls:
            self.create_response_model()
        assert self._response_model_cls is not None  # Satisfy mypy
        return self._response_model_cls


def create_operative_from_model(
    response_model: type[BaseModel],
    *,
    name: str | None = None,
    strict: bool = False,
    auto_retry_parse: bool = True,
) -> Operative:
    """Create Operative from Pydantic model as a single model-based spec.

    Creates a single Spec with the Pydantic model as base_type, enabling
    LNDL parsing with namespaced lvars: <lvar Model.field alias>value</lvar>

    Args:
        response_model: Pydantic BaseModel class
        name: Operation name (defaults to model class name)
        strict: Strict validation
        auto_retry_parse: Enable fuzzy fallback

    Returns:
        Operative instance with single model-based spec
    """
    from lionpride.types import Spec

    # Create single model-based spec
    # spec name is lowercase for LNDL OUT{} matching
    spec_name = (name or response_model.__name__).lower()
    spec = Spec(
        base_type=response_model,
        name=spec_name,
    )

    # Create Operable with single spec
    operable = Operable(
        specs=(spec,),
        name=name or response_model.__name__,
    )

    # Create Operative
    return Operative(
        operable=operable,
        name=name or response_model.__name__,
        strict=strict,
        auto_retry_parse=auto_retry_parse,
    )


def create_action_operative(
    base_model: type[BaseModel] | None = None,
    *,
    reason: bool = False,
    actions: bool = True,
    name: str | None = None,
    strict: bool = False,
    auto_retry_parse: bool = True,
) -> Operative:
    """Create Operative with optional reasoning and action support.

    With LNDL, actions are handled via <lact> tags, not as specs.
    This function creates specs for the data model and optionally reason.

    Action execution is handled by:
    - LLM writes <lact> tags in response
    - LNDL parser extracts these as action calls
    - operate() executes the actions via act()
    - Results are integrated into final response

    Args:
        base_model: Optional Pydantic model for the main data
        reason: Include reasoning field for chain-of-thought
        actions: Enable action execution (via <lact> tags)
        name: Operation name
        strict: Strict validation
        auto_retry_parse: Enable fuzzy fallback

    Returns:
        Operative with action support

    Example:
        >>> from pydantic import BaseModel
        >>> class Analysis(BaseModel):
        ...     summary: str
        ...     metrics: dict
        >>> operative = create_action_operative(
        ...     base_model=Analysis, reason=True, actions=True, name="analyze"
        ... )
        >>> # LNDL specs: analyze (Analysis model), reason (Reason model)
        >>> # Actions handled via <lact> tags, not as specs
    """
    from lionpride.types import Spec

    from ..models import Reason

    # Start with base model as a SINGLE spec if provided
    specs = []
    if base_model:
        # Create ONE spec for the entire base model, not individual field specs
        spec_name = (name or base_model.__name__).lower()
        spec = Spec(
            base_type=base_model,
            name=spec_name,
        )
        specs.append(spec)

    # Add reasoning field if requested
    if reason:
        specs.append(
            Spec(
                base_type=Reason,
                name="reason",
                default=None,
            )
        )

    # Add action specs if requested - they're part of the typed namespace
    if actions:
        from ..models import ActionRequestModel, ActionResponseModel

        specs.append(
            Spec(
                base_type=list[ActionRequestModel],
                name="action_requests",
                default=None,
            )
        )
        specs.append(
            Spec(
                base_type=list[ActionResponseModel],
                name="action_responses",
                default=None,
            )
        )

    # Create Operable with all specs including actions
    operable = Operable(
        specs=tuple(specs),
        name=name or (base_model.__name__ if base_model else "ActionOperative"),
    )

    # Store actions flag in the operative for operate() to check
    # When actions=True, exclude action_responses from request model
    request_exclude = {"action_responses"} if actions else set()

    operative = Operative(
        operable=operable,
        name=name or (base_model.__name__ if base_model else "ActionOperative"),
        strict=strict,
        auto_retry_parse=auto_retry_parse,
        request_exclude=request_exclude,
    )

    # Add a flag to indicate this operative supports actions
    operative._supports_actions = actions

    return operative
