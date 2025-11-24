from typing import Any

from lionpride.types import Operable, Spec

_DEFAULT_OPERABLE: Operable | None = None


def create_default_specs() -> tuple[Spec, ...]:
    func = Spec(
        str,
        name="function",
        description=(
            "Name of function from available tool_schemas. "
            "CRITICAL: Never invent function names. Only use functions "
            "provided in the tool schemas."
        ),
    )
    args = Spec(
        dict,
        name="arguments",
        default_factory=dict,
        description=(
            "Arguments dictionary matching the function's schema. "
            "Keys must match parameter names from tool_schemas."
        ),
    )
    output = Spec(
        Any,
        name="output",
        description="Function output (success) or error message (failure)",
    ).as_nullable()
    reasoning = Spec(
        str,
        name="reasoning",
        description=(
            "Explain your reasoning step-by-step before taking actions. "
            "This helps ensure clarity and correctness in your decisions."
        ),
    ).as_nullable()
    confidence = Spec(
        float,
        name="confidence",
        ge=0.0,
        le=1.0,
        description=(
            "A confidence score between 0 and 1 indicating how sure you are "
            "about your reasoning or action."
        ),
    ).as_nullable()
    return (func, args, output, reasoning, confidence)


def get_default_operable() -> Operable:
    global _DEFAULT_OPERABLE
    if _DEFAULT_OPERABLE is None:
        specs = create_default_specs()
        _DEFAULT_OPERABLE = Operable(specs, name="default_operable")
    return _DEFAULT_OPERABLE
