# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from lionpride.services.types import iModel

from .generate import generate
from .types import GenerateParams, InterpretParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("interpret",)


async def interpret(
    session: Session,
    branch: Branch,
    params: InterpretParams,
) -> str:
    """Interpret and refine user input into clearer prompts.

    Args:
        session: Session for services
        branch: Branch for resource access
        params: Interpret parameters

    Returns:
        Refined instruction string

    Raises:
        ValueError: If required params missing
        PermissionError: If branch doesn't have access to imodel
    """
    if params._is_sentinel(params.text):
        raise ValueError("interpret requires 'text' parameter")

    if params._is_sentinel(params.imodel):
        raise ValueError("interpret requires 'imodel' parameter")

    # Resource access check
    model_name = params.imodel.name if isinstance(params.imodel, iModel) else params.imodel
    if model_name not in branch.resources:
        raise PermissionError(
            f"Branch '{branch.name}' cannot access model '{model_name}'. "
            f"Allowed: {branch.resources or 'none'}"
        )

    # Build interpretation prompt
    system_instruction = (
        "You are given a user's raw instruction or question. Your task is to rewrite it into a clearer, "
        "more structured prompt for an LLM or system, making any implicit or missing details explicit. "
        "Return only the re-written prompt. Do not assume any details not mentioned in the input, nor "
        "give additional instruction than what is explicitly stated."
    )

    guidance = f"Domain hint: {params.domain}. Desired style: {params.style}."
    if params.sample_writing:
        guidance += f" Sample writing style: {params.sample_writing}"

    user_content = f"{guidance}\n\nUser input to refine:\n{params.text}"

    # Build combined instruction
    instruction = f"{system_instruction}\n\n{user_content}"

    # Use generate (stateless)
    gen_params = GenerateParams(
        imodel=model_name,
        instruction=instruction,
        return_as="text",
        imodel_kwargs={"temperature": params.temperature},
    )

    result = await generate(session, branch, gen_params)
    return str(result).strip()
