# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Interpret operation - refine user instructions for better LLM understanding.

Rewrites raw user input into clearer, more structured prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lionpride.session import Branch, Session


async def interpret(
    session: Session,
    branch: Branch | str,
    parameters: dict[str, Any],
) -> str:
    """Interpret and refine user input into clearer prompts.

    Args:
        parameters: Must include:
            - text: str - Raw user instruction to refine
            - imodel: Model to use for interpretation
            Optional:
            - domain: str - Domain hint (default: "general")
            - style: str - Desired style (default: "concise")
            - sample_writing: str - Example of desired output style

    Returns:
        Refined instruction string
    """
    from .generate import generate

    # Extract parameters
    text = parameters.get("text")
    if not text:
        raise ValueError("interpret requires 'text' parameter")

    imodel = parameters.get("imodel")
    if not imodel:
        raise ValueError("interpret requires 'imodel' parameter")

    domain = parameters.get("domain", "general")
    style = parameters.get("style", "concise")
    sample_writing = parameters.get("sample_writing", "")

    # Build interpretation prompt
    instruction = (
        "You are given a user's raw instruction or question. Your task is to rewrite it into a clearer, "
        "more structured prompt for an LLM or system, making any implicit or missing details explicit. "
        "Return only the re-written prompt. Do not assume any details not mentioned in the input, nor "
        "give additional instruction than what is explicitly stated."
    )

    guidance = f"Domain hint: {domain}. Desired style: {style}."
    if sample_writing:
        guidance += f" Sample writing style: {sample_writing}"

    # Build messages for generate
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {
            "role": "user",
            "content": f"{guidance}\n\nUser input to refine:\n{text}",
        },
    ]

    # Use generate (stateless) to avoid polluting conversation history
    gen_params = {
        "messages": messages,
        "imodel": imodel,
        "temperature": parameters.get("temperature", 0.1),
    }

    result = await generate(session, branch, gen_params)

    return str(result).strip()
