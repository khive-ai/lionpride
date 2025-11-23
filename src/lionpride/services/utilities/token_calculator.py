# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable

import tiktoken

logger = logging.getLogger(__name__)


class TokenCalculationError(Exception):
    """Raised when token calculation fails due to encoding/model errors."""

    pass


def get_encoding_name(value: str | None) -> str:
    """Get encoding name for model, with fallback chain.

    Returns:
        Encoding name (defaults to o200k_base if model/encoding not found or None)
    """
    if value is None:
        return "o200k_base"

    try:
        enc = tiktoken.encoding_for_model(value)
        return enc.name
    except KeyError:
        # Not a known model name, try as encoding name
        try:
            tiktoken.get_encoding(value)
            return value
        except Exception as e:
            logger.warning(f"Unknown model/encoding '{value}', falling back to o200k_base: {e}")
            return "o200k_base"


class TokenCalculator:
    @staticmethod
    def calculate_message_tokens(messages: list[dict], /, **kwargs) -> int:
        model = kwargs.get("model", "gpt-4o")
        image_token_cost = kwargs.get("image_token_cost", 500)
        tokenizer = tiktoken.get_encoding(get_encoding_name(model)).encode

        num_tokens = 0
        for msg in messages:
            num_tokens += 4
            _c = msg.get("content")
            num_tokens += TokenCalculator._calculate_chatitem(
                _c, tokenizer=tokenizer, model_name=model, image_token_cost=image_token_cost
            )
        return num_tokens  # buffer for chat

    @staticmethod
    def calculate_embed_token(inputs: list[str], /, **kwargs) -> int:
        if not inputs:
            raise ValueError("inputs must be a non-empty list of strings")

        try:
            tokenizer = tiktoken.get_encoding(
                get_encoding_name(kwargs.get("model", "text-embedding-3-small"))
            ).encode

            return sum(
                TokenCalculator._calculate_embed_item(i, tokenizer=tokenizer) for i in inputs
            )
        except TokenCalculationError:
            # Re-raise from nested calls
            raise
        except Exception as e:
            logger.error(f"Failed to calculate embed tokens: {e}", exc_info=True)
            raise TokenCalculationError(f"Embed token calculation failed: {e}") from e

    @staticmethod
    def tokenize(
        s_: str | None = None,
        /,
        encoding_name: str | None = None,
        tokenizer: Callable | None = None,
        decoder: Callable | None = None,
        return_tokens: bool = False,
        return_decoded: bool = False,
    ) -> int | list[int]:
        if not s_:
            return 0

        if not callable(tokenizer):
            encoding_name = get_encoding_name(encoding_name)
            tokenizer = tiktoken.get_encoding(encoding_name).encode
        if not callable(decoder):
            # Use encoding_name if available, otherwise fallback to default
            decoder_encoding = encoding_name if encoding_name else "o200k_base"
            decoder = tiktoken.get_encoding(decoder_encoding).decode

        try:
            if return_tokens:
                if return_decoded:
                    a = tokenizer(s_)
                    return len(a), decoder(a)
                return tokenizer(s_)
            return len(tokenizer(s_))
        except Exception as e:
            # Actual encoding failure during tokenization - this is an error
            logger.error(
                f"Tokenization failed for input (len={len(s_) if s_ else 0}): {e}", exc_info=True
            )
            raise TokenCalculationError(f"Tokenization failed: {e}") from e

    @staticmethod
    def _calculate_chatitem(
        i_, tokenizer: Callable, model_name: str, image_token_cost: int = 500
    ) -> int:
        try:
            if isinstance(i_, str):
                return TokenCalculator.tokenize(i_, tokenizer=tokenizer)

            if isinstance(i_, dict):
                if "text" in i_:
                    return TokenCalculator._calculate_chatitem(
                        str(i_["text"]), tokenizer, model_name, image_token_cost
                    )
                elif "image_url" in i_:
                    return image_token_cost

            if isinstance(i_, list):
                return sum(
                    TokenCalculator._calculate_chatitem(x, tokenizer, model_name, image_token_cost)
                    for x in i_
                )

            # Unknown type - return 0 is valid (no text content)
            return 0
        except TokenCalculationError:
            # Re-raise tokenization errors from nested calls
            raise
        except Exception as e:
            logger.error(
                f"Failed to calculate chat item tokens (type={type(i_).__name__}): {e}",
                exc_info=True,
            )
            raise TokenCalculationError(f"Chat item token calculation failed: {e}") from e

    @staticmethod
    def _calculate_embed_item(s_, tokenizer: Callable) -> int:
        try:
            if isinstance(s_, str):
                return TokenCalculator.tokenize(s_, tokenizer=tokenizer)

            if isinstance(s_, list):
                return sum(TokenCalculator._calculate_embed_item(x, tokenizer) for x in s_)

            # Unknown type - return 0 is valid (no text content)
            return 0
        except TokenCalculationError:
            # Re-raise tokenization errors from nested calls
            raise
        except Exception as e:
            logger.error(
                f"Failed to calculate embed item tokens (type={type(s_).__name__}): {e}",
                exc_info=True,
            )
            raise TokenCalculationError(f"Embed item token calculation failed: {e}") from e
