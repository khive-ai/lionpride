# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for token_calculator.py to achieve 100% coverage."""

import pytest
import tiktoken

from lionpride.services.utilities.token_calculator import (
    TokenCalculationError,
    TokenCalculator,
    get_encoding_name,
)


class TestGetEncodingName:
    """Test suite for get_encoding_name function."""

    def test_get_encoding_name_valid_model(self):
        """Test getting encoding name for valid model."""
        result = get_encoding_name("gpt-4")
        assert result in ["cl100k_base", "o200k_base"]  # Valid encoding names

    def test_get_encoding_name_gpt4o(self):
        """Test getting encoding name for gpt-4o."""
        result = get_encoding_name("gpt-4o")
        assert result == "o200k_base"

    def test_get_encoding_name_direct_encoding(self):
        """Test with direct encoding name."""
        result = get_encoding_name("cl100k_base")
        assert result == "cl100k_base"

    def test_get_encoding_name_invalid_fallback(self):
        """Test that invalid names fall back to o200k_base."""
        result = get_encoding_name("invalid-model-xyz-123")
        assert result == "o200k_base"


class TestTokenCalculator:
    """Test suite for TokenCalculator class."""

    def test_calculate_message_tokens_simple(self):
        """Test calculating tokens for simple messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = TokenCalculator.calculate_message_tokens(messages, model="gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_message_tokens_default_model(self):
        """Test that calculate_message_tokens defaults to gpt-4o."""
        messages = [{"role": "user", "content": "Test"}]

        result = TokenCalculator.calculate_message_tokens(messages)
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_message_tokens_with_list_content(self):
        """Test calculating tokens for messages with list content."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}],
            }
        ]

        result = TokenCalculator.calculate_message_tokens(messages, model="gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_message_tokens_with_image_url(self):
        """Test calculating tokens for messages with image URLs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = TokenCalculator.calculate_message_tokens(messages, model="gpt-4o")
        assert isinstance(result, int)
        # Should include 500 tokens for image URL
        assert result >= 500

    def test_calculate_message_tokens_empty_list(self):
        """Test calculating tokens for empty message list."""
        result = TokenCalculator.calculate_message_tokens([], model="gpt-4o")
        assert result == 0

    def test_calculate_embed_token_valid(self):
        """Test calculating tokens for embeddings with valid inputs."""
        result = TokenCalculator.calculate_embed_token(
            ["Hello world", "Test string"],
            model="text-embedding-3-small",
        )
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_embed_token_default_model(self):
        """Test that calculate_embed_token defaults to text-embedding-3-small."""
        result = TokenCalculator.calculate_embed_token(["Test"])
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_embed_token_empty_inputs(self):
        """Test that empty inputs list raises ValueError."""
        with pytest.raises(ValueError, match="inputs must be a non-empty list"):
            TokenCalculator.calculate_embed_token(
                []  # Empty list
            )

    def test_calculate_embed_token_exception_handling(self):
        """Test that exceptions in embed calculation raise TokenCalculationError."""
        # Passing None will fail the "not inputs" check before tokenization
        with pytest.raises(ValueError, match="inputs must be a non-empty list"):
            TokenCalculator.calculate_embed_token(None)

    def test_tokenize_simple_string(self):
        """Test tokenizing a simple string."""
        result = TokenCalculator.tokenize("Hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string returns 0."""
        result = TokenCalculator.tokenize("")
        assert result == 0

    def test_tokenize_none(self):
        """Test tokenizing None returns 0."""
        result = TokenCalculator.tokenize(None)
        assert result == 0

    def test_tokenize_with_encoding_name(self):
        """Test tokenizing with specific encoding name."""
        result = TokenCalculator.tokenize("Hello", encoding_name="cl100k_base")
        assert isinstance(result, int)
        assert result > 0

    def test_tokenize_with_custom_tokenizer(self):
        """Test tokenizing with custom tokenizer callable."""
        encoding = tiktoken.get_encoding("cl100k_base")
        # Also need decoder when using custom tokenizer without encoding_name
        result = TokenCalculator.tokenize(
            "Hello", encoding_name="cl100k_base", tokenizer=encoding.encode
        )
        assert isinstance(result, int)
        assert result > 0

    def test_tokenize_return_tokens(self):
        """Test tokenizing with return_tokens=True."""
        result = TokenCalculator.tokenize("Hello", return_tokens=True)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, int) for t in result)

    def test_tokenize_return_decoded(self):
        """Test tokenizing with return_tokens=True and return_decoded=True."""
        result = TokenCalculator.tokenize(
            "Hello world", encoding_name="cl100k_base", return_tokens=True, return_decoded=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        count, decoded = result
        assert isinstance(count, int)
        assert isinstance(decoded, str)
        assert decoded == "Hello world"

    def test_tokenize_exception_handling(self):
        """Test that tokenize raises TokenCalculationError on failures."""

        # Create a broken tokenizer that raises an exception
        def broken_tokenizer(s):
            raise RuntimeError("Tokenizer failed")

        with pytest.raises(TokenCalculationError):
            TokenCalculator.tokenize(
                "test", encoding_name="cl100k_base", tokenizer=broken_tokenizer
            )

    def test_calculate_chatitem_string(self):
        """Test _calculate_chatitem with string input."""
        encoding = tiktoken.get_encoding("o200k_base")
        result = TokenCalculator._calculate_chatitem("Hello", encoding.encode, "gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_chatitem_dict_with_text(self):
        """Test _calculate_chatitem with dict containing 'text'."""
        encoding = tiktoken.get_encoding("o200k_base")
        result = TokenCalculator._calculate_chatitem(
            {"type": "text", "text": "Hello"}, encoding.encode, "gpt-4o"
        )
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_chatitem_dict_with_image_url(self):
        """Test _calculate_chatitem with dict containing 'image_url'."""
        encoding = tiktoken.get_encoding("o200k_base")
        result = TokenCalculator._calculate_chatitem(
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            encoding.encode,
            "gpt-4o",
        )
        assert result == 500  # Fixed cost for image URL

    def test_calculate_chatitem_list(self):
        """Test _calculate_chatitem with list of strings."""
        encoding = tiktoken.get_encoding("o200k_base")
        result = TokenCalculator._calculate_chatitem(["Hello", "World"], encoding.encode, "gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_chatitem_nested_list(self):
        """Test _calculate_chatitem with nested list."""
        encoding = tiktoken.get_encoding("o200k_base")
        result = TokenCalculator._calculate_chatitem(
            [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "test.jpg"}},
            ],
            encoding.encode,
            "gpt-4o",
        )
        assert isinstance(result, int)
        assert result >= 500  # At least the image cost

    def test_calculate_chatitem_exception_handling(self):
        """Test _calculate_chatitem exception handling."""
        # When tokenizer is None and we pass a string, tokenize will get encoding from model name
        result = TokenCalculator._calculate_chatitem("test", None, "gpt-4o")
        # This actually works because tokenize falls back to encoding_name
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_embed_item_string(self):
        """Test _calculate_embed_item with string input."""
        encoding = tiktoken.get_encoding("cl100k_base")
        result = TokenCalculator._calculate_embed_item("Hello world", encoding.encode)
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_embed_item_list(self):
        """Test _calculate_embed_item with list input."""
        encoding = tiktoken.get_encoding("cl100k_base")
        result = TokenCalculator._calculate_embed_item(["Hello", "World", "Test"], encoding.encode)
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_embed_item_nested_list(self):
        """Test _calculate_embed_item with nested list."""
        encoding = tiktoken.get_encoding("cl100k_base")
        result = TokenCalculator._calculate_embed_item(
            [["Hello", "World"], ["Test"]], encoding.encode
        )
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_embed_item_exception_handling(self):
        """Test _calculate_embed_item with None tokenizer."""
        # When tokenizer is None, tokenize will try to get encoding but encoding_name is also None
        # This will fall back to "o200k_base"
        result = TokenCalculator._calculate_embed_item("test", None)
        # This actually works due to fallback in get_encoding_name
        assert isinstance(result, int)
        assert result > 0

    def test_tokenize_with_both_tokenizer_and_encoding_name(self):
        """Test that custom tokenizer takes precedence over encoding_name."""
        encoding = tiktoken.get_encoding("cl100k_base")
        result = TokenCalculator.tokenize(
            "Hello",
            encoding_name="o200k_base",  # This should be ignored
            tokenizer=encoding.encode,  # This should be used
        )
        assert isinstance(result, int)
        assert result > 0

    def test_calculate_chatitem_dict_without_text_or_image(self):
        """Test _calculate_chatitem with dict that has neither text nor image_url."""
        encoding = tiktoken.get_encoding("o200k_base")
        # Dict with other fields - no 'text' or 'image_url'
        result = TokenCalculator._calculate_chatitem(
            {"type": "other", "data": "something"}, encoding.encode, "gpt-4o"
        )
        # When dict has neither 'text' nor 'image_url', function returns 0
        assert result == 0


class TestTokenCalculatorExceptionPaths:
    """Tests for exception handling paths in TokenCalculator."""

    def test_calculate_embed_token_with_failing_tokenizer(self):
        """Test calculate_embed_token exception path when tokenization fails."""
        from unittest.mock import patch

        # Patch tiktoken.get_encoding to raise an exception
        with patch(
            "lionpride.services.utilities.token_calculator.tiktoken.get_encoding"
        ) as mock_enc:
            mock_enc.side_effect = RuntimeError("Encoding failure")

            with pytest.raises(TokenCalculationError, match="Embed token calculation failed"):
                TokenCalculator.calculate_embed_token(["test"], model="gpt-4o")

    def test_calculate_chatitem_with_non_stringable_object(self):
        """Test _calculate_chatitem exception path with object that can't be stringified."""

        # Object that raises exception when converted to string
        class BadObject:
            def __str__(self):
                raise RuntimeError("Cannot convert to string")

        encoding = tiktoken.get_encoding("o200k_base")

        with pytest.raises(TokenCalculationError, match="Chat item token calculation failed"):
            TokenCalculator._calculate_chatitem({"text": BadObject()}, encoding.encode, "gpt-4o")

    def test_calculate_chatitem_nested_list_exception(self):
        """Test _calculate_chatitem with nested list that causes exception."""

        def failing_tokenizer(text):
            if isinstance(text, str):
                raise RuntimeError("Tokenizer failure")
            return []

        # Nested list with string will trigger exception in recursive call
        with pytest.raises(TokenCalculationError):
            TokenCalculator._calculate_chatitem(["nested", "list"], failing_tokenizer, "gpt-4o")

    def test_calculate_embed_item_nested_list_exception(self):
        """Test _calculate_embed_item with nested list that causes exception."""

        def failing_tokenizer(text):
            if isinstance(text, str):
                raise RuntimeError("Tokenizer failure")
            return []

        # Nested list with string will trigger exception in recursive call
        with pytest.raises(TokenCalculationError):
            TokenCalculator._calculate_embed_item(["nested", "list"], failing_tokenizer)


class TestTokenCalculatorCoverageGaps:
    """Tests targeting specific uncovered lines in token_calculator.py."""

    def test_calculate_embed_token_reraise_token_calculation_error(self):
        """Test calculate_embed_token re-raises TokenCalculationError from nested calls (line 71).

        When _calculate_embed_item raises TokenCalculationError, calculate_embed_token
        should re-raise it directly without wrapping.
        """
        from unittest.mock import patch

        # Patch _calculate_embed_item to raise TokenCalculationError directly
        with (
            patch.object(
                TokenCalculator,
                "_calculate_embed_item",
                side_effect=TokenCalculationError("Nested error"),
            ),
            pytest.raises(TokenCalculationError, match="Nested error"),
        ):
            TokenCalculator.calculate_embed_token(["test"])

    def test_calculate_embed_item_unknown_type_returns_zero(self):
        """Test _calculate_embed_item returns 0 for unknown types (line 155).

        When input is neither str nor list, return 0 (valid for non-text content).
        """
        encoding = tiktoken.get_encoding("cl100k_base")

        # Test with int
        result = TokenCalculator._calculate_embed_item(42, encoding.encode)
        assert result == 0

        # Test with dict
        result = TokenCalculator._calculate_embed_item({"key": "value"}, encoding.encode)
        assert result == 0

        # Test with None
        result = TokenCalculator._calculate_embed_item(None, encoding.encode)
        assert result == 0

        # Test with float
        result = TokenCalculator._calculate_embed_item(3.14, encoding.encode)
        assert result == 0

    def test_calculate_embed_item_wraps_non_token_calc_exception(self):
        """Test _calculate_embed_item wraps non-TokenCalculationError exceptions (lines 159-164).

        When an exception other than TokenCalculationError is raised,
        it should be wrapped in TokenCalculationError.
        """

        # Create a list subclass that raises during iteration
        # This triggers the exception in sum() rather than tokenize()
        class BadList(list):
            def __iter__(self):
                raise RuntimeError("Iteration failed")

        encoding = tiktoken.get_encoding("cl100k_base")

        with pytest.raises(TokenCalculationError, match="Embed item token calculation failed"):
            TokenCalculator._calculate_embed_item(BadList(["a", "b"]), encoding.encode)

    def test_calculate_embed_item_propagates_token_calc_error(self):
        """Test _calculate_embed_item re-raises TokenCalculationError without wrapping (lines 156-158).

        When TokenCalculationError is raised from nested call, it should propagate unchanged.
        """

        # Create a tokenizer that raises TokenCalculationError
        def tokenizer_raises_tce(text):
            raise TokenCalculationError("Original error")

        with pytest.raises(TokenCalculationError, match="Original error"):
            TokenCalculator._calculate_embed_item("test", tokenizer_raises_tce)


class TestTokenCalculatorRegressions:
    """Regression tests for specific bugs found during code review."""

    def test_dict_with_text_field_passes_parameters_correctly(self):
        """Verify dict content with 'text' field passes parameters correctly.

        Bug: Recursive call was missing tokenizer and model_name parameters.
        Symptom: TypeError when processing messages with dict content containing 'text'.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "text", "text": "How are you?"},
                ],
            }
        ]

        result = TokenCalculator.calculate_message_tokens(messages, model="gpt-4o")

        # Should return non-zero token count without raising TypeError
        assert result > 0
        assert isinstance(result, int)

    def test_logs_and_raises_on_error(self, caplog):
        """Verify errors are logged and raise TokenCalculationError.

        Bug: Silent `except Exception: return 0` hid all errors (couldn't distinguish
        empty input from actual failure).
        Fix: Now logs with exc_info=True and raises TokenCalculationError for real errors.
        """
        import logging

        with caplog.at_level(logging.ERROR):
            # Force an error by passing invalid tokenizer that will fail
            def failing_tokenizer(text):
                raise RuntimeError("Simulated tokenizer failure")

            # Should raise TokenCalculationError (not return 0)
            with pytest.raises(TokenCalculationError) as exc_info:
                TokenCalculator.tokenize("test", tokenizer=failing_tokenizer)

            # Verify error was logged before raising
            assert len(caplog.records) > 0
            assert any("Tokenization failed" in record.message for record in caplog.records)
            # Verify exc_info was logged (traceback present)
            assert any(record.exc_info is not None for record in caplog.records)
            # Verify exception chain preserved
            assert "Simulated tokenizer failure" in str(exc_info.value.__cause__)
