# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for session/messages/_utils.py edge cases.

Covers:
- _clean_multiline_strings with nested dicts
- _clean_multiline function
"""

import pytest

from lionpride.session.messages._utils import (
    _clean_multiline,
    _clean_multiline_strings,
    _validate_image_url,
)


class TestCleanMultilineStrings:
    """Tests for _clean_multiline_strings function."""

    def test_simple_multiline_string(self):
        """Test cleaning simple multiline string."""
        data = {"text": "line1\nline2\nline3"}
        result = _clean_multiline_strings(data)
        assert "text" in result
        assert result["text"].endswith("\n")

    def test_nested_dict_with_multiline(self):
        """Test cleaning nested dict with multiline strings (covers line 64)."""
        data = {"outer": {"inner": "line1\nline2"}}
        result = _clean_multiline_strings(data)
        assert isinstance(result["outer"], dict)
        assert result["outer"]["inner"].endswith("\n")

    def test_deeply_nested_dict(self):
        """Test deeply nested dict handling."""
        data = {"a": {"b": {"c": "multi\nline\ntext"}}}
        result = _clean_multiline_strings(data)
        assert result["a"]["b"]["c"].endswith("\n")

    def test_list_with_multiline_strings(self):
        """Test list with multiline strings (covers lines 58-62)."""
        data = {"items": ["line1\nline2", "single"]}
        result = _clean_multiline_strings(data)
        # Multiline string in list should be cleaned
        assert result["items"][0].endswith("\n")
        # Single line string should remain unchanged
        assert result["items"][1] == "single"

    def test_non_string_values_unchanged(self):
        """Test non-string values pass through unchanged."""
        data = {"number": 42, "boolean": True, "none_val": None}
        result = _clean_multiline_strings(data)
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["none_val"] is None


class TestCleanMultiline:
    """Tests for _clean_multiline function (covers lines 72-73)."""

    def test_multiline_without_trailing_newline(self):
        """Test multiline string gets trailing newline added."""
        text = "line1\nline2\nline3"
        result = _clean_multiline(text)
        assert result.endswith("\n")

    def test_multiline_with_trailing_newline(self):
        """Test multiline string with trailing newline is preserved."""
        text = "line1\nline2\n"
        result = _clean_multiline(text)
        assert result == "line1\nline2\n"

    def test_trailing_whitespace_stripped(self):
        """Test trailing whitespace on lines is stripped."""
        text = "line1   \nline2  \nline3"
        result = _clean_multiline(text)
        assert "   \n" not in result
        assert "  \n" not in result

    def test_empty_lines_preserved(self):
        """Test empty lines are preserved."""
        text = "line1\n\nline2"
        result = _clean_multiline(text)
        assert "\n\n" in result


class TestValidateImageUrl:
    """Tests for _validate_image_url function."""

    def test_valid_https_url(self):
        """Test valid HTTPS URL passes."""
        _validate_image_url("https://example.com/image.png")  # Should not raise

    def test_valid_http_url(self):
        """Test valid HTTP URL passes."""
        _validate_image_url("http://example.com/image.png")  # Should not raise

    def test_empty_string_raises(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="must be non-empty string"):
            _validate_image_url("")

    def test_none_raises(self):
        """Test None raises ValueError."""
        with pytest.raises(ValueError, match="must be non-empty string"):
            _validate_image_url(None)

    def test_file_scheme_rejected(self):
        """Test file:// URLs are rejected."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            _validate_image_url("file:///etc/passwd")

    def test_javascript_scheme_rejected(self):
        """Test javascript: URLs are rejected."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            _validate_image_url("javascript:alert('xss')")

    def test_data_scheme_rejected(self):
        """Test data: URLs are rejected."""
        with pytest.raises(ValueError, match="must use http:// or https://"):
            _validate_image_url("data:image/png;base64,...")

    def test_missing_domain_rejected(self):
        """Test URL without domain is rejected."""
        with pytest.raises(ValueError, match="missing domain"):
            _validate_image_url("http:///path/to/image.png")
