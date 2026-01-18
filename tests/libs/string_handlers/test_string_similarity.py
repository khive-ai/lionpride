# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionpride.libs.string_handlers._string_similarity import (
    SimilarityAlgo,
    cosine_similarity,
    hamming_similarity,
    jaro_distance,
    jaro_winkler_similarity,
    levenshtein_distance,
    levenshtein_similarity,
    sequence_matcher_similarity,
    string_similarity,
)

# ============================================================================
# Test cosine_similarity - Line 32
# ============================================================================


def test_cosine_similarity_basic():
    """Test basic cosine similarity"""
    assert cosine_similarity("hello", "hello") == 1.0
    assert 0 < cosine_similarity("hello", "help") < 1.0


def test_cosine_similarity_empty():
    """Test cosine similarity with empty strings - lines 25-26"""
    assert cosine_similarity("", "hello") == 0.0
    assert cosine_similarity("hello", "") == 0.0
    assert cosine_similarity("", "") == 0.0


def test_cosine_similarity_no_overlap():
    """Test cosine similarity with no character overlap"""
    assert cosine_similarity("abc", "def") == 0.0


# ============================================================================
# Test string_similarity function
# ============================================================================


def test_string_similarity_jaro_winkler():
    """Test string_similarity with jaro_winkler algorithm"""
    result = string_similarity("hello", ["hello", "help", "world"])
    assert "hello" in result


def test_string_similarity_threshold():
    """Test string_similarity with threshold"""
    result = string_similarity("hello", ["hello", "help", "world"], threshold=0.9)
    assert isinstance(result, list)


def test_string_similarity_most_similar():
    """Test return_most_similar option"""
    result = string_similarity("hello", ["hello", "help", "world"], return_most_similar=True)
    assert isinstance(result, str)


def test_string_similarity_invalid_threshold():
    """Test invalid threshold raises ValueError"""
    with pytest.raises(ValueError, match="threshold must be between"):
        string_similarity("hello", ["world"], threshold=1.5)


def test_string_similarity_empty_correct_words():
    """Test empty correct_words raises ValueError"""
    with pytest.raises(ValueError, match="correct_words must not be empty"):
        string_similarity("hello", [])


def test_string_similarity_invalid_algorithm():
    """Test invalid algorithm raises ValueError"""
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        string_similarity("hello", ["world"], algorithm="invalid_algo")


def test_string_similarity_custom_function():
    """Test custom similarity function"""

    def custom_sim(s1, s2):
        return 1.0 if s1 == s2 else 0.0

    result = string_similarity("hello", ["hello", "world"], algorithm=custom_sim)
    assert "hello" in result


def test_string_similarity_case_sensitive():
    """Test case sensitive matching"""
    result = string_similarity("Hello", ["hello", "Hello"], case_sensitive=True)
    assert "Hello" in result


def test_string_similarity_hamming():
    """Test hamming algorithm"""
    result = string_similarity("hello", ["hello", "hallo"], algorithm="hamming")
    assert "hello" in result


def test_string_similarity_no_matches():
    """Test when no matches found"""
    result = string_similarity("hello", ["xyz", "abc"], threshold=0.9, algorithm="levenshtein")
    assert result is None


# ============================================================================
# Test other similarity algorithms
# ============================================================================


def test_hamming_similarity():
    """Test hamming similarity"""
    assert hamming_similarity("hello", "hello") == 1.0
    assert hamming_similarity("hello", "hallo") == 0.8
    assert hamming_similarity("hello", "help") == 0.0  # Different lengths


def test_jaro_distance():
    """Test jaro distance"""
    assert jaro_distance("", "") == 1.0
    assert jaro_distance("hello", "") == 0.0
    assert jaro_distance("", "hello") == 0.0
    assert 0 < jaro_distance("hello", "hallo") < 1.0


def test_jaro_winkler_similarity():
    """Test jaro winkler similarity"""
    assert jaro_winkler_similarity("hello", "hello") == 1.0
    assert 0 < jaro_winkler_similarity("hello", "hallo") < 1.0


def test_jaro_winkler_invalid_scaling():
    """Test jaro winkler with invalid scaling"""
    with pytest.raises(ValueError, match="Scaling factor must be between"):
        jaro_winkler_similarity("hello", "world", scaling=0.3)


def test_levenshtein_distance():
    """Test levenshtein distance"""
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("hello", "") == 5
    assert levenshtein_distance("", "hello") == 5
    assert levenshtein_distance("hello", "hello") == 0
    assert levenshtein_distance("hello", "hallo") == 1


def test_levenshtein_similarity():
    """Test levenshtein similarity"""
    assert levenshtein_similarity("", "") == 1.0
    assert levenshtein_similarity("hello", "") == 0.0
    assert levenshtein_similarity("", "hello") == 0.0
    assert levenshtein_similarity("hello", "hello") == 1.0
    assert 0 < levenshtein_similarity("hello", "hallo") < 1.0


def test_sequence_matcher_similarity():
    """Test sequence matcher similarity"""
    assert sequence_matcher_similarity("hello", "hello") == 1.0
    assert 0 < sequence_matcher_similarity("hello", "hallo") < 1.0


# ============================================================================
# Test SimilarityAlgo enum
# ============================================================================


def test_similarity_algo_enum_values():
    """Test SimilarityAlgo enum has correct values"""
    assert SimilarityAlgo.JARO_WINKLER.value == "jaro_winkler"
    assert SimilarityAlgo.LEVENSHTEIN.value == "levenshtein"
    assert SimilarityAlgo.SEQUENCE_MATCHER.value == "sequence_matcher"
    assert SimilarityAlgo.HAMMING.value == "hamming"
    assert SimilarityAlgo.COSINE.value == "cosine"


def test_similarity_algo_allowed():
    """Test SimilarityAlgo.allowed() returns all values"""
    allowed = SimilarityAlgo.allowed()
    assert isinstance(allowed, tuple)
    assert len(allowed) == 5
    assert "jaro_winkler" in allowed
    assert "levenshtein" in allowed
    assert "sequence_matcher" in allowed
    assert "hamming" in allowed
    assert "cosine" in allowed


def test_string_similarity_with_enum():
    """Test string_similarity accepts SimilarityAlgo enum"""
    # Test with JARO_WINKLER enum
    result = string_similarity(
        "hello", ["hello", "help", "world"], algorithm=SimilarityAlgo.JARO_WINKLER
    )
    assert "hello" in result

    # Test with LEVENSHTEIN enum
    result = string_similarity(
        "hello", ["hello", "hallo"], algorithm=SimilarityAlgo.LEVENSHTEIN, threshold=0.8
    )
    assert isinstance(result, list)

    # Test with HAMMING enum
    result = string_similarity("hello", ["hello", "hallo"], algorithm=SimilarityAlgo.HAMMING)
    assert "hello" in result


def test_string_similarity_enum_backward_compatible():
    """Test that enum and string produce same results"""
    test_word = "color"
    correct_words = ["colour", "caller", "car"]

    result_str = string_similarity(
        test_word, correct_words, algorithm="jaro_winkler", return_most_similar=True
    )

    result_enum = string_similarity(
        test_word,
        correct_words,
        algorithm=SimilarityAlgo.JARO_WINKLER,
        return_most_similar=True,
    )

    assert result_str == result_enum


# ============================================================================
# Coverage tests for lines 320, 327
# ============================================================================


def test_string_similarity_invalid_algorithm_type():
    """Test algorithm must be string or callable - line 320"""
    with pytest.raises(ValueError, match="algorithm must be a string"):
        string_similarity("hello", ["world"], algorithm=12345)


def test_string_similarity_hamming_different_lengths_skip():
    """Test hamming skips different length strings - line 327"""
    # When using hamming, strings of different lengths should be skipped
    # "hi" (len=2) vs "hello" (len=5) - should skip, but "hello" vs "hello" should match
    result = string_similarity("hello", ["hi", "hello", "hey"], algorithm="hamming", threshold=0.5)
    # "hi" and "hey" have different lengths than "hello", so they should be skipped
    # Only "hello" should match
    assert result == ["hello"]


# ============================================================================
# Security tests - string length limits (pure Python fallback only)
# ============================================================================


class TestSecurityLimits:
    """Test security-related string length limits.

    These limits only apply when rapidfuzz is NOT available.
    When rapidfuzz is installed, it uses O([n/64]*m) bit-vector algorithms
    that are efficient enough to handle large strings safely.
    """

    def test_levenshtein_length_limit(self):
        """Test that pure Python levenshtein rejects strings exceeding MAX_STRING_LENGTH."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            RAPIDFUZZ_AVAILABLE,
            _levenshtein_distance_pure,
        )

        if RAPIDFUZZ_AVAILABLE:
            pytest.skip("Length limits only apply to pure Python fallback")

        long_string = "a" * (MAX_STRING_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            levenshtein_distance(long_string, "short")

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            levenshtein_distance("short", long_string)

    def test_levenshtein_pure_python_length_limit(self):
        """Test pure Python levenshtein function directly."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            _levenshtein_distance_pure,
        )

        long_string = "a" * (MAX_STRING_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            _levenshtein_distance_pure(long_string, "short")

    def test_levenshtein_within_limit(self):
        """Test that levenshtein_distance works for strings within limit."""
        # Use moderately sized strings
        s1 = "a" * 100
        s2 = "b" * 100

        result = levenshtein_distance(s1, s2)
        assert result == 100  # All characters different

    def test_jaro_distance_length_limit(self):
        """Test that pure Python jaro rejects strings exceeding MAX_STRING_LENGTH."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            RAPIDFUZZ_AVAILABLE,
            _jaro_distance_pure,
        )

        if RAPIDFUZZ_AVAILABLE:
            pytest.skip("Length limits only apply to pure Python fallback")

        long_string = "a" * (MAX_STRING_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            jaro_distance(long_string, "short")

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            jaro_distance("short", long_string)

    def test_jaro_pure_python_length_limit(self):
        """Test pure Python jaro function directly."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            _jaro_distance_pure,
        )

        long_string = "a" * (MAX_STRING_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            _jaro_distance_pure(long_string, "short")

    def test_jaro_winkler_length_limit(self):
        """Test that pure Python jaro_winkler rejects strings exceeding MAX_STRING_LENGTH."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            RAPIDFUZZ_AVAILABLE,
        )

        if RAPIDFUZZ_AVAILABLE:
            pytest.skip("Length limits only apply to pure Python fallback")

        long_string = "a" * (MAX_STRING_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            jaro_winkler_similarity(long_string, "short")

    def test_levenshtein_similarity_length_limit(self):
        """Test that pure Python levenshtein_similarity rejects long strings."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            RAPIDFUZZ_AVAILABLE,
        )

        if RAPIDFUZZ_AVAILABLE:
            pytest.skip("Length limits only apply to pure Python fallback")

        long_string = "a" * (MAX_STRING_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            levenshtein_similarity(long_string, "short")


# ============================================================================
# Tests for rapidfuzz integration
# ============================================================================


class TestRapidfuzzIntegration:
    """Test rapidfuzz integration when available."""

    def test_rapidfuzz_available_exported(self):
        """Test that RAPIDFUZZ_AVAILABLE flag is exported."""
        from lionpride.libs.string_handlers._string_similarity import (
            RAPIDFUZZ_AVAILABLE,
        )

        assert isinstance(RAPIDFUZZ_AVAILABLE, bool)

    def test_levenshtein_with_large_strings_when_rapidfuzz(self):
        """Test levenshtein handles large strings when rapidfuzz is available."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            RAPIDFUZZ_AVAILABLE,
        )

        if not RAPIDFUZZ_AVAILABLE:
            pytest.skip("rapidfuzz not installed")

        # This would fail with pure Python due to size limits
        long_string = "a" * (MAX_STRING_LENGTH + 100)

        # Should work with rapidfuzz (no exception)
        result = levenshtein_distance(long_string, "b")
        assert result == MAX_STRING_LENGTH + 100  # All insertions

    def test_jaro_with_large_strings_when_rapidfuzz(self):
        """Test jaro handles large strings when rapidfuzz is available."""
        from lionpride.libs.string_handlers._string_similarity import (
            MAX_STRING_LENGTH,
            RAPIDFUZZ_AVAILABLE,
        )

        if not RAPIDFUZZ_AVAILABLE:
            pytest.skip("rapidfuzz not installed")

        # This would fail with pure Python due to size limits
        long_string = "a" * (MAX_STRING_LENGTH + 100)

        # Should work with rapidfuzz (no exception)
        result = jaro_distance(long_string, long_string)
        assert result == 1.0  # Identical strings

    def test_consistency_between_rapidfuzz_and_pure_python(self):
        """Test that rapidfuzz and pure Python produce consistent results."""
        from lionpride.libs.string_handlers._string_similarity import (
            RAPIDFUZZ_AVAILABLE,
            _jaro_distance_pure,
            _levenshtein_distance_pure,
        )

        test_cases = [
            ("hello", "hello"),
            ("hello", "hallo"),
            ("", ""),
            ("test", ""),
            ("", "test"),
            ("kitten", "sitting"),
        ]

        for s1, s2 in test_cases:
            # Pure Python results
            pure_lev = _levenshtein_distance_pure(s1, s2)
            pure_jaro = _jaro_distance_pure(s1, s2)

            # Main function results (uses rapidfuzz if available)
            main_lev = levenshtein_distance(s1, s2)
            main_jaro = jaro_distance(s1, s2)

            # Results should be consistent
            assert main_lev == pure_lev, f"Levenshtein mismatch for {s1!r}, {s2!r}"
            assert abs(main_jaro - pure_jaro) < 0.001, f"Jaro mismatch for {s1!r}, {s2!r}"
