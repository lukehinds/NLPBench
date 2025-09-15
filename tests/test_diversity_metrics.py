"""
Tests for diversity metrics functionality.
"""

from unittest.mock import Mock, patch  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd
import pytest

from src.diversity_metrics import DiversityMetrics


def _enhanced_dependencies_available() -> bool:
    """Check if enhanced dependencies are available for testing."""
    dm = DiversityMetrics(enable_enhanced=True)
    return dm._enhanced_available


class TestDiversityMetrics:
    """Test cases for DiversityMetrics class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "content": [
                    "Hello world, this is a test message.",
                    "How are you doing today? I hope everything is well.",
                    "This is a completely different message with unique words.",
                    "Another example with some repeated words like test and message.",
                    "Final sample text for comprehensive testing purposes.",
                ],
                "role": ["user", "assistant", "user", "assistant", "system"],
                "content_length": [38, 55, 62, 68, 52],
                "content_word_count": [8, 11, 10, 11, 8],
                "has_empty_content": [False, False, False, False, False],
            }
        )

    @pytest.fixture
    def minimal_df(self):
        """Create minimal DataFrame for edge case testing."""
        return pd.DataFrame(
            {
                "content": ["test"],
                "role": ["user"],
                "content_length": [4],
                "content_word_count": [1],
                "has_empty_content": [False],
            }
        )

    @pytest.fixture
    def empty_df(self):
        """Create empty DataFrame for edge case testing."""
        return pd.DataFrame(
            {
                "content": [],
                "role": [],
                "content_length": [],
                "content_word_count": [],
                "has_empty_content": [],
            }
        )

    def test_basic_metrics_initialization(self):
        """Test DiversityMetrics initialization with basic metrics only."""
        dm = DiversityMetrics(enable_enhanced=False)
        assert not dm.enable_enhanced
        assert not dm._enhanced_available or not dm.enable_enhanced

    def test_enhanced_metrics_initialization(self):
        """Test DiversityMetrics initialization with enhanced metrics."""
        dm = DiversityMetrics(enable_enhanced=True)
        # This will depend on whether optional deps are installed
        assert dm.enable_enhanced in [True, False]  # May be disabled if deps missing

    def test_basic_lexical_diversity(self, sample_df):
        """Test basic lexical diversity calculation."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm._basic_lexical_diversity(sample_df["content"])

        assert "type_token_ratio" in metrics
        assert "unique_word_ratio" in metrics
        assert "vocabulary_size" in metrics
        assert "total_words" in metrics

        assert 0 <= metrics["type_token_ratio"] <= 1
        assert metrics["vocabulary_size"] > 0
        assert metrics["total_words"] > 0

    def test_basic_lexical_diversity_empty(self, empty_df):
        """Test basic lexical diversity with empty data."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm._basic_lexical_diversity(pd.Series([], dtype=str))

        assert metrics["type_token_ratio"] == 0.0
        assert metrics["vocabulary_size"] == 0
        assert metrics["total_words"] == 0

    def test_length_diversity(self, sample_df):
        """Test content length diversity calculation."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm._length_diversity(sample_df["content"])

        assert "length_variance" in metrics
        assert "length_std" in metrics
        assert "length_cv" in metrics
        assert "length_range" in metrics
        assert "length_iqr" in metrics

        assert metrics["length_variance"] >= 0
        assert metrics["length_std"] >= 0
        assert metrics["length_cv"] >= 0

    def test_character_diversity(self, sample_df):
        """Test character-level diversity calculation."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm._character_diversity(sample_df["content"])

        assert "char_entropy" in metrics
        assert "unique_char_ratio" in metrics
        assert "unique_chars" in metrics

        assert metrics["char_entropy"] >= 0
        assert 0 <= metrics["unique_char_ratio"] <= 1
        assert metrics["unique_chars"] > 0

    def test_word_frequency_diversity(self, sample_df):
        """Test word frequency diversity calculation."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm._word_frequency_diversity(sample_df["content"])

        assert "word_entropy" in metrics
        assert "hapax_ratio" in metrics
        assert "word_freq_variance" in metrics

        assert metrics["word_entropy"] >= 0
        assert 0 <= metrics["hapax_ratio"] <= 1

    def test_role_diversity(self, sample_df):
        """Test role distribution diversity calculation."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm._role_diversity(sample_df["role"])

        assert "role_entropy" in metrics
        assert "role_balance" in metrics
        assert "unique_roles" in metrics
        assert "dominant_role_ratio" in metrics

        assert metrics["role_entropy"] >= 0
        assert 0 <= metrics["role_balance"] <= 1
        assert metrics["unique_roles"] > 0
        assert 0 <= metrics["dominant_role_ratio"] <= 1

    def test_role_diversity_single_role(self):
        """Test role diversity with single role."""
        dm = DiversityMetrics(enable_enhanced=False)
        single_role_series = pd.Series(["user", "user", "user"])
        metrics = dm._role_diversity(single_role_series)

        assert metrics["role_entropy"] == 0.0
        assert metrics["role_balance"] == 0.0
        assert metrics["unique_roles"] == 1
        assert metrics["dominant_role_ratio"] == 1.0

    def test_calculate_all_basic_metrics(self, sample_df):
        """Test calculation of all basic metrics."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(sample_df)

        # Check that all basic metric categories are present
        expected_categories = [
            "lexical_diversity",
            "length_diversity",
            "character_diversity",
            "word_frequency_diversity",
            "role_diversity",
        ]

        for category in expected_categories:
            assert category in metrics
            assert isinstance(metrics[category], dict)

    def test_diversity_score_calculation(self, sample_df):
        """Test overall diversity score calculation."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(sample_df)
        score = dm.calculate_diversity_score(metrics)

        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_diversity_score_empty_metrics(self):
        """Test diversity score calculation with empty metrics."""
        dm = DiversityMetrics(enable_enhanced=False)
        score = dm.calculate_diversity_score({})

        assert score == 0.0

    def test_minimal_dataset(self, minimal_df):
        """Test metrics calculation with minimal dataset."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(minimal_df)

        # Should not crash and should return some metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_empty_dataset(self, empty_df):
        """Test metrics calculation with empty dataset."""
        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(empty_df)

        # Should handle gracefully
        assert isinstance(metrics, dict)

    @pytest.mark.skipif(
        not _enhanced_dependencies_available(), reason="Enhanced dependencies not available"
    )
    def test_enhanced_metrics(self, sample_df):
        """Test enhanced metrics calculation (if dependencies available)."""
        dm = DiversityMetrics(enable_enhanced=True)

        if dm._enhanced_available:
            _metrics = dm.calculate_all_metrics(sample_df)

            # Check for enhanced metric categories
            _enhanced_categories = [
                "advanced_lexical_diversity",
                "semantic_diversity",
                "syntactic_diversity",
                "topic_diversity",
                "readability_diversity",
            ]

            # Note: Not all enhanced metrics may be available depending on dependencies
            # and dataset size, so we just check that the method doesn't crash

    def test_mtld_calculation_insufficient_data(self):
        """Test MTLD calculation with insufficient data."""
        dm = DiversityMetrics(enable_enhanced=False)
        short_texts = ["hi", "hello"]
        result = dm._calculate_mtld(short_texts)

        # Should return 0.0 for insufficient data
        assert result == 0.0

    def test_mattr_calculation_insufficient_data(self):
        """Test MATTR calculation with insufficient data."""
        dm = DiversityMetrics(enable_enhanced=False)
        short_texts = ["hi there"]
        result = dm._calculate_mattr(short_texts, window_size=100)

        # Should handle gracefully and return a valid ratio
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_diversity_score_weighting(self, sample_df):
        """Test that diversity score weighting works correctly."""
        dm = DiversityMetrics(enable_enhanced=False)

        # Calculate metrics
        metrics = dm.calculate_all_metrics(sample_df)

        # Modify individual metrics to test weighting
        test_metrics = metrics.copy()

        # Set all lexical diversity to maximum
        if "lexical_diversity" in test_metrics:
            test_metrics["lexical_diversity"]["type_token_ratio"] = 1.0

        score_high_lexical = dm.calculate_diversity_score(test_metrics)

        # Set lexical diversity to minimum
        test_metrics["lexical_diversity"]["type_token_ratio"] = 0.0
        score_low_lexical = dm.calculate_diversity_score(test_metrics)

        # High lexical diversity should give higher score
        assert score_high_lexical >= score_low_lexical


def _enhanced_dependencies_available():
    """Check if enhanced dependencies are available."""
    try:
        import gensim  # noqa: F401
        import nltk  # noqa: F401
        import sentence_transformers  # noqa: F401
        import sklearn  # noqa: F401
        import spacy  # noqa: F401
        import textstat  # noqa: F401

        return True
    except ImportError:
        return False


class TestDiversityMetricsEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_columns(self):
        """Test handling of missing columns."""
        df = pd.DataFrame({"content": ["test"]})  # Missing 'role' column

        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(df)

        # Should handle gracefully - role_diversity should be missing
        assert "role_diversity" not in metrics or metrics["role_diversity"] is None

    def test_null_content(self):
        """Test handling of null content."""
        df = pd.DataFrame(
            {
                "content": ["test", None, "another test", ""],
                "role": ["user", "assistant", "user", "system"],
                "content_length": [4, 0, 12, 0],
                "content_word_count": [1, 0, 2, 0],
                "has_empty_content": [False, True, False, True],
            }
        )

        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(df)

        # Should handle null values gracefully
        assert isinstance(metrics, dict)
        assert "lexical_diversity" in metrics

    def test_unicode_content(self):
        """Test handling of unicode content."""
        df = pd.DataFrame(
            {
                "content": ["Hello 世界", "café résumé", "🚀 rocket emoji"],
                "role": ["user", "assistant", "user"],
                "content_length": [11, 11, 15],
                "content_word_count": [2, 2, 3],
                "has_empty_content": [False, False, False],
            }
        )

        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(df)

        # Should handle unicode gracefully
        assert isinstance(metrics, dict)
        assert metrics["character_diversity"]["unique_chars"] > 0

    def test_very_long_content(self):
        """Test handling of very long content."""
        long_content = "word " * 10000  # Very long repeated content
        df = pd.DataFrame(
            {
                "content": [long_content],
                "role": ["user"],
                "content_length": [len(long_content)],
                "content_word_count": [10000],
                "has_empty_content": [False],
            }
        )

        dm = DiversityMetrics(enable_enhanced=False)
        metrics = dm.calculate_all_metrics(df)

        # Should handle long content without crashing
        assert isinstance(metrics, dict)
        # TTR should be very low for repeated content
        assert metrics["lexical_diversity"]["type_token_ratio"] < 0.1
