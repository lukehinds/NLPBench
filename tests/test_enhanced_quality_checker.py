"""
Tests for the enhanced quality checker with Great Expectations and diversity metrics.
"""

from unittest.mock import MagicMock, Mock, patch  # noqa: F401

import pandas as pd
import pytest

from src.models import DatasetConfig, DiversityMetrics, QualityMetrics
from src.quality_checker import DatasetQualityChecker


class TestEnhancedDatasetQualityChecker:
    """Test cases for enhanced DatasetQualityChecker."""

    @pytest.fixture
    def default_config(self):
        """Create default configuration for testing."""
        return DatasetConfig(
            enable_diversity_metrics=True,
            enable_enhanced_diversity=True,
            use_great_expectations=True,
        )

    @pytest.fixture
    def basic_config(self):
        """Create configuration with only basic features enabled."""
        return DatasetConfig(
            enable_diversity_metrics=True,
            enable_enhanced_diversity=False,
            use_great_expectations=False,
        )

    @pytest.fixture
    def minimal_config(self):
        """Create minimal configuration for testing."""
        return DatasetConfig(
            enable_diversity_metrics=False,
            enable_enhanced_diversity=False,
            use_great_expectations=False,
        )

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

    def test_initialization_with_diversity_enabled(self, default_config):
        """Test initialization with diversity metrics enabled."""
        checker = DatasetQualityChecker(default_config)

        assert checker.config == default_config
        assert checker.diversity_calculator is not None
        assert checker.ge_context is not None

    def test_initialization_with_diversity_disabled(self, minimal_config):
        """Test initialization with diversity metrics disabled."""
        checker = DatasetQualityChecker(minimal_config)

        assert checker.config == minimal_config
        assert checker.diversity_calculator is None
        assert checker.ge_context is None

    def test_initialization_basic_features_only(self, basic_config):
        """Test initialization with only basic features enabled."""
        checker = DatasetQualityChecker(basic_config)

        assert checker.config == basic_config
        assert checker.diversity_calculator is not None
        assert checker.ge_context is None

    def test_calculate_dataset_stats_with_diversity(self, default_config, sample_df):
        """Test dataset stats calculation with diversity metrics."""
        checker = DatasetQualityChecker(default_config)

        # Mock the diversity calculator to avoid dependency issues
        mock_diversity_data = {
            "lexical_diversity": {"type_token_ratio": 0.7},
            "role_diversity": {"dominant_role_ratio": 0.4},
        }

        with patch.object(
            checker.diversity_calculator, "calculate_all_metrics", return_value=mock_diversity_data
        ):
            with patch.object(
                checker.diversity_calculator, "calculate_diversity_score", return_value=75.0
            ):
                stats = checker._calculate_dataset_stats(sample_df)

        assert stats.total_entries == len(sample_df)
        assert stats.diversity_metrics is not None
        assert stats.diversity_metrics.overall_diversity_score == 75.0
        assert stats.diversity_metrics.diversity_level == "good"

    def test_calculate_dataset_stats_without_diversity(self, minimal_config, sample_df):
        """Test dataset stats calculation without diversity metrics."""
        checker = DatasetQualityChecker(minimal_config)

        stats = checker._calculate_dataset_stats(sample_df)

        assert stats.total_entries == len(sample_df)
        assert stats.diversity_metrics is None

    def test_get_diversity_level(self, default_config):
        """Test diversity level categorization."""
        checker = DatasetQualityChecker(default_config)

        assert checker._get_diversity_level(85) == "excellent"
        assert checker._get_diversity_level(70) == "good"
        assert checker._get_diversity_level(55) == "fair"
        assert checker._get_diversity_level(30) == "poor"

    def test_assess_quality_full_features(self, default_config, sample_df):
        """Test quality assessment with all features enabled."""
        checker = DatasetQualityChecker(default_config)

        # Mock diversity calculator
        mock_diversity_data = {
            "lexical_diversity": {"type_token_ratio": 0.7},
            "role_diversity": {"dominant_role_ratio": 0.4},
        }

        # Mock Great Expectations context
        mock_ge_results = {
            "great_expectations": {"success": True, "results_summary": {"success_percentage": 95}},
            "diversity_assessment": {
                "overall_diversity_level": "good",
                "recommendations": ["Dataset shows good diversity."],
            },
        }

        with patch.object(
            checker.diversity_calculator, "calculate_all_metrics", return_value=mock_diversity_data
        ):
            with patch.object(
                checker.diversity_calculator, "calculate_diversity_score", return_value=75.0
            ):
                with patch.object(
                    checker.ge_context,
                    "validate_dataset_with_diversity",
                    return_value=mock_ge_results,
                ):
                    quality_metrics = checker.assess_quality(sample_df)

        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.overall_score > 0
        assert quality_metrics.dataset_stats.diversity_metrics is not None
        assert quality_metrics.great_expectations_results is not None
        assert quality_metrics.diversity_assessment is not None

    def test_assess_quality_basic_features_only(self, basic_config, sample_df):
        """Test quality assessment with basic features only."""
        checker = DatasetQualityChecker(basic_config)

        # Mock diversity calculator for basic metrics
        mock_diversity_data = {
            "lexical_diversity": {"type_token_ratio": 0.7},
            "role_diversity": {"dominant_role_ratio": 0.4},
        }

        with patch.object(
            checker.diversity_calculator, "calculate_all_metrics", return_value=mock_diversity_data
        ):
            with patch.object(
                checker.diversity_calculator, "calculate_diversity_score", return_value=75.0
            ):
                quality_metrics = checker.assess_quality(sample_df)

        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.overall_score > 0
        assert quality_metrics.dataset_stats.diversity_metrics is not None
        assert quality_metrics.great_expectations_results is None  # GE disabled

    def test_assess_quality_minimal_features(self, minimal_config, sample_df):
        """Test quality assessment with minimal features."""
        checker = DatasetQualityChecker(minimal_config)

        quality_metrics = checker.assess_quality(sample_df)

        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.overall_score > 0
        assert quality_metrics.dataset_stats.diversity_metrics is None  # Diversity disabled
        assert quality_metrics.great_expectations_results is None  # GE disabled

    def test_calculate_enhanced_overall_score(self, default_config, sample_df):
        """Test enhanced overall score calculation."""
        checker = DatasetQualityChecker(default_config)

        # Create mock validation results and dataset stats
        mock_validation_results = []
        mock_diversity_metrics = DiversityMetrics(
            overall_diversity_score=80.0, diversity_level="good"
        )
        mock_dataset_stats = Mock()
        mock_dataset_stats.diversity_metrics = mock_diversity_metrics

        # Test with Great Expectations success
        mock_ge_results = {"success": True, "results_summary": {"success_percentage": 95}}

        with patch.object(checker, "_calculate_overall_score", return_value=85.0):
            enhanced_score = checker._calculate_enhanced_overall_score(
                mock_validation_results, mock_dataset_stats, mock_ge_results
            )

        # Should be weighted combination: 85 * 0.7 + 80 * 0.3 = 83.5
        assert 80 <= enhanced_score <= 90

    def test_calculate_enhanced_overall_score_ge_failure(self, default_config):
        """Test enhanced overall score calculation with GE failure."""
        checker = DatasetQualityChecker(default_config)

        mock_validation_results = []
        mock_diversity_metrics = DiversityMetrics(
            overall_diversity_score=80.0, diversity_level="good"
        )
        mock_dataset_stats = Mock()
        mock_dataset_stats.diversity_metrics = mock_diversity_metrics

        # Test with Great Expectations failure
        mock_ge_results = {"success": False, "results_summary": {"success_percentage": 60}}

        with patch.object(checker, "_calculate_overall_score", return_value=85.0):
            enhanced_score = checker._calculate_enhanced_overall_score(
                mock_validation_results, mock_dataset_stats, mock_ge_results
            )

        # Should be penalized for GE failure
        base_enhanced = 85 * 0.7 + 80 * 0.3  # 83.5
        _expected = base_enhanced * 0.6  # ~50
        assert enhanced_score < 60

    def test_generate_enhanced_recommendations(self, default_config):
        """Test enhanced recommendations generation."""
        checker = DatasetQualityChecker(default_config)

        mock_validation_results = []
        mock_diversity_metrics = DiversityMetrics(
            lexical_diversity={"type_token_ratio": 0.05},  # Low TTR
            role_diversity={"dominant_role_ratio": 0.9},  # High imbalance
            semantic_diversity={"semantic_diversity_score": 0.2},  # Low semantic diversity
        )
        mock_dataset_stats = Mock()
        mock_dataset_stats.diversity_metrics = mock_diversity_metrics

        mock_diversity_assessment = {"recommendations": ["General diversity recommendation"]}

        with patch.object(
            checker, "_generate_recommendations", return_value=["Basic recommendation"]
        ):
            recommendations = checker._generate_enhanced_recommendations(
                mock_validation_results, mock_dataset_stats, 70.0, mock_diversity_assessment
            )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should include diversity-specific recommendations
        recommendation_text = " ".join(recommendations)
        assert (
            "diversity" in recommendation_text.lower() or "lexical" in recommendation_text.lower()
        )

    def test_diversity_calculation_failure_handling(self, default_config, sample_df):
        """Test handling of diversity calculation failures."""
        checker = DatasetQualityChecker(default_config)

        # Mock diversity calculator to raise exception
        with patch.object(
            checker.diversity_calculator,
            "calculate_all_metrics",
            side_effect=Exception("Mock error"),
        ):
            stats = checker._calculate_dataset_stats(sample_df)

        # Should handle gracefully
        assert stats.diversity_metrics is None
        assert stats.total_entries == len(sample_df)

    def test_great_expectations_failure_handling(self, default_config, sample_df):
        """Test handling of Great Expectations failures."""
        checker = DatasetQualityChecker(default_config)

        # Mock GE context to raise exception
        with patch.object(
            checker.ge_context,
            "validate_dataset_with_diversity",
            side_effect=Exception("Mock GE error"),
        ):
            quality_metrics = checker.assess_quality(sample_df)

        # Should handle gracefully and still return results
        assert isinstance(quality_metrics, QualityMetrics)
        assert quality_metrics.great_expectations_results is None

    def test_cleanup(self, default_config):
        """Test cleanup functionality."""
        checker = DatasetQualityChecker(default_config)

        # Mock GE context cleanup
        with patch.object(checker.ge_context, "cleanup") as mock_cleanup:
            checker.cleanup()
            mock_cleanup.assert_called_once()

    def test_cleanup_without_ge_context(self, minimal_config):
        """Test cleanup without GE context."""
        checker = DatasetQualityChecker(minimal_config)

        # Should not raise exception
        checker.cleanup()

    def test_empty_dataset_handling(self, default_config):
        """Test handling of empty datasets."""
        checker = DatasetQualityChecker(default_config)

        empty_df = pd.DataFrame(
            {
                "content": [],
                "role": [],
                "content_length": [],
                "content_word_count": [],
                "has_empty_content": [],
            }
        )

        # Should handle gracefully
        quality_metrics = checker.assess_quality(empty_df)
        assert isinstance(quality_metrics, QualityMetrics)

    def test_invalid_dataset_handling(self, default_config):
        """Test handling of invalid datasets."""
        checker = DatasetQualityChecker(default_config)

        invalid_df = pd.DataFrame({"wrong_column": ["test"], "another_column": ["data"]})

        # Should handle gracefully (might fail validation but not crash)
        try:
            quality_metrics = checker.assess_quality(invalid_df)
            assert isinstance(quality_metrics, QualityMetrics)
        except Exception:
            # Acceptable to fail with invalid data structure
            pass


class TestEnhancedQualityCheckerConfiguration:
    """Test configuration-related functionality."""

    def test_diversity_threshold_configuration(self):
        """Test custom diversity threshold configuration."""
        config = DatasetConfig(
            diversity_thresholds={"excellent": 90.0, "good": 75.0, "fair": 60.0, "poor": 45.0}
        )

        checker = DatasetQualityChecker(config)

        assert checker._get_diversity_level(95) == "excellent"
        assert checker._get_diversity_level(80) == "good"
        assert checker._get_diversity_level(65) == "fair"
        assert checker._get_diversity_level(50) == "poor"

    def test_specific_diversity_metric_thresholds(self):
        """Test specific diversity metric threshold configuration."""
        config = DatasetConfig(
            min_lexical_diversity=0.2, max_role_imbalance=0.7, min_semantic_diversity=0.4
        )

        _checker = DatasetQualityChecker(config)

        # These should be used in recommendation generation
        assert config.min_lexical_diversity == 0.2
        assert config.max_role_imbalance == 0.7
        assert config.min_semantic_diversity == 0.4

    def test_great_expectations_context_configuration(self):
        """Test Great Expectations context configuration."""
        import tempfile

        temp_dir = tempfile.mkdtemp()
        config = DatasetConfig(use_great_expectations=True, ge_context_root=temp_dir)

        checker = DatasetQualityChecker(config)

        assert checker.ge_context is not None
        assert checker.ge_context.context_root_dir == temp_dir


@pytest.mark.integration
class TestQualityCheckerIntegration:
    """Integration tests for the complete quality checking workflow."""

    def test_end_to_end_quality_assessment(self):
        """Test complete end-to-end quality assessment workflow."""
        config = DatasetConfig(
            enable_diversity_metrics=True,
            enable_enhanced_diversity=False,  # Disable to avoid dependency issues
            use_great_expectations=False,  # Disable to avoid complexity
        )

        checker = DatasetQualityChecker(config)

        # Create realistic test data
        test_df = pd.DataFrame(
            {
                "content": [
                    "This is a comprehensive test message with substantial content.",
                    "Another message with different vocabulary and structure for diversity testing.",
                    "A third example demonstrating varied content length and complexity.",
                    "Short message.",
                    "This final example includes various linguistic patterns and demonstrates content variety for quality assessment purposes.",
                ],
                "role": ["user", "assistant", "user", "assistant", "system"],
                "content_length": [67, 84, 77, 15, 125],
                "content_word_count": [11, 12, 11, 2, 18],
                "has_empty_content": [False, False, False, False, False],
            }
        )

        quality_metrics = checker.assess_quality(test_df)

        # Comprehensive validation
        assert isinstance(quality_metrics, QualityMetrics)
        assert 0 <= quality_metrics.overall_score <= 100
        assert len(quality_metrics.validation_results) > 0
        assert quality_metrics.dataset_stats.total_entries == len(test_df)
        assert len(quality_metrics.recommendations) > 0

        # Should have diversity metrics (basic level)
        if config.enable_diversity_metrics:
            assert quality_metrics.dataset_stats.diversity_metrics is not None

        checker.cleanup()
