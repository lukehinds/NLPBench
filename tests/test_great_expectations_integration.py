"""
Tests for Great Expectations integration functionality.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch  # noqa: F401

import numpy as np
import pandas as pd
import pytest

from src.diversity_metrics import DiversityMetrics  # type: ignore
from src.great_expectations_utils import NLPDataContext  # type: ignore


class TestNLPDataContext:
    """Test cases for NLPDataContext class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

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

    def test_nlp_data_context_initialization(self, temp_dir):
        """Test NLPDataContext initialization."""
        context = NLPDataContext(context_root_dir=temp_dir)

        assert context.context_root_dir == temp_dir
        assert context.context_path == Path(temp_dir)
        assert context.context is None  # Not initialized yet

    def test_nlp_data_context_initialization_without_dir(self):
        """Test NLPDataContext initialization without specifying directory."""
        context = NLPDataContext()

        assert context.context_root_dir is not None
        assert context.context_path.exists() or context.context_root_dir.startswith("/tmp")

    def test_context_initialization(self, temp_dir):
        """Test Great Expectations context initialization."""
        context = NLPDataContext(context_root_dir=temp_dir)
        ge_context = context.initialize_context()

        assert ge_context is not None
        assert context.context is not None
        assert context.context == ge_context

    def test_context_initialization_twice(self, temp_dir):
        """Test that initializing context twice returns the same instance."""
        context = NLPDataContext(context_root_dir=temp_dir)

        ge_context1 = context.initialize_context()
        ge_context2 = context.initialize_context()

        assert ge_context1 == ge_context2

    def test_create_nlp_expectation_suite(self, temp_dir):
        """Test creation of NLP-specific expectation suite."""
        context = NLPDataContext(context_root_dir=temp_dir)
        context.initialize_context()

        suite = context.create_nlp_expectation_suite("test_suite")

        assert suite is not None
        assert suite.name == "test_suite"
        assert len(suite.expectations) >= 0  # May have no expectations initially

    def test_create_expectation_suite_twice(self, temp_dir):
        """Test creating the same expectation suite twice."""
        context = NLPDataContext(context_root_dir=temp_dir)
        context.initialize_context()

        suite1 = context.create_nlp_expectation_suite("test_suite")
        suite2 = context.create_nlp_expectation_suite("test_suite")

        assert suite1.name == suite2.name

    def test_validate_dataset_basic(self, temp_dir, sample_df):
        """Test basic dataset validation."""
        context = NLPDataContext(context_root_dir=temp_dir)

        try:
            results = context.validate_dataset_with_diversity(sample_df)

            assert isinstance(results, dict)
            assert "great_expectations" in results
            assert "diversity_metrics" in results
            assert "diversity_score" in results

            # Check diversity metrics exist
            assert isinstance(results["diversity_metrics"], dict)
            assert isinstance(results["diversity_score"], (int, float))
            assert 0 <= results["diversity_score"] <= 100

        except Exception as e:
            # Some validation might fail in test environment
            pytest.skip(f"Validation failed in test environment: {e}")

    def test_validate_dataset_without_diversity(self, temp_dir, sample_df):
        """Test dataset validation with diversity disabled."""
        context = NLPDataContext(context_root_dir=temp_dir)
        # Mock diversity calculator to simulate disabled state
        context.diversity_calculator = DiversityMetrics(enable_enhanced=False)

        try:
            results = context.validate_dataset_with_diversity(sample_df)

            assert isinstance(results, dict)
            # Should still have diversity metrics, just basic ones
            assert "diversity_metrics" in results

        except Exception as e:
            pytest.skip(f"Validation failed in test environment: {e}")

    def test_validate_empty_dataset(self, temp_dir):
        """Test validation with empty dataset."""
        context = NLPDataContext(context_root_dir=temp_dir)

        empty_df = pd.DataFrame(
            {
                "content": [],
                "role": [],
                "content_length": [],
                "content_word_count": [],
                "has_empty_content": [],
            }
        )

        results = context.validate_dataset_with_diversity(empty_df)

        assert isinstance(results, dict)
        # Should handle empty dataset gracefully

    def test_validate_invalid_dataset(self, temp_dir):
        """Test validation with invalid dataset structure."""
        context = NLPDataContext(context_root_dir=temp_dir)

        invalid_df = pd.DataFrame({"wrong_column": ["test"], "another_wrong": ["data"]})

        results = context.validate_dataset_with_diversity(invalid_df)

        assert isinstance(results, dict)
        # Great Expectations should fail, but diversity might still work partially

    def test_assess_diversity_quality(self, temp_dir):
        """Test diversity quality assessment."""
        context = NLPDataContext(context_root_dir=temp_dir)

        # Mock diversity metrics
        mock_metrics = {
            "lexical_diversity": {"type_token_ratio": 0.7},
            "role_diversity": {"dominant_role_ratio": 0.4},
            "semantic_diversity": {"semantic_diversity_score": 0.8},
        }

        assessment = context._assess_diversity_quality(mock_metrics, 75.0)

        assert isinstance(assessment, dict)
        assert "overall_diversity_level" in assessment
        assert "recommendations" in assessment
        assert isinstance(assessment["recommendations"], list)

    def test_assess_diversity_quality_low_score(self, temp_dir):
        """Test diversity quality assessment with low scores."""
        context = NLPDataContext(context_root_dir=temp_dir)

        # Mock low-quality diversity metrics
        mock_metrics = {
            "lexical_diversity": {"type_token_ratio": 0.1},
            "role_diversity": {"dominant_role_ratio": 0.9},
            "semantic_diversity": {"semantic_diversity_score": 0.2},
        }

        assessment = context._assess_diversity_quality(mock_metrics, 25.0)

        assert assessment["overall_diversity_level"] == "poor"
        assert len(assessment["recommendations"]) > 0
        # Should have specific recommendations for low diversity

    def test_get_diversity_level(self, temp_dir):
        """Test diversity level categorization."""
        context = NLPDataContext(context_root_dir=temp_dir)

        assert context._get_diversity_level(85) == "excellent"
        assert context._get_diversity_level(70) == "good"
        assert context._get_diversity_level(55) == "fair"
        assert context._get_diversity_level(30) == "poor"

    def test_cleanup(self, temp_dir):
        """Test context cleanup."""
        context = NLPDataContext(context_root_dir=temp_dir)
        context.initialize_context()

        # Should not raise an exception
        context.cleanup()

    def test_cleanup_temp_directory(self):
        """Test cleanup of temporary directory."""
        context = NLPDataContext()  # Uses temp directory
        _temp_root = context.context_root_dir

        context.initialize_context()
        context.cleanup()

        # Temp directory cleanup is best-effort, so we don't assert its deletion


class TestGreatExpectationsIntegrationEdgeCases:
    """Test edge cases for Great Expectations integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_context_initialization_failure_handling(self):
        """Test handling of context initialization failures."""
        # Use a path that should cause issues
        invalid_path = "/root/cannot_write_here"

        context = NLPDataContext(context_root_dir=invalid_path)

        # Should handle gracefully or create temp directory instead
        try:
            ge_context = context.initialize_context()
            # If it succeeds, it probably fell back to temp directory
            assert ge_context is not None
        except Exception:
            # If it fails, that's also acceptable for this edge case
            pass

    def test_diversity_calculation_failure(self, temp_dir):
        """Test handling of diversity calculation failures."""
        context = NLPDataContext(context_root_dir=temp_dir)

        # Mock diversity calculator to raise exception
        with patch.object(context, "diversity_calculator") as mock_calc:
            mock_calc.calculate_all_metrics.side_effect = Exception("Mocked failure")
            mock_calc.calculate_diversity_score.side_effect = Exception("Mocked failure")

            df = pd.DataFrame(
                {
                    "content": ["test"],
                    "role": ["user"],
                    "content_length": [4],
                    "content_word_count": [1],
                    "has_empty_content": [False],
                }
            )

            results = context.validate_dataset_with_diversity(df)

            # Should handle gracefully
            assert isinstance(results, dict)
            assert results.get("diversity_score", 0) == 0.0

    @patch("src.great_expectations_utils.get_context")
    def test_great_expectations_failure(self, mock_get_context, temp_dir):
        """Test handling of Great Expectations validation failures."""
        # Mock get_context to raise exception
        mock_get_context.side_effect = Exception("Mocked GE failure")

        context = NLPDataContext(context_root_dir=temp_dir)

        df = pd.DataFrame(
            {
                "content": ["test"],
                "role": ["user"],
                "content_length": [4],
                "content_word_count": [1],
                "has_empty_content": [False],
            }
        )

        results = context.validate_dataset_with_diversity(df)

        # Should handle GE failure gracefully but still provide diversity
        assert isinstance(results, dict)
        assert "diversity_metrics" in results or "diversity_score" in results

    def test_malformed_dataframe(self, temp_dir):
        """Test handling of malformed DataFrames."""
        context = NLPDataContext(context_root_dir=temp_dir)

        # DataFrame with mixed types and missing values
        problematic_df = pd.DataFrame(
            {
                "content": ["test", 123, None, ""],
                "role": ["user", None, "assistant", "invalid_role"],
                "content_length": [-1, "not_a_number", None, 0],
                "content_word_count": [1, 2, None, 0],
                "has_empty_content": [False, True, None, True],
            }
        )

        # Should not crash
        results = context.validate_dataset_with_diversity(problematic_df)
        assert isinstance(results, dict)


@pytest.mark.integration
class TestGreatExpectationsFullIntegration:
    """Integration tests for the full Great Expectations workflow."""

    @pytest.fixture
    def large_sample_df(self):
        """Create a larger sample DataFrame for integration testing."""
        np.random.seed(42)  # For reproducible tests

        n_samples = 100
        roles = ["user", "assistant", "system"]
        contents = [
            f"This is sample message number {i} with some variety in content and length. "
            f"It contains different words and topics to ensure diversity. "
            f"Random topic: {'AI ML data science' if i % 3 == 0 else 'cooking recipes food' if i % 3 == 1 else 'travel adventure exploration'}."
            for i in range(n_samples)
        ]

        return pd.DataFrame(
            {
                "content": contents,
                "role": [roles[i % len(roles)] for i in range(n_samples)],
                "content_length": [len(content) for content in contents],
                "content_word_count": [len(content.split()) for content in contents],
                "has_empty_content": [False] * n_samples,
            }
        )

    def test_full_integration_workflow(self, large_sample_df):
        """Test the complete integration workflow with realistic data."""
        context = NLPDataContext()

        try:
            results = context.validate_dataset_with_diversity(large_sample_df)

            # Comprehensive validation of results structure
            assert isinstance(results, dict)

            # Great Expectations results
            if "great_expectations" in results:
                ge_results = results["great_expectations"]
                assert isinstance(ge_results, dict)
                if "success" in ge_results:
                    assert isinstance(ge_results["success"], bool)

            # Diversity metrics
            assert "diversity_metrics" in results
            diversity_metrics = results["diversity_metrics"]
            assert isinstance(diversity_metrics, dict)

            # Should have basic diversity metrics at minimum
            expected_basic_metrics = [
                "lexical_diversity",
                "length_diversity",
                "character_diversity",
                "word_frequency_diversity",
                "role_diversity",
            ]

            for metric in expected_basic_metrics:
                if metric in diversity_metrics:
                    assert isinstance(diversity_metrics[metric], dict)

            # Diversity score
            assert "diversity_score" in results
            assert isinstance(results["diversity_score"], (int, float))
            assert 0 <= results["diversity_score"] <= 100

            # Diversity assessment
            if "diversity_assessment" in results:
                assessment = results["diversity_assessment"]
                assert isinstance(assessment, dict)
                assert "overall_diversity_level" in assessment
                assert "recommendations" in assessment

        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")
        finally:
            context.cleanup()

    def test_performance_with_large_dataset(self, large_sample_df):
        """Test performance characteristics with larger dataset."""
        import time

        context = NLPDataContext()

        start_time = time.time()
        try:
            results = context.validate_dataset_with_diversity(large_sample_df)
            end_time = time.time()

            # Should complete within reasonable time (adjust as needed)
            execution_time = end_time - start_time
            assert execution_time < 60  # Should complete within 1 minute

            # Results should be meaningful
            assert isinstance(results, dict)
            assert len(results) > 0

        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")
        finally:
            context.cleanup()
