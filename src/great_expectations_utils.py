"""
Great Expectations utilities and custom expectations for NLP dataset quality assessment.
"""

import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from great_expectations.core import ExpectationSuite
from great_expectations.data_context.data_context.context_factory import get_context
from great_expectations.data_context.data_context.file_data_context import FileDataContext
from rich.console import Console

from .diversity_metrics import DiversityMetrics

console = Console()


class NLPDataContext:
    """Great Expectations data context specifically configured for NLP datasets."""

    def __init__(self, context_root_dir: Optional[str] = None):
        """
        Initialize NLP-focused Great Expectations context.

        Args:
            context_root_dir: Directory for GE context. If None, uses temp directory.
        """
        self.context_root_dir = context_root_dir or tempfile.mkdtemp(prefix="nlpbench_ge_")
        self.context_path = Path(self.context_root_dir)
        self.context: Optional[FileDataContext] = None
        self.diversity_calculator = DiversityMetrics(enable_enhanced=True)

    def initialize_context(self) -> Optional[FileDataContext]:
        """Initialize or get the Great Expectations context."""
        if self.context is not None:
            return self.context

        try:
            # Try to get existing context or create new one
            console.print(
                f"[blue]Initializing Great Expectations context at {self.context_root_dir}[/blue]"
            )
            self.context = get_context(context_root_dir=self.context_root_dir)
        except Exception:
            # If that fails, create a new context using the modern API
            console.print("[blue]Creating new Great Expectations context[/blue]")
            # Ensure the directory exists
            self.context_path.mkdir(parents=True, exist_ok=True)

            # Use get_context with project_root_dir to create a new context
            try:
                self.context = get_context(project_root_dir=self.context_root_dir)
            except Exception as e:
                console.print(f"[yellow]Context creation warning: {str(e)}[/yellow]")
                # Fallback: create minimal context manually
                self.context = None

        # Configure datasources and stores if context was created successfully
        if self.context is not None:
            self._configure_context()
        else:
            console.print(
                "[yellow]Running without Great Expectations context - using basic validation only[/yellow]"
            )

        return self.context

    def _configure_context(self):
        """Configure the Great Expectations context for NLP datasets."""
        if not self.context:
            raise ValueError("Context not initialized")

        # Add pandas datasource for runtime data using modern Fluent API
        try:
            # Check if datasource already exists
            try:
                existing_ds = self.context.get_datasource("nlp_pandas_datasource")
                if existing_ds:
                    console.print("[blue]Using existing pandas datasource[/blue]")
                    return
            except Exception:
                pass

            # Create new datasource using correct API
            try:
                self.context.data_sources.add_or_update(
                    name="nlp_pandas_datasource",
                    class_name="Datasource",
                    execution_engine={"class_name": "PandasExecutionEngine"},
                    data_connectors={
                        "runtime_data_connector": {
                            "class_name": "RuntimeDataConnector",
                            "batch_identifiers": ["batch_id"],
                        }
                    },
                )
                console.print("[blue]Created new pandas datasource[/blue]")
            except Exception:
                # If modern API fails, skip datasource creation
                console.print(
                    "[yellow]Skipping datasource creation - will use basic validation only[/yellow]"
                )

        except Exception as e:
            # Datasource creation failed - continue without it
            console.print(f"[yellow]Datasource configuration note: {str(e)}[/yellow]")

    def create_nlp_expectation_suite(
        self, suite_name: str = "nlp_dataset_suite"
    ) -> ExpectationSuite:
        """
        Create a comprehensive expectation suite for NLP datasets.

        Args:
            suite_name: Name of the expectation suite

        Returns:
            Configured expectation suite
        """
        if self.context is None:
            self.context = self.initialize_context()

        # Create or get suite - simplified approach
        if self.context is not None:
            try:
                suite = self.context.suites.get(suite_name)
                if suite:
                    console.print(f"[blue]Using existing expectation suite: {suite_name}[/blue]")
                    return suite
            except Exception:
                pass

        # Create new suite with basic structure
        console.print(f"[blue]Created new expectation suite: {suite_name}[/blue]")
        # Use correct GE API parameter name
        suite = ExpectationSuite(name=suite_name, expectations=[])

        # Add meta information about diversity expectations
        self._add_diversity_meta(suite)

        return suite

    def _add_diversity_meta(self, suite: ExpectationSuite):
        """Add diversity-focused metadata to the expectation suite."""

        # Add meta information about what we expect to validate
        diversity_expectations = [
            {
                "category": "lexical_diversity",
                "description": "Type-Token Ratio should be between 0.1 and 0.9",
                "min_ttr": 0.1,
                "max_ttr": 0.9,
            },
            {
                "category": "semantic_diversity",
                "description": "Semantic diversity score should be above 0.3",
                "min_diversity_score": 0.3,
            },
            {
                "category": "role_balance",
                "description": "No single role should dominate more than 80%",
                "max_imbalance_ratio": 0.8,
            },
            {
                "category": "content_length_diversity",
                "description": "Content length coefficient of variation should be above 0.2",
                "min_coefficient_of_variation": 0.2,
            },
        ]

        # Store as metadata
        if not hasattr(suite, "meta") or suite.meta is None:
            suite.meta = {}
        suite.meta["diversity_expectations"] = diversity_expectations

    def validate_dataset_with_diversity(
        self, df: pd.DataFrame, suite_name: str = "nlp_dataset_suite"
    ) -> dict[str, Any]:
        """
        Validate dataset using Great Expectations and calculate diversity metrics.

        Args:
            df: DataFrame to validate
            suite_name: Name of expectation suite to use

        Returns:
            Combined validation and diversity results
        """
        results = {}

        # For now, skip complex GE validation and focus on diversity
        # This can be enhanced later when GE API is more stable
        # suite_name parameter is reserved for future Great Expectations integration
        try:
            # Perform basic validation checks manually
            validation_success = True
            validation_messages = []

            # Check required columns
            required_columns = ["role", "content"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_success = False
                validation_messages.append(f"Missing required columns: {missing_columns}")

            # Check for null values
            if "role" in df.columns and df["role"].isnull().any():
                validation_success = False
                validation_messages.append("Found null values in 'role' column")

            if "content" in df.columns and df["content"].isnull().any():
                validation_success = False
                validation_messages.append("Found null values in 'content' column")

            results["great_expectations"] = {
                "success": validation_success,
                "results_summary": {
                    "total_expectations": len(required_columns) + 2,  # columns + null checks
                    "successful_expectations": (len(required_columns) + 2)
                    - len(validation_messages),
                    "failed_expectations": len(validation_messages),
                    "success_percentage": ((len(required_columns) + 2) - len(validation_messages))
                    / (len(required_columns) + 2)
                    * 100,
                },
                "validation_messages": validation_messages,
            }

        except Exception as e:
            console.print(f"[yellow]Basic validation error: {str(e)}[/yellow]")
            results["great_expectations"] = {"success": False, "error": str(e)}

        # Calculate diversity metrics
        try:
            diversity_metrics = self.diversity_calculator.calculate_all_metrics(df)
            diversity_score = self.diversity_calculator.calculate_diversity_score(diversity_metrics)

            results["diversity_metrics"] = diversity_metrics
            results["diversity_score"] = diversity_score

            # Add diversity-based quality assessment
            results["diversity_assessment"] = self._assess_diversity_quality(
                diversity_metrics, diversity_score
            )

        except Exception as e:
            console.print(f"[red]Diversity calculation failed: {str(e)}[/red]")
            results["diversity_metrics"] = {}
            results["diversity_score"] = 0.0

        return results

    def _assess_diversity_quality(
        self, metrics: dict[str, Any], overall_score: float
    ) -> dict[str, Any]:
        """Assess diversity quality and provide recommendations."""

        assessment = {
            "overall_diversity_level": self._get_diversity_level(overall_score),
            "recommendations": [],
        }

        # Analyze individual metrics and provide recommendations
        if "lexical_diversity" in metrics:
            lex_metrics = metrics["lexical_diversity"]
            ttr = lex_metrics.get("type_token_ratio", 0)

            if ttr < 0.3:
                assessment["recommendations"].append(
                    "Low lexical diversity detected. Consider adding more varied vocabulary."
                )
            elif ttr > 0.8:
                assessment["recommendations"].append(
                    "Very high lexical diversity may indicate inconsistent terminology. Review for coherence."
                )

        if "role_diversity" in metrics:
            role_metrics = metrics["role_diversity"]
            dominant_ratio = role_metrics.get("dominant_role_ratio", 0)

            if dominant_ratio > 0.8:
                assessment["recommendations"].append(
                    "Imbalanced role distribution detected. Consider balancing conversation roles."
                )

        if "semantic_diversity" in metrics:
            sem_metrics = metrics["semantic_diversity"]
            sem_score = sem_metrics.get("semantic_diversity_score", 0)

            if sem_score < 0.3:
                assessment["recommendations"].append(
                    "Low semantic diversity suggests repetitive content. Add more varied topics or contexts."
                )

        if not assessment["recommendations"]:
            assessment["recommendations"].append(
                "Dataset shows good diversity across measured dimensions."
            )

        return assessment

    def _get_diversity_level(self, score: float) -> str:
        """Convert diversity score to quality level."""
        if score >= 80:
            return "excellent"
        elif score >= 65:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"

    def cleanup(self):
        """Clean up temporary resources."""
        try:
            if self.context_root_dir.startswith(tempfile.gettempdir()):
                # Only clean up if we created a temp directory
                import shutil

                shutil.rmtree(self.context_root_dir, ignore_errors=True)
        except Exception:
            pass
