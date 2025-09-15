from typing import Optional

import pandas as pd
from rich.console import Console

from .diversity_metrics import DiversityMetrics as DiversityCalculator
from .great_expectations_utils import NLPDataContext
from .models import DatasetConfig, DatasetStats, DiversityMetrics, QualityMetrics, ValidationResult

console = Console()


class DatasetQualityChecker:
    """Enhanced quality checker using Great Expectations and comprehensive diversity metrics."""

    def __init__(self, config: DatasetConfig):
        self.config = config

        # Initialize diversity calculator
        self.diversity_calculator: Optional[DiversityCalculator] = None
        if config.enable_diversity_metrics:
            self.diversity_calculator = DiversityCalculator(
                enable_enhanced=config.enable_enhanced_diversity
            )

        # Initialize Great Expectations context
        self.ge_context: Optional[NLPDataContext] = None
        if config.use_great_expectations:
            self.ge_context = NLPDataContext(context_root_dir=config.ge_context_root)

    def _calculate_dataset_stats(self, df: pd.DataFrame) -> DatasetStats:
        """Calculate basic statistics about the dataset including diversity metrics."""

        # Role distribution
        role_dist = df["role"].value_counts().to_dict()

        # Content length statistics
        content_lengths = df["content_length"]

        # Count duplicates based on content
        duplicate_count = df.duplicated(subset=["content"]).sum()

        # Empty content count
        empty_content_count = df["has_empty_content"].sum()

        # Calculate diversity metrics if enabled
        diversity_metrics = None
        if self.diversity_calculator:
            try:
                console.print("[blue]Calculating diversity metrics...[/blue]")
                diversity_data = self.diversity_calculator.calculate_all_metrics(df)
                diversity_score = self.diversity_calculator.calculate_diversity_score(
                    diversity_data
                )

                # Get diversity level
                diversity_level = self._get_diversity_level(diversity_score)

                diversity_metrics = DiversityMetrics(
                    lexical_diversity=diversity_data.get("lexical_diversity"),
                    length_diversity=diversity_data.get("length_diversity"),
                    character_diversity=diversity_data.get("character_diversity"),
                    word_frequency_diversity=diversity_data.get("word_frequency_diversity"),
                    role_diversity=diversity_data.get("role_diversity"),
                    advanced_lexical_diversity=diversity_data.get("advanced_lexical_diversity"),
                    semantic_diversity=diversity_data.get("semantic_diversity"),
                    syntactic_diversity=diversity_data.get("syntactic_diversity"),
                    topic_diversity=diversity_data.get("topic_diversity"),
                    readability_diversity=diversity_data.get("readability_diversity"),
                    overall_diversity_score=diversity_score,
                    diversity_level=diversity_level,
                )
            except Exception as e:
                console.print(f"[red]Diversity calculation failed: {str(e)}[/red]")

        # Handle empty dataframes
        if len(df) == 0:
            return DatasetStats(
                total_entries=0,
                unique_entries=0,
                duplicate_count=0,
                role_distribution={},
                avg_content_length=0.0,
                min_content_length=0,
                max_content_length=0,
                empty_content_count=0,
                diversity_metrics=diversity_metrics,
            )

        return DatasetStats(
            total_entries=len(df),
            unique_entries=int(df["content"].nunique()),
            duplicate_count=int(duplicate_count),
            role_distribution=role_dist,
            avg_content_length=float(content_lengths.mean()) if not content_lengths.empty else 0.0,
            min_content_length=int(content_lengths.min()) if not content_lengths.empty else 0,
            max_content_length=int(content_lengths.max()) if not content_lengths.empty else 0,
            empty_content_count=int(empty_content_count),
            diversity_metrics=diversity_metrics,
        )

    def _get_diversity_level(self, score: float) -> str:
        """Convert diversity score to quality level."""
        thresholds = self.config.diversity_thresholds
        if score >= thresholds["excellent"]:
            return "excellent"
        elif score >= thresholds["good"]:
            return "good"
        elif score >= thresholds["fair"]:
            return "fair"
        else:
            return "poor"

    def _validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that the DataFrame has the expected schema."""

        required_columns = [
            "role",
            "content",
            "content_length",
            "content_word_count",
            "has_empty_content",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return ValidationResult(
                check_name="Schema Validation",
                success=False,
                success_percentage=0.0,
                failed_count=len(missing_columns),
                total_count=len(required_columns),
                error_message=f"Missing columns: {missing_columns}",
            )

        return ValidationResult(
            check_name="Schema Validation",
            success=True,
            success_percentage=100.0,
            failed_count=0,
            total_count=len(required_columns),
        )

    def _validate_non_null_values(self, df: pd.DataFrame, column: str) -> ValidationResult:
        """Validate that a column has no null values."""

        null_count = df[column].isnull().sum()
        total_count = len(df)
        success_percentage = (
            ((total_count - null_count) / total_count * 100) if total_count > 0 else 0
        )

        return ValidationResult(
            check_name=f"Non-null Values ({column})",
            success=null_count == 0,
            success_percentage=success_percentage,
            failed_count=int(null_count),
            total_count=total_count,
        )

    def _validate_roles(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that all roles are in the allowed set."""

        invalid_roles = ~df["role"].isin(self.config.allowed_roles)
        failed_count = invalid_roles.sum()
        total_count = len(df)
        success_percentage = (
            ((total_count - failed_count) / total_count * 100) if total_count > 0 else 0
        )

        return ValidationResult(
            check_name="Valid Roles (role)",
            success=failed_count == 0,
            success_percentage=success_percentage,
            failed_count=int(failed_count),
            total_count=total_count,
        )

    def _validate_content_length(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that content lengths are within acceptable range."""

        invalid_lengths = ~df["content_length"].between(
            self.config.min_content_length, self.config.max_content_length
        )
        failed_count = invalid_lengths.sum()
        total_count = len(df)
        success_percentage = (
            ((total_count - failed_count) / total_count * 100) if total_count > 0 else 0
        )

        return ValidationResult(
            check_name="Content Length Range (content_length)",
            success=failed_count == 0,
            success_percentage=success_percentage,
            failed_count=int(failed_count),
            total_count=total_count,
        )

    def _validate_word_count(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that word counts are positive."""

        invalid_word_counts = df["content_word_count"] < 1
        failed_count = invalid_word_counts.sum()
        total_count = len(df)
        success_percentage = (
            ((total_count - failed_count) / total_count * 100) if total_count > 0 else 0
        )

        return ValidationResult(
            check_name="Content Length Range (content_word_count)",
            success=failed_count == 0,
            success_percentage=success_percentage,
            failed_count=int(failed_count),
            total_count=total_count,
        )

    def _validate_empty_content(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that there is no empty content."""

        failed_count = df["has_empty_content"].sum()
        total_count = len(df)
        success_percentage = (
            ((total_count - failed_count) / total_count * 100) if total_count > 0 else 0
        )

        return ValidationResult(
            check_name="No Empty Content (has_empty_content)",
            success=failed_count == 0,
            success_percentage=success_percentage,
            failed_count=int(failed_count),
            total_count=total_count,
        )

    def _validate_content_uniqueness(self, df: pd.DataFrame) -> ValidationResult:
        """Validate content uniqueness."""

        duplicate_count = df.duplicated(subset=["content"]).sum()
        total_count = len(df)
        success_percentage = (
            ((total_count - duplicate_count) / total_count * 100) if total_count > 0 else 0
        )

        return ValidationResult(
            check_name="Content Uniqueness (content)",
            success=duplicate_count == 0,
            success_percentage=success_percentage,
            failed_count=int(duplicate_count),
            total_count=total_count,
        )

    def _execute_validation(self, df: pd.DataFrame) -> list[ValidationResult]:
        """Execute all validation checks and return results."""

        console.print("[blue]Executing quality validations...[/blue]")

        validation_results = []

        # Schema validation
        validation_results.append(self._validate_schema(df))

        # Non-null validations
        validation_results.append(self._validate_non_null_values(df, "role"))
        validation_results.append(self._validate_non_null_values(df, "content"))

        # Role validation
        validation_results.append(self._validate_roles(df))

        # Content length validation
        validation_results.append(self._validate_content_length(df))
        validation_results.append(self._validate_word_count(df))

        # Empty content validation
        validation_results.append(self._validate_empty_content(df))

        # Uniqueness validation
        validation_results.append(self._validate_content_uniqueness(df))

        return validation_results

    def _calculate_overall_score(
        self, validation_results: list[ValidationResult], dataset_stats: DatasetStats
    ) -> float:
        """Calculate an overall quality score based on validation results and statistics."""

        if not validation_results:
            return 0.0

        # Base score from validation success rates
        validation_scores = [result.success_percentage for result in validation_results]
        base_score = sum(validation_scores) / len(validation_scores)

        # Apply penalties for specific issues
        penalty = 0.0

        # Duplicate penalty
        if dataset_stats.total_entries > 0:
            duplicate_rate = dataset_stats.duplicate_count / dataset_stats.total_entries
            if duplicate_rate > self.config.duplicate_threshold:
                penalty += (duplicate_rate - self.config.duplicate_threshold) * 100

            # Empty content penalty
            empty_rate = dataset_stats.empty_content_count / dataset_stats.total_entries
            if empty_rate > self.config.empty_content_threshold:
                penalty += (empty_rate - self.config.empty_content_threshold) * 200

        # Small dataset penalty (less reliable for quality assessment)
        if dataset_stats.total_entries < 100:
            penalty += 5.0

        final_score = max(0.0, min(100.0, base_score - penalty))
        return final_score

    def _generate_recommendations(
        self,
        validation_results: list[ValidationResult],
        dataset_stats: DatasetStats,
        overall_score: float,
    ) -> list[str]:
        """Generate quality improvement recommendations."""

        recommendations = []

        # Score-based recommendations
        if overall_score < self.config.quality_thresholds["poor"]:
            recommendations.append(
                "Dataset quality is poor. Consider comprehensive data cleaning and validation."
            )
        elif overall_score < self.config.quality_thresholds["fair"]:
            recommendations.append(
                "Dataset quality needs improvement. Focus on addressing validation failures."
            )

        # Specific issue recommendations
        failed_validations = [r for r in validation_results if not r.success]

        for validation in failed_validations:
            if "Schema" in validation.check_name:
                recommendations.append(
                    "Fix schema issues: ensure all required columns are present with correct types."
                )
            elif "Non-null" in validation.check_name:
                recommendations.append("Remove or fix entries with missing required fields.")
            elif "Valid Roles" in validation.check_name:
                recommendations.append(
                    f"Standardize role values to: {', '.join(self.config.allowed_roles)}."
                )
            elif "Content Length" in validation.check_name:
                recommendations.append(
                    "Review content length distribution and remove entries that are too short or too long."
                )
            elif "Empty Content" in validation.check_name:
                recommendations.append(
                    "Remove entries with empty content or provide meaningful content."
                )
            elif "Uniqueness" in validation.check_name:
                recommendations.append("Remove duplicate entries to improve dataset diversity.")

        # Statistical recommendations
        if dataset_stats.duplicate_count > dataset_stats.total_entries * 0.1:
            recommendations.append("High number of duplicates detected. Consider deduplication.")

        if dataset_stats.empty_content_count > 0:
            recommendations.append("Remove entries with empty content to improve data quality.")

        if dataset_stats.role_distribution:
            role_imbalance = max(dataset_stats.role_distribution.values()) / sum(
                dataset_stats.role_distribution.values()
            )
            if role_imbalance > 0.8:
                recommendations.append(
                    "Consider balancing role distribution for more diverse conversations."
                )

        if not recommendations:
            recommendations.append(
                "Dataset quality looks good! Consider periodic quality checks as data grows."
            )

        return recommendations

    def assess_quality(self, df: pd.DataFrame) -> QualityMetrics:
        """Perform comprehensive quality assessment using Great Expectations and diversity metrics."""

        try:
            console.print("[blue]Starting enhanced dataset quality assessment...[/blue]")

            # Calculate basic statistics (including diversity)
            dataset_stats = self._calculate_dataset_stats(df)

            # Execute traditional validations
            validation_results = self._execute_validation(df)

            # Execute Great Expectations validation if enabled
            ge_results = None
            diversity_assessment = None
            if self.ge_context:
                try:
                    console.print("[blue]Running Great Expectations validation...[/blue]")
                    ge_validation = self.ge_context.validate_dataset_with_diversity(df)
                    ge_results = ge_validation.get("great_expectations")
                    diversity_assessment = ge_validation.get("diversity_assessment")
                except Exception as e:
                    console.print(
                        f"[yellow]Great Expectations validation failed: {str(e)}[/yellow]"
                    )

            # Calculate overall score (enhanced with diversity)
            overall_score = self._calculate_enhanced_overall_score(
                validation_results, dataset_stats, ge_results
            )

            # Generate comprehensive recommendations
            recommendations = self._generate_enhanced_recommendations(
                validation_results, dataset_stats, overall_score, diversity_assessment
            )

            quality_metrics = QualityMetrics(
                overall_score=overall_score,
                validation_results=validation_results,
                dataset_stats=dataset_stats,
                recommendations=recommendations,
                great_expectations_results=ge_results,
                diversity_assessment=diversity_assessment,
            )

            # Enhanced reporting
            diversity_score = (
                dataset_stats.diversity_metrics.overall_diversity_score
                if dataset_stats.diversity_metrics
                else 0.0
            )
            console.print(
                f"[green]Quality assessment completed. Overall score: {overall_score:.1f}/100[/green]"
            )
            if diversity_score > 0:
                diversity_level = (
                    dataset_stats.diversity_metrics.diversity_level
                    if dataset_stats.diversity_metrics
                    else "unknown"
                )
                console.print(
                    f"[blue]Diversity score: {diversity_score:.1f}/100 ({diversity_level})[/blue]"
                )

            return quality_metrics

        except Exception as e:
            console.print(f"[red]Error during quality assessment: {str(e)}[/red]")
            raise

    def _calculate_enhanced_overall_score(
        self,
        validation_results: list[ValidationResult],
        dataset_stats: DatasetStats,
        ge_results: Optional[dict],
    ) -> float:
        """Calculate enhanced overall quality score including diversity metrics."""

        # Start with traditional score
        base_score = self._calculate_overall_score(validation_results, dataset_stats)

        # Enhance with diversity if available
        if dataset_stats.diversity_metrics:
            diversity_score = dataset_stats.diversity_metrics.overall_diversity_score
            # Weighted combination: 70% traditional quality, 30% diversity
            enhanced_score = (base_score * 0.7) + (diversity_score * 0.3)
        else:
            enhanced_score = base_score

        # Adjust based on Great Expectations results
        if ge_results and ge_results.get("success") is False:
            # Penalize for GE validation failures
            ge_success_rate = ge_results.get("results_summary", {}).get("success_percentage", 100)
            enhanced_score = enhanced_score * (ge_success_rate / 100)

        return max(0.0, min(100.0, enhanced_score))

    def _generate_enhanced_recommendations(
        self,
        validation_results: list[ValidationResult],
        dataset_stats: DatasetStats,
        overall_score: float,
        diversity_assessment: Optional[dict],
    ) -> list[str]:
        """Generate enhanced recommendations including diversity insights."""

        # Start with traditional recommendations
        recommendations = self._generate_recommendations(
            validation_results, dataset_stats, overall_score
        )

        # Add diversity-specific recommendations
        if diversity_assessment:
            diversity_recs = diversity_assessment.get("recommendations", [])
            recommendations.extend(diversity_recs)

        # Add specific diversity metric recommendations
        if dataset_stats.diversity_metrics:
            diversity_metrics = dataset_stats.diversity_metrics

            # Lexical diversity recommendations
            if diversity_metrics.lexical_diversity:
                ttr = diversity_metrics.lexical_diversity.get("type_token_ratio", 0)
                if ttr < self.config.min_lexical_diversity:
                    recommendations.append(
                        f"Low lexical diversity (TTR: {ttr:.3f}). Consider adding more varied vocabulary to improve dataset richness."
                    )

            # Role diversity recommendations
            if diversity_metrics.role_diversity:
                dominant_ratio = diversity_metrics.role_diversity.get("dominant_role_ratio", 0)
                if dominant_ratio > self.config.max_role_imbalance:
                    recommendations.append(
                        f"Role imbalance detected (dominant role: {dominant_ratio:.1%}). Balance conversation roles for better model training."
                    )

            # Semantic diversity recommendations (if available)
            if diversity_metrics.semantic_diversity:
                sem_score = diversity_metrics.semantic_diversity.get("semantic_diversity_score", 0)
                if sem_score < self.config.min_semantic_diversity:
                    recommendations.append(
                        f"Low semantic diversity (score: {sem_score:.3f}). Add more varied topics and contexts to improve content diversity."
                    )

        # Add installation recommendations if enhanced metrics failed
        if self.config.enable_enhanced_diversity and not dataset_stats.diversity_metrics:
            recommendations.append(
                "Enhanced diversity metrics unavailable. Install optional dependencies with: pip install nlpbench[diversity]"
            )

        return list(set(recommendations))  # Remove duplicates

    def cleanup(self):
        """Clean up temporary resources."""
        if self.ge_context:
            try:
                self.ge_context.cleanup()
            except Exception:
                pass
