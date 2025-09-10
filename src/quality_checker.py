import pandas as pd
from rich.console import Console

from .models import DatasetConfig, DatasetStats, QualityMetrics, ValidationResult

console = Console()


class DatasetQualityChecker:
    """Uses pandas and custom validation logic to perform quality checks on datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    def _calculate_dataset_stats(self, df: pd.DataFrame) -> DatasetStats:
        """Calculate basic statistics about the dataset."""

        # Role distribution
        role_dist = df["role"].value_counts().to_dict()

        # Content length statistics
        content_lengths = df["content_length"]

        # Count duplicates based on content
        duplicate_count = df.duplicated(subset=["content"]).sum()

        # Empty content count
        empty_content_count = df["has_empty_content"].sum()

        return DatasetStats(
            total_entries=len(df),
            unique_entries=df["content"].nunique(),
            duplicate_count=int(duplicate_count),
            role_distribution=role_dist,
            avg_content_length=float(content_lengths.mean()),
            min_content_length=int(content_lengths.min()),
            max_content_length=int(content_lengths.max()),
            empty_content_count=int(empty_content_count),
        )

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
        """Perform comprehensive quality assessment on the dataset."""

        try:
            console.print("[blue]Starting dataset quality assessment...[/blue]")

            # Calculate basic statistics
            dataset_stats = self._calculate_dataset_stats(df)

            # Execute validations
            validation_results = self._execute_validation(df)

            # Calculate overall score
            overall_score = self._calculate_overall_score(validation_results, dataset_stats)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                validation_results, dataset_stats, overall_score
            )

            quality_metrics = QualityMetrics(
                overall_score=overall_score,
                validation_results=validation_results,
                dataset_stats=dataset_stats,
                recommendations=recommendations,
            )

            console.print(
                f"[green]Quality assessment completed. Overall score: {overall_score:.1f}/100[/green]"
            )

            return quality_metrics

        except Exception as e:
            console.print(f"[red]Error during quality assessment: {str(e)}[/red]")
            raise

    def cleanup(self):
        """Clean up temporary resources (no-op for this implementation)."""
        pass
