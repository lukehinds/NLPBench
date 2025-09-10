from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ConversationEntry(BaseModel):
    """Model for a single conversation entry with role and content."""

    role: str = Field(..., description="Role of the speaker (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")

    @field_validator("role")
    def validate_role(cls, v):
        """Ensure role is not empty."""
        if not v or not v.strip():
            raise ValueError("Role cannot be empty")
        return v.strip().lower()

    @field_validator("content")
    def validate_content(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()


class DatasetStats(BaseModel):
    """Basic statistics about the dataset."""

    total_entries: int = Field(..., description="Total number of entries in the dataset")
    unique_entries: int = Field(..., description="Number of unique entries")
    duplicate_count: int = Field(..., description="Number of duplicate entries")
    role_distribution: dict[str, int] = Field(
        default_factory=dict, description="Distribution of roles"
    )
    avg_content_length: float = Field(..., description="Average content length in characters")
    min_content_length: int = Field(..., description="Minimum content length")
    max_content_length: int = Field(..., description="Maximum content length")
    empty_content_count: int = Field(default=0, description="Number of entries with empty content")


class ValidationResult(BaseModel):
    """Result of a single validation check."""

    check_name: str = Field(..., description="Name of the validation check")
    success: bool = Field(..., description="Whether the check passed")
    success_percentage: float = Field(..., description="Percentage of successful validations")
    failed_count: int = Field(default=0, description="Number of failed validations")
    total_count: int = Field(..., description="Total number of items checked")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional validation details"
    )
    error_message: Optional[str] = Field(default=None, description="Error message if check failed")


class QualityMetrics(BaseModel):
    """Comprehensive quality metrics for the dataset."""

    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")
    validation_results: list[ValidationResult] = Field(
        default_factory=list, description="Individual validation results"
    )
    dataset_stats: DatasetStats = Field(..., description="Basic dataset statistics")
    recommendations: list[str] = Field(
        default_factory=list, description="Quality improvement recommendations"
    )


class QualityReport(BaseModel):
    """Complete quality assessment report."""

    dataset_name: str = Field(..., description="Name of the analyzed dataset")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Report generation timestamp"
    )
    quality_metrics: QualityMetrics = Field(..., description="Quality assessment metrics")
    execution_time: float = Field(..., description="Time taken to generate the report in seconds")
    config_used: dict[str, Any] = Field(
        default_factory=dict, description="Configuration used for the analysis"
    )


class DatasetConfig(BaseModel):
    """Configuration for dataset quality assessment."""

    min_content_length: int = Field(default=10, description="Minimum expected content length")
    max_content_length: int = Field(default=50000, description="Maximum expected content length")
    allowed_roles: list[str] = Field(
        default=["user", "assistant", "system", "document"], description="Allowed role values"
    )
    duplicate_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Maximum allowed duplicate percentage"
    )
    empty_content_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Maximum allowed empty content percentage"
    )
    quality_thresholds: dict[str, float] = Field(
        default={"excellent": 95.0, "good": 85.0, "fair": 70.0, "poor": 50.0},
        description="Quality score thresholds",
    )

    @field_validator("allowed_roles")
    def validate_allowed_roles(cls, v):
        """Ensure allowed roles are normalized."""
        return [role.strip().lower() for role in v if role.strip()]


class HuggingFaceDatasetInfo(BaseModel):
    """Information about a Hugging Face dataset."""

    repo_id: str = Field(..., description="Dataset repository ID")
    config_name: Optional[str] = Field(default=None, description="Dataset configuration name")
    split: str = Field(default="train", description="Dataset split to analyze")
    revision: Optional[str] = Field(default=None, description="Dataset revision/branch")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory for the dataset")
    token: Optional[str] = Field(default=None, description="Hugging Face authentication token")
