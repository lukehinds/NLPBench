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


class DiversityMetrics(BaseModel):
    """Comprehensive diversity metrics for the dataset."""

    # Basic diversity metrics (always available)
    lexical_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Lexical diversity metrics"
    )
    length_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Content length diversity metrics"
    )
    character_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Character-level diversity metrics"
    )
    word_frequency_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Word frequency diversity metrics"
    )
    role_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Role distribution diversity metrics"
    )

    # Enhanced diversity metrics (optional dependencies)
    advanced_lexical_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Advanced lexical diversity (MTLD, MATTR)"
    )
    semantic_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Semantic diversity using embeddings"
    )
    syntactic_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Syntactic diversity (POS, dependencies)"
    )
    topic_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Topic diversity using LDA"
    )
    readability_diversity: Optional[dict[str, float]] = Field(
        default=None, description="Readability score diversity"
    )

    # Overall diversity score
    overall_diversity_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall diversity score (0-100)"
    )
    diversity_level: str = Field(default="unknown", description="Diversity quality level")


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

    # Enhanced with diversity information
    diversity_metrics: Optional[DiversityMetrics] = Field(
        default=None, description="Comprehensive diversity metrics"
    )


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

    # Great Expectations integration
    great_expectations_results: Optional[dict[str, Any]] = Field(
        default=None, description="Great Expectations validation results"
    )

    # Enhanced diversity assessment
    diversity_assessment: Optional[dict[str, Any]] = Field(
        default=None, description="Comprehensive diversity quality assessment"
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

    # Diversity configuration
    enable_diversity_metrics: bool = Field(
        default=True, description="Enable diversity metrics calculation"
    )
    enable_enhanced_diversity: bool = Field(
        default=True, description="Enable enhanced diversity metrics (requires optional deps)"
    )

    diversity_thresholds: dict[str, float] = Field(
        default={"excellent": 80.0, "good": 65.0, "fair": 50.0, "poor": 35.0},
        description="Diversity score thresholds",
    )

    # Specific diversity metric thresholds
    min_lexical_diversity: float = Field(
        default=0.1, description="Minimum acceptable type-token ratio"
    )
    max_role_imbalance: float = Field(
        default=0.8, description="Maximum allowed dominant role ratio"
    )
    min_semantic_diversity: float = Field(
        default=0.3, description="Minimum semantic diversity score"
    )

    # Great Expectations configuration
    use_great_expectations: bool = Field(
        default=True, description="Enable Great Expectations validation"
    )
    ge_context_root: Optional[str] = Field(
        default=None, description="Great Expectations context root directory"
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


class KaggleDatasetInfo(BaseModel):
    """Information about a Kaggle dataset."""

    dataset_id: str = Field(..., description="Kaggle dataset ID (e.g., 'username/dataset-name')")
    version: Optional[str] = Field(default=None, description="Dataset version (default: latest)")
    path: Optional[str] = Field(default=None, description="Specific file path within dataset")
    force_download: bool = Field(default=False, description="Force re-download even if cached")


class DatasetInfo(BaseModel):
    """Generic dataset information that supports multiple sources."""

    source_type: str = Field(..., description="Type of dataset source ('huggingface' or 'kaggle')")
    huggingface_info: Optional[HuggingFaceDatasetInfo] = Field(
        default=None, description="HuggingFace dataset info"
    )
    kaggle_info: Optional[KaggleDatasetInfo] = Field(
        default=None, description="Kaggle dataset info"
    )

    def __post_init__(self):
        """Validate that appropriate info is provided for the source type."""
        if self.source_type == "huggingface" and self.huggingface_info is None:
            raise ValueError("huggingface_info is required when source_type is 'huggingface'")
        if self.source_type == "kaggle" and self.kaggle_info is None:
            raise ValueError("kaggle_info is required when source_type is 'kaggle'")
