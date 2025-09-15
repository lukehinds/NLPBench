# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLPBench is a comprehensive tool for analyzing the quality of NLP datasets using Great Expectations and advanced diversity metrics. It provides quality assessments, diversity measurements, reports, and recommendations for improving datasets.

## Key Commands

### Development & Testing
```bash
# Run tests
uv run pytest
uv run pytest --cov=src --cov-report=html  # with coverage

# Run a single test file
uv run pytest tests/test_quality.py

# Code quality checks
uv run ruff check .              # Linting
uv run ruff check --fix .        # Auto-fix linting issues
uv run ruff format .             # Format code
uv run mypy src/                 # Type checking

# Run all checks
make check
```

### Running NLPBench
```bash
# Basic usage
uv run nlpbench --hf-repo databricks/databricks-dolly-15k
uv run nlpbench --kaggle-dataset username/dataset-name

# Generate HTML report
uv run nlpbench --hf-repo microsoft/DialoGPT-medium --output-format html --output-file report.html

# Quick inspection
uv run nlpbench inspect --hf-repo microsoft/DialoGPT-small --samples 3
```

### Building & Installation
```bash
# Install dependencies
uv sync
uv sync --extra diversity        # with enhanced diversity metrics
uv sync --group dev             # with development dependencies

# Build package
uv build
```

## Architecture

### Core Components
- **DatasetLoader** (`src/dataset_loader.py`): Handles HuggingFace dataset loading and format detection
- **KaggleDatasetLoader** (`src/kaggle_loader.py`): Handles Kaggle dataset integration
- **DatasetQualityChecker** (`src/quality_checker.py`): Main quality analysis engine
- **DiversityMetrics** (`src/diversity_metrics.py`): Calculates lexical, semantic, syntactic, and topic diversity
- **GreatExpectationsValidator** (`src/great_expectations_utils.py`): Integrates Great Expectations framework for validation
- **ReportGenerator** (`src/report_generator.py`): Generates console, JSON, and HTML reports

### Dataset Format Detection
The system automatically detects various dataset formats:
- Conversations format: `{"conversations": [...]}`
- Messages format: `{"messages": [...]}`
- Instruction-following: `{"instruction": ..., "input": ..., "output": ...}`
- Q&A format: `{"question": ..., "answer": ...}`
- Prompt-response: `{"prompt": ..., "response": ...}`

### Quality Metrics Flow
1. Dataset is loaded and format is auto-detected
2. Schema validation checks required columns and data types
3. Content quality analysis measures lengths, duplicates, role consistency
4. Diversity metrics calculate various diversity scores (basic and enhanced)
5. Great Expectations runs comprehensive validation suite
6. Overall quality score (0-100) is calculated with weighted components
7. Report is generated in requested format

### Dependency Structure
- **Core**: Always available - basic quality checks, simple diversity metrics
- **Enhanced (optional)**: Advanced diversity metrics requiring sentence-transformers, spaCy, etc.
- Dependencies are checked at runtime; features gracefully degrade if optional packages missing

## Configuration

Configuration is handled via JSON files (`.nlpbench.json`):
- `min_content_length`, `max_content_length`: Content length boundaries
- `allowed_roles`: Valid conversation roles
- `duplicate_threshold`: Maximum allowed duplicate percentage
- `quality_thresholds`: Score boundaries for quality levels

## Testing Approach

Tests are organized by component:
- `test_quality.py`: Core quality checking functionality
- `test_diversity_metrics.py`: Diversity calculation tests
- `test_great_expectations_integration.py`: Great Expectations validation
- `test_enhanced_quality_checker.py`: Enhanced feature tests

Run specific test categories with pytest markers:
- `pytest -m "not slow"`: Skip slow tests
- `pytest -m integration`: Run only integration tests