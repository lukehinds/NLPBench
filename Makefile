.PHONY: help install install-dev test test-cov lint format check clean build docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	uv sync

install-dev: ## Install the package with development dependencies
	uv sync --group dev

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting (ruff check)
	uv run ruff check .

lint-fix: ## Run linting with auto-fix
	uv run ruff check --fix .

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check code formatting
	uv run ruff format --check .

type-check: ## Run type checking with mypy
	uv run mypy src/

check: lint format-check type-check ## Run all code quality checks

clean: ## Clean up temporary files and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build the package
	uv build

publish-test: build ## Publish to Test PyPI
	uv publish --repository testpypi

publish: build ## Publish to PyPI
	uv publish

docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

# Development commands
dev-setup: ## Set up development environment
	uv sync --group dev
	uv run pre-commit install

pre-commit: ## Run pre-commit hooks
	uv run pre-commit run --all-files

# NLPBench specific commands
demo: ## Run a demo analysis on example data
	uv run nlpbench --hf-repo microsoft/DialoGPT-small --output-format console

demo-html: ## Generate HTML report demo
	uv run nlpbench --hf-repo microsoft/DialoGPT-small --output-format html --output-file demo_report.html

config: ## Create default configuration
	uv run nlpbench init-config

inspect-demo: ## Inspect a small dataset
	uv run nlpbench inspect --hf-repo microsoft/DialoGPT-small --samples 3