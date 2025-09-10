import time
from pathlib import Path

import click
from rich.console import Console

from .config import get_config_example, load_config, save_default_config
from .dataset_loader import DatasetLoader
from .models import HuggingFaceDatasetInfo, QualityReport
from .quality_checker import DatasetQualityChecker
from .report_generator import ReportGenerator

console = Console()


@click.group(invoke_without_command=True)
@click.option(
    "--hf-repo",
    type=str,
    help="Hugging Face dataset repository (e.g., 'databricks/databricks-dolly-15k')",
)
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--config-name", type=str, help="Dataset configuration name")
@click.option(
    "--split", type=str, default="train", help="Dataset split to analyze (default: train)"
)
@click.option(
    "--output-format",
    type=click.Choice(["console", "json", "html", "all"]),
    default="console",
    help="Output format (default: console)",
)
@click.option(
    "--output-file", type=click.Path(), help="Output file path (required for json/html formats)"
)
@click.option("--cache-dir", type=click.Path(), help="Cache directory for datasets")
@click.option("--token", type=str, help="Hugging Face authentication token")
@click.pass_context
def cli(ctx, hf_repo, config, config_name, split, output_format, output_file, cache_dir, token):
    """
    NLPBench - Dataset Quality Measurement Tool

    Analyze dataset quality using Great Expectations and generate comprehensive reports.

    Examples:
      nlpbench --hf-repo databricks/databricks-dolly-15k
      nlpbench --hf-repo microsoft/DialoGPT-medium --output-format html --output-file report.html
      nlpbench analyze --hf-repo org/dataset --config custom_config.json
    """

    # If no command is provided, run the analyze command
    if ctx.invoked_subcommand is None:
        if not hf_repo:
            console.print(
                "[red]Error: --hf-repo is required when running without a subcommand[/red]"
            )
            console.print("Use 'nlpbench --help' for usage information.")
            ctx.exit(1)

        ctx.invoke(
            analyze,
            hf_repo=hf_repo,
            config=config,
            config_name=config_name,
            split=split,
            output_format=output_format,
            output_file=output_file,
            cache_dir=cache_dir,
            token=token,
        )


@cli.command()
@click.option("--hf-repo", type=str, required=True, help="Hugging Face dataset repository")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--config-name", type=str, help="Dataset configuration name")
@click.option("--split", type=str, default="train", help="Dataset split to analyze")
@click.option(
    "--output-format",
    type=click.Choice(["console", "json", "html", "all"]),
    default="console",
    help="Output format",
)
@click.option("--output-file", type=click.Path(), help="Output file path")
@click.option("--cache-dir", type=click.Path(), help="Cache directory for datasets")
@click.option("--token", type=str, help="Hugging Face authentication token")
def analyze(hf_repo, config, config_name, split, output_format, output_file, cache_dir, token):
    """Analyze a Hugging Face dataset for quality issues."""

    try:
        start_time = time.time()

        # Load configuration
        dataset_config = load_config(config)
        console.print(f"[blue]Starting quality analysis for: {hf_repo}[/blue]")

        # Validate output requirements
        if output_format in ["json", "html"] and not output_file:
            console.print(f"[red]Error: --output-file is required for {output_format} format[/red]")
            return

        # Create dataset info
        dataset_info = HuggingFaceDatasetInfo(
            repo_id=hf_repo, config_name=config_name, split=split, cache_dir=cache_dir, token=token
        )

        # Load and process dataset
        console.print("[blue]Loading dataset...[/blue]")
        loader = DatasetLoader(dataset_info)
        loader.load_dataset()
        loader.process_dataset()

        if not loader.processed_data:
            console.print("[red]Error: No data could be processed from the dataset[/red]")
            return

        # Convert to DataFrame for quality checking
        df = loader.to_dataframe()
        console.print(f"[green]Dataset loaded successfully: {len(df)} entries[/green]")

        # Display sample data
        sample_data = loader.get_sample_data(3)
        if sample_data:
            console.print("\n[blue]Sample entries:[/blue]")
            for i, sample in enumerate(sample_data, 1):
                console.print(
                    f"[dim]{i}. [{sample['role']}] {sample['content']} (length: {sample['content_length']})[/dim]"
                )

        # Perform quality assessment
        console.print("\n[blue]Performing quality assessment...[/blue]")
        quality_checker = DatasetQualityChecker(dataset_config)

        try:
            quality_metrics = quality_checker.assess_quality(df)

            # Create report
            execution_time = time.time() - start_time
            report = QualityReport(
                dataset_name=hf_repo,
                quality_metrics=quality_metrics,
                execution_time=execution_time,
                config_used=dataset_config.dict(),
            )

            # Generate reports
            report_generator = ReportGenerator()

            if output_format == "console" or output_format == "all":
                console.print("\n")
                report_generator.generate_console_report(report)

            if output_format == "json" or output_format == "all":
                json_file = (
                    Path(output_file)
                    if output_file
                    else Path(f"{hf_repo.replace('/', '_')}_quality_report.json")
                )
                report_generator.generate_json_report(report, json_file)

            if output_format == "html" or output_format == "all":
                html_file = (
                    Path(output_file)
                    if output_file
                    else Path(f"{hf_repo.replace('/', '_')}_quality_report.html")
                )
                report_generator.generate_html_report(report, html_file)

            # Summary message
            quality_level = (
                "excellent"
                if quality_metrics.overall_score >= 95
                else "good"
                if quality_metrics.overall_score >= 85
                else "fair"
                if quality_metrics.overall_score >= 70
                else "poor"
            )

            console.print("\n[green]Analysis completed successfully![/green]")
            console.print(
                f"[blue]Overall quality score: {quality_metrics.overall_score:.1f}/100 ({quality_level})[/blue]"
            )
            console.print(f"[dim]Analysis time: {execution_time:.2f} seconds[/dim]")

        finally:
            # Clean up resources
            quality_checker.cleanup()

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during analysis: {str(e)}[/red]")
        raise


@cli.command()
@click.option(
    "--output", type=click.Path(), default=".nlpbench.json", help="Output config file path"
)
@click.option(
    "--format",
    "config_format",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Configuration file format",
)
def init_config(output, config_format):
    """Create a default configuration file."""

    try:
        save_default_config(output)
        console.print(f"[green]Default configuration saved to: {output}[/green]")
        console.print("[blue]Edit this file to customize quality assessment parameters.[/blue]")
    except Exception as e:
        console.print(f"[red]Error creating config file: {str(e)}[/red]")


@cli.command()
@click.option(
    "--format",
    "config_format",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Configuration format to display",
)
def show_config(config_format):
    """Show example configuration."""

    try:
        example_config = get_config_example()
        console.print("[blue]Example configuration:[/blue]\n")
        console.print(example_config)
    except Exception as e:
        console.print(f"[red]Error showing config: {str(e)}[/red]")


@cli.command()
@click.option("--hf-repo", type=str, required=True, help="Hugging Face dataset repository")
@click.option("--config-name", type=str, help="Dataset configuration name")
@click.option("--split", type=str, default="train", help="Dataset split to inspect")
@click.option("--samples", type=int, default=5, help="Number of samples to show")
@click.option("--cache-dir", type=click.Path(), help="Cache directory for datasets")
@click.option("--token", type=str, help="Hugging Face authentication token")
def inspect(hf_repo, config_name, split, samples, cache_dir, token):
    """Inspect a dataset without running full quality analysis."""

    try:
        console.print(f"[blue]Inspecting dataset: {hf_repo}[/blue]")

        dataset_info = HuggingFaceDatasetInfo(
            repo_id=hf_repo, config_name=config_name, split=split, cache_dir=cache_dir, token=token
        )

        loader = DatasetLoader(dataset_info)
        loader.load_dataset()
        loader.process_dataset()

        # Show dataset info
        info = loader.get_dataset_info()
        console.print("\n[green]Dataset Info:[/green]")
        console.print(f"  Repository: {info['repo_id']}")
        console.print(f"  Split: {info['split']}")
        console.print(f"  Raw entries: {info['raw_entries']:,}")
        console.print(f"  Processed entries: {info['processed_entries']:,}")
        console.print(f"  Features: {info.get('features', [])}")

        # Show role distribution
        if "role_distribution" in info:
            console.print("\n[green]Role Distribution:[/green]")
            for role, count in info["role_distribution"].items():
                console.print(f"  {role}: {count:,}")

        # Show sample data
        sample_data = loader.get_sample_data(samples)
        if sample_data:
            console.print(f"\n[green]Sample entries (showing {len(sample_data)}):[/green]")
            for i, sample in enumerate(sample_data, 1):
                console.print(f"\n[dim]{i}. Role: {sample['role']}[/dim]")
                console.print(f"[dim]   Content: {sample['content']}[/dim]")
                console.print(f"[dim]   Length: {sample['content_length']} characters[/dim]")

    except Exception as e:
        console.print(f"[red]Error during inspection: {str(e)}[/red]")


@cli.command()
def version():
    """Show version information."""
    console.print("NLPBench v1.0.0")
    console.print("Hugging Face Dataset Quality Measurement Tool")
    console.print("Built with Great Expectations, Rich, and Click")


if __name__ == "__main__":
    cli()
