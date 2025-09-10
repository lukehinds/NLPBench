import json
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .models import DatasetStats, QualityReport, ValidationResult

console = Console(record=True)


class ReportGenerator:
    """Generates comprehensive quality reports in various formats."""

    def __init__(self):
        self.console = console

    def generate_console_report(self, report: QualityReport) -> None:
        """Generate a rich console report."""

        self.console.print()
        self.console.rule(f"[bold blue]Dataset Quality Report: {report.dataset_name}[/bold blue]")
        self.console.print()

        # Overall summary panel
        self._print_summary_panel(report)

        # Dataset statistics
        self._print_dataset_stats(report.quality_metrics.dataset_stats)

        # Validation results
        self._print_validation_results(report.quality_metrics.validation_results)

        # Quality score breakdown
        self._print_quality_score(report.quality_metrics.overall_score)

        # Recommendations
        self._print_recommendations(report.quality_metrics.recommendations)

        # Footer
        self._print_footer(report)

    def _print_summary_panel(self, report: QualityReport) -> None:
        """Print the summary panel."""

        stats = report.quality_metrics.dataset_stats
        score = report.quality_metrics.overall_score

        # Determine quality level and color
        quality_level, color = self._get_quality_level_and_color(score)

        summary_text = f"""
[bold]Dataset:[/bold] {report.dataset_name}
[bold]Total Entries:[/bold] {stats.total_entries:,}
[bold]Unique Entries:[/bold] {stats.unique_entries:,}
[bold]Quality Score:[/bold] [{color}]{score:.1f}/100 ({quality_level})[/{color}]
[bold]Analysis Time:[/bold] {report.execution_time:.2f}s
[bold]Generated:[/bold] {report.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
        """.strip()

        panel = Panel(
            summary_text,
            title="[bold]Quality Assessment Summary[/bold]",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def _print_dataset_stats(self, stats: DatasetStats) -> None:
        """Print dataset statistics table."""

        table = Table(title="Dataset Statistics", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Details", style="dim")

        # Basic stats
        table.add_row("Total Entries", f"{stats.total_entries:,}", "")
        table.add_row(
            "Unique Entries",
            f"{stats.unique_entries:,}",
            f"{(stats.unique_entries / stats.total_entries * 100):.1f}% unique",
        )
        table.add_row(
            "Duplicates",
            f"{stats.duplicate_count:,}",
            f"{(stats.duplicate_count / stats.total_entries * 100):.1f}% of total",
        )
        table.add_row(
            "Empty Content",
            f"{stats.empty_content_count:,}",
            f"{(stats.empty_content_count / stats.total_entries * 100):.1f}% of total",
        )

        # Content length stats
        table.add_row("Avg Content Length", f"{stats.avg_content_length:.0f}", "characters")
        table.add_row("Min Content Length", f"{stats.min_content_length:,}", "characters")
        table.add_row("Max Content Length", f"{stats.max_content_length:,}", "characters")

        self.console.print(table)
        self.console.print()

        # Role distribution
        if stats.role_distribution:
            role_table = Table(title="Role Distribution", border_style="green")
            role_table.add_column("Role", style="bold")
            role_table.add_column("Count", justify="right")
            role_table.add_column("Percentage", justify="right")

            total_roles = sum(stats.role_distribution.values())
            for role, count in sorted(stats.role_distribution.items()):
                percentage = (count / total_roles * 100) if total_roles > 0 else 0
                role_table.add_row(role, f"{count:,}", f"{percentage:.1f}%")

            self.console.print(role_table)
            self.console.print()

    def _print_validation_results(self, validation_results: list[ValidationResult]) -> None:
        """Print validation results table."""

        table = Table(title="Validation Results", border_style="yellow")
        table.add_column("Check", style="bold", width=30)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Success Rate", justify="right", width=12)
        table.add_column("Failed/Total", justify="right", width=15)

        for result in validation_results:
            status_text, status_color = self._get_validation_status(
                result.success_percentage, result.success
            )
            success_rate_color = self._get_success_rate_color(result.success_percentage)

            table.add_row(
                result.check_name,
                f"[{status_color}]{status_text}[/{status_color}]",
                f"[{success_rate_color}]{result.success_percentage:.1f}%[/{success_rate_color}]",
                f"{result.failed_count:,}/{result.total_count:,}",
            )

        self.console.print(table)
        self.console.print()

    def _print_quality_score(self, score: float) -> None:
        """Print quality score with visual representation."""

        quality_level, color = self._get_quality_level_and_color(score)

        # Create a visual bar
        bar_length = 50
        filled_length = int((score / 100) * bar_length)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        score_text = f"""
[bold]Overall Quality Score[/bold]
[{color}]{bar}[/{color}]
[bold {color}]{score:.1f}/100 - {quality_level}[/bold {color}]
        """.strip()

        panel = Panel(score_text, border_style=color, padding=(1, 2))

        self.console.print(panel)
        self.console.print()

    def _print_recommendations(self, recommendations: list[str]) -> None:
        """Print recommendations."""

        if not recommendations:
            return

        tree = Tree("[bold]Quality Improvement Recommendations[/bold]")

        for i, recommendation in enumerate(recommendations, 1):
            tree.add(f"{i}. {recommendation}")

        panel = Panel(tree, border_style="magenta", padding=(1, 2))

        self.console.print(panel)
        self.console.print()

    def _print_footer(self, report: QualityReport) -> None:
        """Print report footer."""

        footer_text = (
            f"Report generated by NLPBench on {report.timestamp.strftime('%Y-%m-%d at %H:%M:%S')}"
        )
        self.console.print(Align.center(f"[dim]{footer_text}[/dim]"))
        self.console.rule(style="dim")

    def _get_quality_level_and_color(self, score: float) -> tuple[str, str]:
        """Get quality level description and color based on score."""

        if score >= 95:
            return "Excellent", "bright_green"
        elif score >= 85:
            return "Good", "green"
        elif score >= 70:
            return "Fair", "yellow"
        elif score >= 50:
            return "Poor", "red"
        else:
            return "Very Poor", "bright_red"

    def _get_success_rate_color(self, rate: float) -> str:
        """Get color for success rate display."""

        if rate >= 95:
            return "bright_green"
        elif rate >= 80:
            return "green"
        elif rate >= 60:
            return "yellow"
        else:
            return "red"

    def _get_validation_status(
        self, success_percentage: float, is_success: bool
    ) -> tuple[str, str]:
        """Get validation status text and color with granular levels."""

        if success_percentage == 100.0:
            return "✓ PASS", "bright_green"
        elif success_percentage >= 90.0:
            return "◉ GOOD", "green"
        elif success_percentage >= 75.0:
            return "◑ WARN", "yellow"
        elif success_percentage >= 50.0:
            return "◑ POOR", "orange"
        else:
            return "✗ FAIL", "red"

    def generate_json_report(
        self, report: QualityReport, output_path: Optional[Path] = None
    ) -> str:
        """Generate a JSON report."""

        # Convert to dict for JSON serialization
        report_dict = self._report_to_dict(report)

        json_content = json.dumps(report_dict, indent=2, ensure_ascii=False, default=str)

        if output_path:
            output_path.write_text(json_content, encoding="utf-8")
            self.console.print(f"[green]JSON report saved to: {output_path}[/green]")

        return json_content

    def generate_html_report(
        self, report: QualityReport, output_path: Optional[Path] = None
    ) -> str:
        """Generate an HTML report."""

        html_template = self._get_html_template()

        # Convert report to dict for template rendering
        report_dict = self._report_to_dict(report)

        # Add additional computed values for template
        report_dict["quality_level"], report_dict["quality_color"] = (
            self._get_quality_level_and_color(report.quality_metrics.overall_score)
        )

        # Render template
        template = Template(html_template)
        html_content = template.render(report=report_dict)

        if output_path:
            output_path.write_text(html_content, encoding="utf-8")
            self.console.print(f"[green]HTML report saved to: {output_path}[/green]")

        return html_content

    def _report_to_dict(self, report: QualityReport) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""

        return {
            "dataset_name": report.dataset_name,
            "timestamp": report.timestamp.isoformat(),
            "execution_time": report.execution_time,
            "overall_score": report.quality_metrics.overall_score,
            "dataset_stats": {
                "total_entries": report.quality_metrics.dataset_stats.total_entries,
                "unique_entries": report.quality_metrics.dataset_stats.unique_entries,
                "duplicate_count": report.quality_metrics.dataset_stats.duplicate_count,
                "role_distribution": report.quality_metrics.dataset_stats.role_distribution,
                "avg_content_length": report.quality_metrics.dataset_stats.avg_content_length,
                "min_content_length": report.quality_metrics.dataset_stats.min_content_length,
                "max_content_length": report.quality_metrics.dataset_stats.max_content_length,
                "empty_content_count": report.quality_metrics.dataset_stats.empty_content_count,
            },
            "validation_results": [
                {
                    "check_name": result.check_name,
                    "success": result.success,
                    "success_percentage": result.success_percentage,
                    "failed_count": result.failed_count,
                    "total_count": result.total_count,
                    "error_message": result.error_message,
                }
                for result in report.quality_metrics.validation_results
            ],
            "recommendations": report.quality_metrics.recommendations,
            "config_used": report.config_used,
        }

    def _get_html_template(self) -> str:
        """Get HTML template for report generation."""

        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Quality Report - {{ report.dataset_name }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .summary {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .score-bar {
            width: 100%;
            height: 30px;
            background-color: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .score-fill {
            height: 100%;
            transition: width 0.5s ease;
        }
        .excellent { background-color: #28a745; }
        .good { background-color: #20c997; }
        .fair { background-color: #ffc107; }
        .poor { background-color: #fd7e14; }
        .very-poor { background-color: #dc3545; }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #495057;
            color: white;
            font-weight: 600;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .recommendations {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .footer {
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            padding-top: 20px;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Dataset Quality Report</h1>
            <h2>{{ report.dataset_name }}</h2>
        </div>

        <div class="summary">
            <h3>Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div>
                    <strong>Total Entries:</strong> {{ "{:,}".format(report.dataset_stats.total_entries) }}<br>
                    <strong>Unique Entries:</strong> {{ "{:,}".format(report.dataset_stats.unique_entries) }}<br>
                    <strong>Duplicates:</strong> {{ "{:,}".format(report.dataset_stats.duplicate_count) }}
                </div>
                <div>
                    <strong>Quality Score:</strong> {{ "%.1f"|format(report.overall_score) }}/100<br>
                    <strong>Quality Level:</strong> {{ report.quality_level }}<br>
                    <strong>Analysis Time:</strong> {{ "%.2f"|format(report.execution_time) }}s
                </div>
            </div>

            <div style="margin-top: 20px;">
                <strong>Quality Score:</strong>
                <div class="score-bar">
                    <div class="score-fill {{ report.quality_color.replace('_', '-') }}"
                         style="width: {{ report.overall_score }}%"></div>
                </div>
                <div style="text-align: center; font-weight: bold;">
                    {{ "%.1f"|format(report.overall_score) }}/100 - {{ report.quality_level }}
                </div>
            </div>
        </div>

        <h3>Validation Results</h3>
        <table>
            <thead>
                <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Success Rate</th>
                    <th>Failed/Total</th>
                </tr>
            </thead>
            <tbody>
                {% for result in report.validation_results %}
                <tr>
                    <td>{{ result.check_name }}</td>
                    <td class="{{ 'pass' if result.success else 'fail' }}">
                        {{ '✓ PASS' if result.success else '✗ FAIL' }}
                    </td>
                    <td>{{ "%.1f"|format(result.success_percentage) }}%</td>
                    <td>{{ "{:,}".format(result.failed_count) }}/{{ "{:,}".format(result.total_count) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h3>Dataset Statistics</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Average Content Length</td><td>{{ "%.0f"|format(report.dataset_stats.avg_content_length) }} characters</td></tr>
                <tr><td>Min Content Length</td><td>{{ "{:,}".format(report.dataset_stats.min_content_length) }} characters</td></tr>
                <tr><td>Max Content Length</td><td>{{ "{:,}".format(report.dataset_stats.max_content_length) }} characters</td></tr>
                <tr><td>Empty Content Count</td><td>{{ "{:,}".format(report.dataset_stats.empty_content_count) }}</td></tr>
            </tbody>
        </table>

        {% if report.dataset_stats.role_distribution %}
        <h3>Role Distribution</h3>
        <table>
            <thead>
                <tr>
                    <th>Role</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {% for role, count in report.dataset_stats.role_distribution.items() %}
                <tr>
                    <td>{{ role }}</td>
                    <td>{{ "{:,}".format(count) }}</td>
                    <td>{{ "%.1f"|format(count / report.dataset_stats.total_entries * 100) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if report.recommendations %}
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>
                {% for recommendation in report.recommendations %}
                <li>{{ recommendation }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <div class="footer">
            <p>Report generated by NLPBench on {{ report.timestamp }}</p>
        </div>
    </div>
</body>
</html>
        """.strip()

    def save_console_output(self, output_path: Path) -> None:
        """Save the console output to a text file."""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.console.export_text())

        self.console.print(f"[green]Console output saved to: {output_path}[/green]")
