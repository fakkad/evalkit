"""EvalKit CLI -- typer-based command interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from evalkit import __version__
from evalkit.models import TestSuite

app = typer.Typer(
    name="evalkit",
    help="CLI eval harness for LLMs.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    suite_path: Path = typer.Argument(..., help="Path to YAML eval suite"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save results"
    ),
    format: str = typer.Option(
        "both", "--format", "-f", help="Output format: json, html, or both"
    ),
) -> None:
    """Run an eval suite against a model."""
    from evalkit.report import generate_html, save_json
    from evalkit.runner import run_suite

    result = run_suite(suite_path)

    # Print summary
    _print_summary(result)

    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if format in ("json", "both"):
            json_path = save_json(result, output_dir / "results.json")
            console.print(f"[dim]JSON results: {json_path}[/dim]")
        if format in ("html", "both"):
            html_path = generate_html(result, output_dir / "report.html")
            console.print(f"[dim]HTML report: {html_path}[/dim]")

    if not result.passed:
        raise typer.Exit(code=1)


@app.command()
def compare(
    results1: Path = typer.Argument(..., help="First results JSON file"),
    results2: Path = typer.Argument(..., help="Second results JSON file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output HTML diff report path"
    ),
) -> None:
    """Compare two eval result sets and show regressions."""
    from evalkit.compare import generate_diff_html, print_comparison
    from evalkit.report import load_json

    r1 = load_json(results1)
    r2 = load_json(results2)

    print_comparison(r1, r2)

    if output:
        generate_diff_html(r1, r2, output)
        console.print(f"\n[dim]Diff report: {output}[/dim]")

    # Exit non-zero if regressions
    from evalkit.compare import compare_results

    comp = compare_results(r1, r2)
    if comp["regressions"]:
        raise typer.Exit(code=1)


@app.command()
def validate(
    suite_path: Path = typer.Argument(..., help="Path to YAML eval suite"),
) -> None:
    """Validate a YAML test suite configuration."""
    try:
        with suite_path.open() as f:
            data = yaml.safe_load(f)
        suite = TestSuite(**data)
        console.print(f"[green]Valid[/green] suite: {suite.name}")
        console.print(f"  Test cases: {len(suite.test_cases)}")
        console.print(f"  Metrics: {', '.join(m.type for m in suite.metrics)}")
        console.print(f"  Provider: {suite.model.provider}")
        console.print(f"  Model: {suite.model.model}")
        if suite.thresholds:
            console.print(f"  Thresholds: {suite.thresholds}")
    except Exception as e:
        console.print(f"[red]Invalid[/red]: {e}")
        raise typer.Exit(code=1)


@app.command()
def init(
    output: Path = typer.Option(
        Path("suite.yaml"), "--output", "-o", help="Output YAML path"
    ),
) -> None:
    """Generate an example YAML test suite."""
    example = {
        "name": "my-eval-suite",
        "description": "Example evaluation suite",
        "model": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "params": {"temperature": 0, "max_tokens": 1024},
        },
        "test_cases": [
            {
                "id": "greeting-1",
                "input": "Hello, I need help with my order",
                "expected_output": "I'd be happy to help you with your order",
                "metadata": {"category": "greeting"},
            },
            {
                "id": "factual-1",
                "input": "What is the capital of France?",
                "expected_output": "Paris",
                "metadata": {"category": "factual"},
            },
        ],
        "metrics": [
            {"type": "exact_match"},
            {
                "type": "semantic_similarity",
                "params": {"model": "text-embedding-3-small"},
            },
        ],
        "thresholds": {"exact_match": 0.5, "semantic_similarity": 0.8},
    }

    with output.open("w") as f:
        yaml.dump(example, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created[/green] example suite: {output}")


def _print_summary(result) -> None:
    """Print a rich summary table of the eval run."""
    console.print()
    table = Table(title="Eval Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    for metric_name, score in result.aggregate_scores.items():
        threshold = None
        for v in result.threshold_violations:
            if v.metric_name == metric_name:
                threshold = v.expected
                break

        threshold_str = f"{threshold:.2f}" if threshold is not None else "--"
        is_violation = threshold is not None
        status = "[red]FAIL[/red]" if is_violation else "[green]PASS[/green]"

        table.add_row(metric_name, f"{score:.4f}", threshold_str, status)

    console.print(table)

    overall = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
    console.print(f"\n  Suite: {result.suite_name} | Status: {overall}")
    console.print(
        f"  Cases: {result.total_cases} | Duration: {result.duration_ms:.0f}ms\n"
    )

    if result.threshold_violations:
        console.print("[red]Threshold violations:[/red]")
        for v in result.threshold_violations:
            console.print(
                f"  {v.metric_name}: {v.actual:.4f} < {v.expected:.4f}"
            )
        console.print()


if __name__ == "__main__":
    app()
