"""EvalKit CLI — typer-based command interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from evalkit import __version__

app = typer.Typer(
    name="evalkit",
    help="CLI-first LLM evaluation framework.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    suite: Path = typer.Argument(..., help="Path to YAML eval suite"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save results to JSONL"
    ),
) -> None:
    """Run an eval suite against a model."""
    from evalkit.runner import run_suite

    result = run_suite(suite_path=suite, model_override=model, output_path=output)
    if not result.passed:
        raise typer.Exit(code=1)


@app.command()
def compare(
    baseline: Path = typer.Argument(..., help="Baseline results JSONL"),
    current: Path = typer.Argument(..., help="Current results JSONL"),
) -> None:
    """Compare two result sets and show regressions."""
    from evalkit.loader import load_results

    base = load_results(baseline)
    curr = load_results(current)

    base_map = {r.case_id: r for r in base.results}
    curr_map = {r.case_id: r for r in curr.results}

    all_ids = sorted(set(base_map) | set(curr_map))

    table = Table(title="Comparison: Baseline vs Current")
    table.add_column("Case ID", style="bold")
    table.add_column("Baseline", justify="center")
    table.add_column("Current", justify="center")
    table.add_column("Delta", justify="right")
    table.add_column("Status")

    regressions = 0
    improvements = 0

    for case_id in all_ids:
        b = base_map.get(case_id)
        c = curr_map.get(case_id)

        b_score = f"{b.weighted_score:.3f}" if b else "—"
        c_score = f"{c.weighted_score:.3f}" if c else "—"

        if b and c:
            delta = c.weighted_score - b.weighted_score
            delta_str = f"{delta:+.3f}"
            if c.passed and not b.passed:
                status = "[green]FIXED[/green]"
                improvements += 1
            elif not c.passed and b.passed:
                status = "[red]REGRESSED[/red]"
                regressions += 1
            elif delta > 0.01:
                status = "[green]IMPROVED[/green]"
                improvements += 1
            elif delta < -0.01:
                status = "[yellow]DEGRADED[/yellow]"
                regressions += 1
            else:
                status = "[dim]UNCHANGED[/dim]"
        elif c and not b:
            delta_str = "NEW"
            status = "[blue]NEW[/blue]"
        else:
            delta_str = "REMOVED"
            status = "[dim]REMOVED[/dim]"

        table.add_row(case_id, b_score, c_score, delta_str, status)

    console.print(table)

    summary = Table(title="Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Baseline Pass Rate", f"{base.pass_rate:.1%}")
    summary.add_row("Current Pass Rate", f"{curr.pass_rate:.1%}")
    summary.add_row(
        "Delta", f"{(curr.pass_rate - base.pass_rate):+.1%}"
    )
    summary.add_row("Regressions", f"[red]{regressions}[/red]")
    summary.add_row("Improvements", f"[green]{improvements}[/green]")
    console.print(summary)

    if regressions > 0:
        raise typer.Exit(code=1)


@app.command()
def report(
    results_dir: Path = typer.Argument(..., help="Directory containing result JSONL files"),
    output: Path = typer.Option("report.html", "--output", "-o", help="Output HTML path"),
) -> None:
    """Generate an HTML report from result files."""
    from evalkit.reporter import generate_report

    generate_report(results_dir, output)
    console.print(f"[green]Report generated:[/green] {output}")


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"evalkit {__version__}")


if __name__ == "__main__":
    app()
