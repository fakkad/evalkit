"""Compare command -- side-by-side comparison of two eval runs."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.table import Table

from evalkit import __version__
from evalkit.models import SuiteResult
from evalkit.report import load_json

_TEMPLATES_DIR = Path(__file__).parent / "templates"

console = Console()


def compare_results(
    result1: SuiteResult,
    result2: SuiteResult,
) -> dict:
    """Compare two SuiteResults and return comparison data."""
    # Metric deltas
    all_metrics = set(result1.aggregate_scores.keys()) | set(
        result2.aggregate_scores.keys()
    )
    metric_deltas = {}
    regressions = []

    for metric in sorted(all_metrics):
        score1 = result1.aggregate_scores.get(metric, 0.0)
        score2 = result2.aggregate_scores.get(metric, 0.0)
        delta = score2 - score1
        metric_deltas[metric] = {
            "run1": score1,
            "run2": score2,
            "delta": delta,
            "regressed": delta < -0.01,
        }
        if delta < -0.01:
            regressions.append(metric)

    # Per-case comparison
    cases1 = {r.test_case_id: r for r in result1.results}
    cases2 = {r.test_case_id: r for r in result2.results}
    all_case_ids = sorted(set(cases1.keys()) | set(cases2.keys()))

    case_comparisons = []
    for case_id in all_case_ids:
        c1 = cases1.get(case_id)
        c2 = cases2.get(case_id)
        case_comparisons.append({
            "case_id": case_id,
            "run1_passed": c1.passed if c1 else None,
            "run2_passed": c2.passed if c2 else None,
            "run1_metrics": {mr.metric_name: mr.score for mr in c1.metric_results} if c1 else {},
            "run2_metrics": {mr.metric_name: mr.score for mr in c2.metric_results} if c2 else {},
        })

    return {
        "run1_name": result1.suite_name,
        "run2_name": result2.suite_name,
        "run1_passed": result1.passed,
        "run2_passed": result2.passed,
        "metric_deltas": metric_deltas,
        "regressions": regressions,
        "case_comparisons": case_comparisons,
    }


def print_comparison(result1: SuiteResult, result2: SuiteResult) -> None:
    """Print a rich comparison table to the console."""
    comp = compare_results(result1, result2)

    # Metric summary
    table = Table(title="Metric Comparison")
    table.add_column("Metric", style="bold")
    table.add_column("Run 1", justify="right")
    table.add_column("Run 2", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status")

    for metric, data in comp["metric_deltas"].items():
        delta_str = f"{data['delta']:+.4f}"
        if data["regressed"]:
            status = "[red]REGRESSION[/red]"
        elif data["delta"] > 0.01:
            status = "[green]IMPROVED[/green]"
        else:
            status = "[dim]UNCHANGED[/dim]"

        table.add_row(
            metric,
            f"{data['run1']:.4f}",
            f"{data['run2']:.4f}",
            delta_str,
            status,
        )

    console.print(table)

    # Per-case table
    case_table = Table(title="Per-Case Comparison")
    case_table.add_column("Case ID", style="bold")
    case_table.add_column("Run 1", justify="center")
    case_table.add_column("Run 2", justify="center")
    case_table.add_column("Status")

    for case in comp["case_comparisons"]:
        r1 = (
            "[green]PASS[/green]"
            if case["run1_passed"]
            else "[red]FAIL[/red]"
            if case["run1_passed"] is not None
            else "[dim]N/A[/dim]"
        )
        r2 = (
            "[green]PASS[/green]"
            if case["run2_passed"]
            else "[red]FAIL[/red]"
            if case["run2_passed"] is not None
            else "[dim]N/A[/dim]"
        )

        if case["run1_passed"] and not case["run2_passed"]:
            status = "[red]REGRESSED[/red]"
        elif not case["run1_passed"] and case["run2_passed"]:
            status = "[green]FIXED[/green]"
        else:
            status = "[dim]UNCHANGED[/dim]"

        case_table.add_row(case["case_id"], r1, r2, status)

    console.print(case_table)

    if comp["regressions"]:
        console.print(
            f"\n[red]Regressions detected in: {', '.join(comp['regressions'])}[/red]"
        )


def generate_diff_html(
    result1: SuiteResult,
    result2: SuiteResult,
    output_path: str | Path,
) -> Path:
    """Generate an HTML diff report comparing two runs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comp = compare_results(result1, result2)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    # Render as a comparison report
    html = template.render(
        result=result1,
        comparison=comp,
        version=__version__,
    )
    output_path.write_text(html)
    return output_path
