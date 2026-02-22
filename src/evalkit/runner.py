"""Runner — orchestrates suite execution end-to-end."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from evalkit.executor import call_model
from evalkit.loader import load_suite, save_results
from evalkit.models import EvalResult, EvalSuite, TestResult
from evalkit.scorer import score_case
from evalkit.threshold import evaluate_suite

console = Console()


def run_suite(
    suite_path: str | Path,
    model_override: str | None = None,
    output_path: str | Path | None = None,
) -> EvalResult:
    """Run an entire eval suite and return results."""
    suite = load_suite(suite_path)
    model = model_override or suite.model

    console.print(
        f"\n[bold]Running suite:[/bold] {suite.name} ({len(suite.cases)} cases)"
    )
    console.print(f"[dim]Model: {model}[/dim]\n")

    results: list[TestResult] = []

    for i, case in enumerate(suite.cases, 1):
        console.print(f"  [{i}/{len(suite.cases)}] {case.id}...", end=" ")

        system_prompt = case.system_prompt or suite.default_system_prompt

        # Call the target model
        response = call_model(
            input_text=case.input,
            model=model,
            system_prompt=system_prompt,
        )

        # Score against all metrics
        result = score_case(
            case_id=case.id,
            input_text=case.input,
            expected=case.expected,
            actual=response["text"],
            metrics=case.metrics,
            latency_ms=response["latency_ms"],
            tokens_used=response["tokens_used"],
        )

        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"{status} (score: {result.weighted_score:.2f})")
        results.append(result)

    eval_result = evaluate_suite(
        suite_name=suite.name,
        model=model,
        results=results,
        suite_pass_rate=suite.suite_pass_rate,
    )

    _print_summary(eval_result)

    if output_path:
        save_results(eval_result, output_path)
        console.print(f"\n[dim]Results saved to {output_path}[/dim]")

    return eval_result


def _print_summary(result: EvalResult) -> None:
    """Print a summary table of the eval run."""
    console.print()
    table = Table(title="Eval Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
    table.add_row("Suite", result.suite_name)
    table.add_row("Model", result.model)
    table.add_row("Status", status)
    table.add_row("Pass Rate", f"{result.pass_rate:.1%}")
    table.add_row("Cases", f"{result.passed_cases}/{result.total_cases}")
    table.add_row("Mean Score", f"{result.mean_score:.3f}")
    table.add_row("Total Latency", f"{result.total_latency_ms:.0f}ms")
    table.add_row("Total Tokens", str(result.total_tokens))

    console.print(table)
