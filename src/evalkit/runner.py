"""Async test runner -- orchestrates suite execution end-to-end."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from evalkit.metrics import METRIC_REGISTRY
from evalkit.models import (
    EvalResult,
    MetricConfig,
    SuiteResult,
    TestCase,
    TestSuite,
)
from evalkit.providers import PROVIDER_REGISTRY
from evalkit.providers.base import Provider
from evalkit.threshold import check_thresholds

console = Console()

# Semaphore for rate limiting parallel requests
_DEFAULT_CONCURRENCY = 5


def load_suite(path: str | Path) -> TestSuite:
    """Load and validate a test suite from YAML."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return TestSuite(**data)


async def _run_single_case(
    case: TestCase,
    provider: Provider,
    metrics_config: list[MetricConfig],
    semaphore: asyncio.Semaphore,
) -> EvalResult:
    """Run a single test case: call LLM, run all metrics."""
    async with semaphore:
        # Call the LLM
        response = await provider.generate(case.input)

        # Run all configured metrics
        metric_results = []
        for mc in metrics_config:
            metric_cls = METRIC_REGISTRY.get(mc.type)
            if metric_cls is None:
                continue
            engine = metric_cls()
            result = await engine.score(
                input=case.input,
                output=response,
                expected=case.expected_output,
                params=mc.params,
            )
            metric_results.append(result)

        return EvalResult(
            test_case_id=case.id,
            model_response=response,
            metric_results=metric_results,
            passed=True,  # will be set by threshold engine
        )


async def run_suite_async(
    suite: TestSuite,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> SuiteResult:
    """Run all test cases in the suite with parallel execution."""
    start_time = time.perf_counter()

    # Create the provider
    provider_cls = PROVIDER_REGISTRY.get(suite.model.provider)
    if provider_cls is None:
        raise ValueError(f"Unknown provider: {suite.model.provider}")

    provider = provider_cls(
        model=suite.model.model,
        params=suite.model.params,
    )

    semaphore = asyncio.Semaphore(concurrency)

    console.print(f"\n[bold]Running suite:[/bold] {suite.name}")
    console.print(
        f"[dim]Provider: {suite.model.provider} | "
        f"Model: {suite.model.model} | "
        f"Cases: {len(suite.test_cases)}[/dim]\n"
    )

    # Run all cases in parallel
    tasks = [
        _run_single_case(case, provider, suite.metrics, semaphore)
        for case in suite.test_cases
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Running {len(tasks)} test cases...", total=len(tasks)
        )
        results = await asyncio.gather(*tasks)
        progress.update(task, completed=len(tasks))

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Compute aggregate scores per metric
    aggregate_scores: dict[str, float] = {}
    metric_totals: dict[str, list[float]] = {}
    for result in results:
        for mr in result.metric_results:
            metric_totals.setdefault(mr.metric_name, []).append(mr.score)

    for name, scores in metric_totals.items():
        aggregate_scores[name] = sum(scores) / len(scores) if scores else 0.0

    # Check thresholds
    violations = check_thresholds(aggregate_scores, suite.thresholds)

    # Mark individual results pass/fail based on thresholds
    for result in results:
        case_passed = True
        for mr in result.metric_results:
            threshold = suite.thresholds.get(mr.metric_name)
            if threshold is not None and mr.score < threshold:
                case_passed = False
        result.passed = case_passed

    suite_passed = len(violations) == 0

    return SuiteResult(
        suite_name=suite.name,
        description=suite.description,
        model_config_used=suite.model,
        results=list(results),
        aggregate_scores=aggregate_scores,
        threshold_violations=violations,
        passed=suite_passed,
        total_cases=len(results),
        duration_ms=elapsed_ms,
    )


def run_suite(
    suite_path: str | Path,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> SuiteResult:
    """Synchronous wrapper for run_suite_async."""
    suite = load_suite(suite_path)
    return asyncio.run(run_suite_async(suite, concurrency))
