"""Threshold engine — determines suite pass/fail and exit codes."""

from __future__ import annotations

from evalkit.models import EvalResult, TestResult


def evaluate_suite(
    suite_name: str,
    model: str,
    results: list[TestResult],
    suite_pass_rate: float = 1.0,
) -> EvalResult:
    """Compute suite-level pass/fail from individual case results.

    Exit code semantics:
      0 = all passed
      1 = suite failed (pass rate below threshold)
      2 = error during evaluation
    """
    total = len(results)
    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = total - passed_cases
    pass_rate = passed_cases / total if total > 0 else 0.0
    mean_score = sum(r.weighted_score for r in results) / total if total > 0 else 0.0

    return EvalResult(
        suite_name=suite_name,
        model=model,
        results=results,
        pass_rate=pass_rate,
        passed=pass_rate >= suite_pass_rate,
        total_cases=total,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        mean_score=mean_score,
        total_latency_ms=sum(r.latency_ms for r in results),
        total_tokens=sum(r.tokens_used for r in results),
    )
