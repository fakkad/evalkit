"""Scorer — applies metrics to test cases and computes weighted scores."""

from __future__ import annotations

from evalkit.metrics.exact_match import ExactMatchMetric
from evalkit.metrics.llm_judge import LLMJudgeMetric
from evalkit.metrics.rubric import RubricMetric
from evalkit.metrics.semantic_sim import SemanticSimMetric
from evalkit.models import MetricConfig, MetricResult, MetricType, TestResult

_METRIC_ENGINES = {
    MetricType.EXACT_MATCH: ExactMatchMetric(),
    MetricType.SEMANTIC_SIM: SemanticSimMetric(),
    MetricType.LLM_JUDGE: LLMJudgeMetric(),
    MetricType.RUBRIC: RubricMetric(),
}


def score_case(
    case_id: str,
    input_text: str,
    expected: str | None,
    actual: str,
    metrics: list[MetricConfig],
    latency_ms: float = 0.0,
    tokens_used: int = 0,
) -> TestResult:
    """Score a single test case against all configured metrics.

    Uses AND logic: all metrics must pass for the case to pass.
    """
    metric_results: list[MetricResult] = []
    total_weight = sum(m.weight for m in metrics)

    for metric_cfg in metrics:
        engine = _METRIC_ENGINES[metric_cfg.type]
        result = engine.score(
            expected=expected or "",
            actual=actual,
            threshold=metric_cfg.threshold,
            params=metric_cfg.params,
        )
        metric_results.append(result)

    # Weighted score
    if total_weight > 0:
        weighted_score = sum(
            r.score * m.weight for r, m in zip(metric_results, metrics)
        ) / total_weight
    else:
        weighted_score = 0.0

    # AND logic: all metrics must pass
    all_passed = all(r.passed for r in metric_results)

    return TestResult(
        case_id=case_id,
        input=input_text,
        expected=expected,
        actual=actual,
        metric_results=metric_results,
        passed=all_passed,
        weighted_score=weighted_score,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
    )
