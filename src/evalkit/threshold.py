"""Threshold engine -- compare aggregate scores against thresholds."""

from __future__ import annotations

from evalkit.models import ThresholdViolation


def check_thresholds(
    aggregate_scores: dict[str, float],
    thresholds: dict[str, float],
) -> list[ThresholdViolation]:
    """Compare aggregate scores against thresholds.

    If ANY metric falls below its threshold, the suite fails.
    Returns list of violations with metric name, expected, actual.
    """
    violations: list[ThresholdViolation] = []

    for metric_name, threshold in thresholds.items():
        actual = aggregate_scores.get(metric_name, 0.0)
        if actual < threshold:
            violations.append(
                ThresholdViolation(
                    metric_name=metric_name,
                    expected=threshold,
                    actual=actual,
                )
            )

    return violations
