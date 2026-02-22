"""Exact match metric — string/normalized comparison."""

from __future__ import annotations

import re
from typing import Any

from evalkit.models import MetricResult, MetricType


class ExactMatchMetric:
    """Compare actual output to expected via exact or normalized string match."""

    metric_type = MetricType.EXACT_MATCH

    def score(
        self,
        expected: str,
        actual: str,
        threshold: float = 1.0,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        normalize = params.get("normalize", True)
        ignore_case = params.get("ignore_case", True)
        ignore_punctuation = params.get("ignore_punctuation", False)

        e = expected
        a = actual

        if normalize:
            e = " ".join(e.split())
            a = " ".join(a.split())

        if ignore_case:
            e = e.lower()
            a = a.lower()

        if ignore_punctuation:
            e = re.sub(r"[^\w\s]", "", e)
            a = re.sub(r"[^\w\s]", "", a)

        match = e == a
        score_val = 1.0 if match else 0.0

        return MetricResult(
            metric_type=self.metric_type,
            score=score_val,
            passed=score_val >= threshold,
            threshold=threshold,
            details={
                "expected_normalized": e,
                "actual_normalized": a,
                "exact_match": match,
            },
        )
