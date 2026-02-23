"""Exact match metric -- normalized string comparison."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from evalkit.metrics.base import Metric
from evalkit.models import MetricResult


class ExactMatchMetric(Metric):
    """Compare output to expected via exact or normalized string match.

    Supports lowercase normalization, whitespace stripping, punctuation
    removal, and optional fuzzy matching via SequenceMatcher.
    """

    async def score(
        self,
        input: str,
        output: str,
        expected: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        normalize = params.get("normalize", True)
        ignore_case = params.get("ignore_case", True)
        ignore_punctuation = params.get("ignore_punctuation", False)
        fuzzy_threshold = params.get("fuzzy_threshold", None)

        if expected is None:
            return MetricResult(
                metric_name="exact_match",
                score=0.0,
                details={"error": "no expected_output provided"},
            )

        e = expected
        a = output

        if normalize:
            e = " ".join(e.split())
            a = " ".join(a.split())

        if ignore_case:
            e = e.lower()
            a = a.lower()

        if ignore_punctuation:
            e = re.sub(r"[^\w\s]", "", e)
            a = re.sub(r"[^\w\s]", "", a)

        if e == a:
            score_val = 1.0
        elif fuzzy_threshold is not None:
            score_val = SequenceMatcher(None, e, a).ratio()
        else:
            score_val = 1.0 if e == a else 0.0

        return MetricResult(
            metric_name="exact_match",
            score=score_val,
            details={
                "expected_normalized": e,
                "actual_normalized": a,
                "exact_match": e == a,
            },
        )
