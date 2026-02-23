"""Abstract base class for all metric engines."""

from __future__ import annotations

import abc
from typing import Any

from evalkit.models import MetricResult


class Metric(abc.ABC):
    """Base class for metric engines. All scores normalized to 0.0-1.0."""

    @abc.abstractmethod
    async def score(
        self,
        input: str,
        output: str,
        expected: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        """Score a model output.

        Args:
            input: The original prompt.
            output: The model's response.
            expected: The expected/reference output (optional).
            params: Metric-specific parameters from the suite config.

        Returns:
            MetricResult with score in [0.0, 1.0].
        """
        ...
