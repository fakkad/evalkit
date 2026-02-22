"""Rubric metric — binary pass/fail LLM assertion."""

from __future__ import annotations

import json
import os
from typing import Any

from evalkit.models import MetricResult, MetricType

_RUBRIC_SYSTEM = """You are a strict evaluator. Given a rubric assertion, determine if the actual output satisfies it.

Return ONLY valid JSON:
{
  "passed": true/false,
  "reasoning": "brief explanation"
}"""

_RUBRIC_PROMPT = """## Assertion
{assertion}

## Input
{input}

## Expected Output (reference)
{expected}

## Actual Output
{actual}

Does the actual output satisfy the assertion? Be strict."""


class RubricMetric:
    """Binary pass/fail LLM assertion against a rubric statement."""

    metric_type = MetricType.RUBRIC

    def score(
        self,
        expected: str,
        actual: str,
        threshold: float = 1.0,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        assertion = params.get("assertion", "The output is accurate and complete.")
        input_text = params.get("input", "")
        judge_model = params.get("judge_model", "claude-haiku-4-5-20241022")

        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        prompt = _RUBRIC_PROMPT.format(
            assertion=assertion,
            input=input_text,
            expected=expected,
            actual=actual,
        )

        response = client.messages.create(
            model=judge_model,
            max_tokens=512,
            temperature=0,
            system=_RUBRIC_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        try:
            result = json.loads(text)
            passed = bool(result["passed"])
            reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            passed = "true" in text.lower() and "false" not in text.lower()
            reasoning = text

        score_val = 1.0 if passed else 0.0

        return MetricResult(
            metric_type=self.metric_type,
            score=score_val,
            passed=score_val >= threshold,
            threshold=threshold,
            details={
                "assertion": assertion,
                "passed": passed,
                "reasoning": reasoning,
                "judge_model": judge_model,
            },
        )
