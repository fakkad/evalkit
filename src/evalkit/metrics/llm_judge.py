"""LLM Judge metric — G-Eval: CoT evaluation steps from rubric, score 1-10."""

from __future__ import annotations

import json
import os
from typing import Any

from evalkit.models import MetricResult, MetricType

_COT_SYSTEM = """You are an expert evaluator. Given a task description and evaluation criteria, generate detailed chain-of-thought evaluation steps, then score the response.

Return ONLY valid JSON with this schema:
{
  "reasoning": "step-by-step evaluation reasoning",
  "score": <integer 1-10>
}"""

_EVAL_PROMPT = """## Task
{task}

## Evaluation Criteria
{criteria}

## Expected Output
{expected}

## Actual Output
{actual}

Evaluate the actual output against the expected output using the criteria above. Think step by step, then provide a score from 1 (completely wrong) to 10 (perfect)."""


class LLMJudgeMetric:
    """G-Eval style LLM judge. Generates CoT eval steps, scores 1-10, normalizes to 0-1."""

    metric_type = MetricType.LLM_JUDGE

    def score(
        self,
        expected: str,
        actual: str,
        threshold: float = 0.7,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        criteria = params.get(
            "criteria", "Accuracy, completeness, and relevance of the response."
        )
        task = params.get("task", "Evaluate the quality of the LLM response.")
        judge_model = params.get("judge_model", "claude-haiku-4-5-20241022")

        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        prompt = _EVAL_PROMPT.format(
            task=task, criteria=criteria, expected=expected, actual=actual
        )

        response = client.messages.create(
            model=judge_model,
            max_tokens=1024,
            temperature=0,
            system=_COT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        try:
            result = json.loads(text)
            raw_score = int(result["score"])
            reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract score from text
            import re

            match = re.search(r"\b(\d+)\s*/?\s*10\b", text)
            raw_score = int(match.group(1)) if match else 5
            reasoning = text

        raw_score = max(1, min(10, raw_score))
        normalized = (raw_score - 1) / 9  # map 1-10 to 0.0-1.0

        return MetricResult(
            metric_type=self.metric_type,
            score=normalized,
            passed=normalized >= threshold,
            threshold=threshold,
            details={
                "raw_score": raw_score,
                "normalized_score": normalized,
                "reasoning": reasoning,
                "judge_model": judge_model,
            },
        )
