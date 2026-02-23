"""Rubric metric -- multi-criteria weighted rubric evaluation via LLM."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from evalkit.metrics.base import Metric
from evalkit.models import MetricResult

_RUBRIC_SYSTEM = """You are an expert evaluator. You will score a response on a specific criterion.

Return ONLY valid JSON with this schema:
{
  "justification": "your reasoning for the score",
  "score": <integer 1-5>
}"""

_RUBRIC_PROMPT = """## Original Prompt
{input}

## LLM Response
{output}

## Expected Output (reference)
{expected}

## Criterion
**{criterion}**: {description}

Score the response on this criterion from 1 (does not meet) to 5 (fully meets). Provide justification."""


async def _score_criterion(
    client: httpx.AsyncClient,
    input_text: str,
    output: str,
    expected: str,
    criterion: str,
    description: str,
    model: str,
    provider: str,
) -> dict[str, Any]:
    """Score a single rubric criterion."""
    api_key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")

    prompt = _RUBRIC_PROMPT.format(
        input=input_text,
        output=output,
        expected=expected or "(none provided)",
        criterion=criterion,
        description=description,
    )

    try:
        if provider == "anthropic":
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 512,
                    "temperature": 0,
                    "system": _RUBRIC_SYSTEM,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
        else:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 512,
                    "messages": [
                        {"role": "system", "content": _RUBRIC_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]

        result = json.loads(text)
        return {
            "criterion": criterion,
            "score": max(1, min(5, int(result.get("score", 3)))),
            "justification": result.get("justification", ""),
        }
    except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError):
        # Fallback
        try:
            match = re.search(r"\b([1-5])\b", text)  # type: ignore[possibly-undefined]
            score = int(match.group(1)) if match else 3
        except Exception:
            score = 3
        return {
            "criterion": criterion,
            "score": score,
            "justification": "parse error",
        }


class RubricMetric(Metric):
    """Multi-criteria rubric evaluation. Each criterion scored 1-5 by LLM,
    weighted average normalized to 0-1."""

    async def score(
        self,
        input: str,
        output: str,
        expected: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        rubric_items = params.get("rubric", [])
        model = params.get("model", "claude-haiku-4-5-20251001")

        if not rubric_items:
            return MetricResult(
                metric_name="rubric",
                score=0.0,
                details={"error": "no rubric criteria provided"},
            )

        # Determine provider from model name
        if "claude" in model:
            provider = "anthropic"
        else:
            provider = "openai"

        criterion_results = []
        async with httpx.AsyncClient() as client:
            for item in rubric_items:
                result = await _score_criterion(
                    client=client,
                    input_text=input,
                    output=output,
                    expected=expected or "",
                    criterion=item.get("criterion", ""),
                    description=item.get("description", ""),
                    model=model,
                    provider=provider,
                )
                result["weight"] = item.get("weight", 1.0)
                criterion_results.append(result)

        # Weighted average
        total_weight = sum(r["weight"] for r in criterion_results)
        if total_weight > 0:
            weighted_sum = sum(
                ((r["score"] - 1) / 4) * r["weight"] for r in criterion_results
            )
            normalized = weighted_sum / total_weight
        else:
            normalized = 0.0

        return MetricResult(
            metric_name="rubric",
            score=normalized,
            details={
                "criterion_results": criterion_results,
                "model": model,
            },
        )
