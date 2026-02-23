"""LLM Judge metric -- G-Eval style scoring via judge LLM."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from evalkit.metrics.base import Metric
from evalkit.models import MetricResult

_JUDGE_SYSTEM = """You are an expert evaluator. You will be given a prompt, an LLM response, and evaluation criteria. Evaluate the response using chain-of-thought reasoning, then assign a score from 1 (terrible) to 5 (excellent).

Return ONLY valid JSON with this schema:
{
  "reasoning": "your step-by-step evaluation",
  "score": <integer 1-5>
}"""

_JUDGE_PROMPT = """## Original Prompt
{input}

## LLM Response
{output}

## Expected Output (reference)
{expected}

## Evaluation Criteria
{criteria}

Evaluate the LLM response against the criteria. Think step by step, then provide a score from 1 to 5."""


async def _call_judge(
    client: httpx.AsyncClient,
    prompt: str,
    system: str,
    model: str,
    provider: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call the judge LLM with retry logic."""
    api_key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")

    for attempt in range(max_retries):
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
                        "max_tokens": 1024,
                        "temperature": 0,
                        "system": system,
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
                        "max_tokens": 1024,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]

            # Parse JSON response
            result = json.loads(text)
            return result

        except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                # Last attempt: try to extract score from raw text
                if "text" in dir():
                    match = re.search(r"\b([1-5])\b", text)  # type: ignore[has-type]
                    score = int(match.group(1)) if match else 3
                    return {"reasoning": text, "score": score}  # type: ignore[has-type]
                return {"reasoning": f"judge call failed: {e}", "score": 3}

    return {"reasoning": "max retries exceeded", "score": 3}


class LLMJudgeMetric(Metric):
    """G-Eval style LLM judge. Scores 1-5, normalized to 0-1."""

    async def score(
        self,
        input: str,
        output: str,
        expected: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> MetricResult:
        params = params or {}
        criteria = params.get("criteria", "helpfulness")
        model = params.get("model", "claude-haiku-4-5-20251001")

        # Determine provider from model name
        if "claude" in model:
            provider = "anthropic"
        else:
            provider = "openai"

        prompt = _JUDGE_PROMPT.format(
            input=input,
            output=output,
            expected=expected or "(none provided)",
            criteria=criteria,
        )

        async with httpx.AsyncClient() as client:
            result = await _call_judge(
                client=client,
                prompt=prompt,
                system=_JUDGE_SYSTEM,
                model=model,
                provider=provider,
            )

        raw_score = max(1, min(5, int(result.get("score", 3))))
        normalized = (raw_score - 1) / 4  # map 1-5 to 0.0-1.0
        reasoning = result.get("reasoning", "")

        return MetricResult(
            metric_name="llm_judge",
            score=normalized,
            details={
                "raw_score": raw_score,
                "normalized_score": normalized,
                "reasoning": reasoning,
                "judge_model": model,
                "criteria": criteria,
            },
        )
