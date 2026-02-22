"""YAML/JSONL loader for eval suites and results."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from evalkit.models import EvalResult, EvalSuite, TestResult


def load_suite(path: str | Path) -> EvalSuite:
    """Load an eval suite from a YAML file."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return EvalSuite(**data)


def save_results(results: EvalResult, path: str | Path) -> None:
    """Save eval results as JSONL (one TestResult per line, header first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        # First line: suite-level metadata
        header = {
            "schema_version": results.schema_version,
            "suite_name": results.suite_name,
            "model": results.model,
            "timestamp": results.timestamp,
            "pass_rate": results.pass_rate,
            "passed": results.passed,
            "total_cases": results.total_cases,
            "passed_cases": results.passed_cases,
            "failed_cases": results.failed_cases,
            "mean_score": results.mean_score,
            "total_latency_ms": results.total_latency_ms,
            "total_tokens": results.total_tokens,
        }
        f.write(json.dumps(header) + "\n")
        # Subsequent lines: per-case results
        for result in results.results:
            f.write(result.model_dump_json() + "\n")


def load_results(path: str | Path) -> EvalResult:
    """Load eval results from a JSONL file."""
    path = Path(path)
    lines = path.read_text().strip().split("\n")
    header = json.loads(lines[0])
    test_results = [TestResult(**json.loads(line)) for line in lines[1:]]
    return EvalResult(results=test_results, **header)
