"""Tests for core data models and loader."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from evalkit.loader import load_results, load_suite, save_results
from evalkit.models import (
    EvalResult,
    EvalSuite,
    MetricConfig,
    MetricResult,
    MetricType,
    TestCase,
    TestResult,
)


def test_test_case_creation():
    case = TestCase(
        id="test-1",
        input="What is 2+2?",
        expected="4",
        metrics=[MetricConfig(type=MetricType.EXACT_MATCH)],
    )
    assert case.id == "test-1"
    assert case.metrics[0].weight == 1.0
    assert case.metrics[0].threshold == 0.5


def test_eval_suite_defaults():
    suite = EvalSuite(
        name="test-suite",
        cases=[
            TestCase(
                id="c1",
                input="hi",
                expected="hello",
                metrics=[MetricConfig(type=MetricType.EXACT_MATCH)],
            )
        ],
    )
    assert suite.schema_version == 1
    assert suite.suite_pass_rate == 1.0


def test_load_suite_yaml(tmp_path):
    suite_data = {
        "schema_version": 1,
        "name": "test",
        "cases": [
            {
                "id": "c1",
                "input": "hello",
                "expected": "world",
                "metrics": [{"type": "exact_match", "threshold": 1.0}],
            }
        ],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(suite_data))

    suite = load_suite(path)
    assert suite.name == "test"
    assert len(suite.cases) == 1
    assert suite.cases[0].metrics[0].type == MetricType.EXACT_MATCH


def test_save_and_load_results(tmp_path):
    results = EvalResult(
        suite_name="test",
        model="test-model",
        results=[
            TestResult(
                case_id="c1",
                input="hi",
                expected="hello",
                actual="hello",
                metric_results=[
                    MetricResult(
                        metric_type=MetricType.EXACT_MATCH,
                        score=1.0,
                        passed=True,
                        threshold=1.0,
                    )
                ],
                passed=True,
                weighted_score=1.0,
            )
        ],
        pass_rate=1.0,
        passed=True,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        mean_score=1.0,
    )

    path = tmp_path / "results.jsonl"
    save_results(results, path)
    loaded = load_results(path)

    assert loaded.suite_name == "test"
    assert loaded.pass_rate == 1.0
    assert len(loaded.results) == 1
    assert loaded.results[0].passed is True
