"""Tests for core pydantic models and YAML loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from evalkit.models import (
    EvalResult,
    MetricConfig,
    MetricResult,
    ModelConfig,
    SuiteResult,
    TestCase,
    TestSuite,
    ThresholdViolation,
)
from evalkit.runner import load_suite


def test_test_case_creation():
    case = TestCase(
        id="test-1",
        input="What is 2+2?",
        expected_output="4",
        metadata={"category": "math"},
    )
    assert case.id == "test-1"
    assert case.expected_output == "4"
    assert case.metadata == {"category": "math"}


def test_test_case_optional_expected():
    case = TestCase(id="test-2", input="Hello")
    assert case.expected_output is None
    assert case.metadata == {}


def test_test_suite_defaults():
    suite = TestSuite(
        name="test-suite",
        test_cases=[TestCase(id="c1", input="hi")],
    )
    assert suite.model.provider == "anthropic"
    assert suite.model.model == "claude-sonnet-4-20250514"
    assert suite.metrics == []
    assert suite.thresholds == {}


def test_model_config_custom():
    mc = ModelConfig(
        provider="openai",
        model="gpt-4o",
        params={"temperature": 0.5},
    )
    assert mc.provider == "openai"
    assert mc.params["temperature"] == 0.5


def test_metric_result():
    mr = MetricResult(
        metric_name="exact_match",
        score=0.85,
        details={"exact_match": False},
    )
    assert mr.metric_name == "exact_match"
    assert mr.score == 0.85


def test_eval_result():
    er = EvalResult(
        test_case_id="c1",
        model_response="hello",
        metric_results=[
            MetricResult(metric_name="exact_match", score=1.0)
        ],
        passed=True,
    )
    assert er.test_case_id == "c1"
    assert er.passed is True


def test_suite_result():
    sr = SuiteResult(
        suite_name="test",
        results=[
            EvalResult(test_case_id="c1", passed=True),
        ],
        aggregate_scores={"exact_match": 0.9},
        threshold_violations=[],
        passed=True,
        total_cases=1,
    )
    assert sr.passed is True
    assert sr.aggregate_scores["exact_match"] == 0.9


def test_threshold_violation():
    tv = ThresholdViolation(
        metric_name="semantic_similarity",
        expected=0.8,
        actual=0.65,
    )
    assert tv.expected == 0.8
    assert tv.actual == 0.65


def test_load_suite_yaml(tmp_path):
    suite_data = {
        "name": "yaml-test",
        "description": "Test loading from YAML",
        "model": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "params": {"temperature": 0},
        },
        "test_cases": [
            {
                "id": "c1",
                "input": "hello",
                "expected_output": "world",
                "metadata": {"tag": "greeting"},
            }
        ],
        "metrics": [
            {"type": "exact_match"},
        ],
        "thresholds": {"exact_match": 0.5},
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(suite_data))

    suite = load_suite(path)
    assert suite.name == "yaml-test"
    assert len(suite.test_cases) == 1
    assert suite.test_cases[0].expected_output == "world"
    assert suite.metrics[0].type == "exact_match"
    assert suite.thresholds["exact_match"] == 0.5


def test_load_suite_minimal_yaml(tmp_path):
    suite_data = {
        "name": "minimal",
        "test_cases": [
            {"id": "c1", "input": "hi"},
        ],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(suite_data))

    suite = load_suite(path)
    assert suite.name == "minimal"
    assert suite.model.provider == "anthropic"


def test_suite_result_serialization():
    sr = SuiteResult(
        suite_name="test",
        results=[
            EvalResult(
                test_case_id="c1",
                model_response="ok",
                metric_results=[
                    MetricResult(metric_name="exact_match", score=1.0)
                ],
                passed=True,
            ),
        ],
        aggregate_scores={"exact_match": 1.0},
        threshold_violations=[],
        passed=True,
        total_cases=1,
    )
    json_str = sr.model_dump_json()
    loaded = SuiteResult.model_validate_json(json_str)
    assert loaded.suite_name == "test"
    assert loaded.results[0].test_case_id == "c1"
