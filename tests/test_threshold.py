"""Tests for the threshold engine."""

from evalkit.models import MetricResult, MetricType, TestResult
from evalkit.threshold import evaluate_suite


def _make_result(case_id: str, passed: bool, score: float = 1.0) -> TestResult:
    return TestResult(
        case_id=case_id,
        input="test",
        actual="test",
        metric_results=[
            MetricResult(
                metric_type=MetricType.EXACT_MATCH,
                score=score,
                passed=passed,
                threshold=1.0,
            )
        ],
        passed=passed,
        weighted_score=score,
    )


def test_all_pass():
    results = [_make_result("c1", True), _make_result("c2", True)]
    eval_result = evaluate_suite("test", "model", results, suite_pass_rate=1.0)
    assert eval_result.passed is True
    assert eval_result.pass_rate == 1.0


def test_partial_fail_below_threshold():
    results = [_make_result("c1", True), _make_result("c2", False, 0.0)]
    eval_result = evaluate_suite("test", "model", results, suite_pass_rate=0.8)
    assert eval_result.passed is False
    assert eval_result.pass_rate == 0.5


def test_partial_fail_above_threshold():
    results = [
        _make_result("c1", True),
        _make_result("c2", True),
        _make_result("c3", False, 0.0),
    ]
    eval_result = evaluate_suite("test", "model", results, suite_pass_rate=0.6)
    assert eval_result.passed is True


def test_empty_suite():
    eval_result = evaluate_suite("test", "model", [], suite_pass_rate=1.0)
    assert eval_result.total_cases == 0
    assert eval_result.pass_rate == 0.0
