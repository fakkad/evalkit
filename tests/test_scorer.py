"""Tests for the scorer module."""

from evalkit.models import MetricConfig, MetricType
from evalkit.scorer import score_case


def test_score_case_single_metric_pass():
    result = score_case(
        case_id="t1",
        input_text="What is 2+2?",
        expected="4",
        actual="4",
        metrics=[MetricConfig(type=MetricType.EXACT_MATCH, threshold=1.0)],
    )
    assert result.passed is True
    assert result.weighted_score == 1.0


def test_score_case_single_metric_fail():
    result = score_case(
        case_id="t1",
        input_text="What is 2+2?",
        expected="4",
        actual="5",
        metrics=[MetricConfig(type=MetricType.EXACT_MATCH, threshold=1.0)],
    )
    assert result.passed is False
    assert result.weighted_score == 0.0


def test_score_case_and_logic():
    """AND logic: if any metric fails, the case fails."""
    result = score_case(
        case_id="t1",
        input_text="test",
        expected="hello",
        actual="hello",
        metrics=[
            MetricConfig(type=MetricType.EXACT_MATCH, threshold=1.0, weight=1.0),
            MetricConfig(type=MetricType.EXACT_MATCH, threshold=1.0, weight=1.0,
                        params={"ignore_case": False, "normalize": False}),
        ],
    )
    # Both exact match with same input should pass
    assert result.passed is True


def test_score_case_weighted_score():
    result = score_case(
        case_id="t1",
        input_text="test",
        expected="hello",
        actual="goodbye",
        metrics=[
            MetricConfig(type=MetricType.EXACT_MATCH, threshold=1.0, weight=2.0),
            MetricConfig(type=MetricType.EXACT_MATCH, threshold=0.0, weight=1.0),
        ],
    )
    # score=0 for both, weighted = (0*2 + 0*1)/3 = 0
    assert result.weighted_score == 0.0
