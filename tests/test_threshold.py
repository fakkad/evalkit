"""Tests for the threshold engine."""

from evalkit.threshold import check_thresholds


def test_no_violations():
    scores = {"exact_match": 0.9, "semantic_similarity": 0.85}
    thresholds = {"exact_match": 0.5, "semantic_similarity": 0.8}
    violations = check_thresholds(scores, thresholds)
    assert len(violations) == 0


def test_single_violation():
    scores = {"exact_match": 0.3, "semantic_similarity": 0.85}
    thresholds = {"exact_match": 0.5, "semantic_similarity": 0.8}
    violations = check_thresholds(scores, thresholds)
    assert len(violations) == 1
    assert violations[0].metric_name == "exact_match"
    assert violations[0].expected == 0.5
    assert violations[0].actual == 0.3


def test_multiple_violations():
    scores = {"exact_match": 0.3, "semantic_similarity": 0.5}
    thresholds = {"exact_match": 0.5, "semantic_similarity": 0.8}
    violations = check_thresholds(scores, thresholds)
    assert len(violations) == 2


def test_exact_threshold():
    scores = {"exact_match": 0.5}
    thresholds = {"exact_match": 0.5}
    violations = check_thresholds(scores, thresholds)
    assert len(violations) == 0


def test_missing_metric_in_scores():
    scores = {}
    thresholds = {"exact_match": 0.5}
    violations = check_thresholds(scores, thresholds)
    assert len(violations) == 1
    assert violations[0].actual == 0.0


def test_empty_thresholds():
    scores = {"exact_match": 0.1}
    thresholds = {}
    violations = check_thresholds(scores, thresholds)
    assert len(violations) == 0


def test_threshold_violation_details():
    scores = {"llm_judge": 0.65}
    thresholds = {"llm_judge": 0.7}
    violations = check_thresholds(scores, thresholds)
    assert len(violations) == 1
    v = violations[0]
    assert v.metric_name == "llm_judge"
    assert v.expected == 0.7
    assert v.actual == 0.65
