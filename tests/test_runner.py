"""Tests for the async runner with mocked providers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from evalkit.models import MetricConfig, ModelConfig, TestCase, TestSuite
from evalkit.runner import _run_single_case, load_suite, run_suite_async


@pytest.fixture
def simple_suite():
    return TestSuite(
        name="test-suite",
        description="Test suite for runner tests",
        model=ModelConfig(provider="anthropic", model="test-model"),
        test_cases=[
            TestCase(id="c1", input="What is 2+2?", expected_output="4"),
            TestCase(id="c2", input="Capital of France?", expected_output="Paris"),
        ],
        metrics=[MetricConfig(type="exact_match")],
        thresholds={"exact_match": 0.5},
    )


@pytest.mark.asyncio
async def test_run_single_case_pass():
    """Test a single case that passes exact match."""
    case = TestCase(id="c1", input="What is 2+2?", expected_output="4")
    metrics = [MetricConfig(type="exact_match")]

    mock_provider = AsyncMock()
    mock_provider.generate.return_value = "4"

    semaphore = asyncio.Semaphore(5)
    result = await _run_single_case(case, mock_provider, metrics, semaphore)

    assert result.test_case_id == "c1"
    assert result.model_response == "4"
    assert len(result.metric_results) == 1
    assert result.metric_results[0].score == 1.0


@pytest.mark.asyncio
async def test_run_single_case_fail():
    """Test a single case that fails exact match."""
    case = TestCase(id="c1", input="What is 2+2?", expected_output="4")
    metrics = [MetricConfig(type="exact_match")]

    mock_provider = AsyncMock()
    mock_provider.generate.return_value = "five"

    semaphore = asyncio.Semaphore(5)
    result = await _run_single_case(case, mock_provider, metrics, semaphore)

    assert result.test_case_id == "c1"
    assert result.metric_results[0].score == 0.0


@pytest.mark.asyncio
async def test_run_suite_async_all_pass(simple_suite):
    """Test full suite run where all cases pass."""
    with patch("evalkit.runner.PROVIDER_REGISTRY") as mock_registry:
        mock_provider_instance = MagicMock()
        mock_provider_instance.generate = AsyncMock(side_effect=["4", "Paris"])
        mock_provider_cls = MagicMock(return_value=mock_provider_instance)
        mock_registry.get.return_value = mock_provider_cls

        result = await run_suite_async(simple_suite)

    assert result.suite_name == "test-suite"
    assert result.total_cases == 2
    assert result.passed is True
    assert result.aggregate_scores["exact_match"] == 1.0
    assert len(result.threshold_violations) == 0


@pytest.mark.asyncio
async def test_run_suite_async_threshold_fail(simple_suite):
    """Test suite where threshold is violated."""
    simple_suite.thresholds["exact_match"] = 0.9

    with patch("evalkit.runner.PROVIDER_REGISTRY") as mock_registry:
        mock_provider_instance = MagicMock()
        # First passes, second fails
        mock_provider_instance.generate = AsyncMock(side_effect=["4", "London"])
        mock_provider_cls = MagicMock(return_value=mock_provider_instance)
        mock_registry.get.return_value = mock_provider_cls

        result = await run_suite_async(simple_suite)

    assert result.total_cases == 2
    assert result.aggregate_scores["exact_match"] == 0.5
    assert result.passed is False
    assert len(result.threshold_violations) == 1
    assert result.threshold_violations[0].metric_name == "exact_match"


def test_load_suite_yaml(tmp_path):
    """Test YAML suite loading."""
    suite_data = {
        "name": "runner-test",
        "model": {"provider": "openai", "model": "gpt-4o"},
        "test_cases": [
            {"id": "c1", "input": "hi", "expected_output": "hello"},
        ],
        "metrics": [{"type": "exact_match"}],
        "thresholds": {"exact_match": 0.5},
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(suite_data))

    suite = load_suite(path)
    assert suite.name == "runner-test"
    assert suite.model.provider == "openai"
    assert len(suite.test_cases) == 1


@pytest.mark.asyncio
async def test_unknown_provider():
    """Test that unknown provider raises ValueError."""
    suite = TestSuite(
        name="test",
        model=ModelConfig(provider="unknown", model="x"),
        test_cases=[TestCase(id="c1", input="hi")],
    )
    with pytest.raises(ValueError, match="Unknown provider"):
        await run_suite_async(suite)
