# EvalKit

CLI-first LLM evaluation framework. Define test suites in YAML, score outputs with 4 metric types, enforce pass/fail thresholds in CI.

## Why

Existing eval frameworks (DeepEval, promptfoo) are platform-coupled or heavyweight. EvalKit is a pip-installable CLI that owns the scoring math and blocks CI on regressions.

## Install

```bash
pip install -e .
```

## Quick Start

### 1. Define a test suite

```yaml
# evals/qa.yaml
schema_version: 1
name: qa-basic
model: claude-sonnet-4-20250514
suite_pass_rate: 0.8

cases:
  - id: capital-france
    input: "What is the capital of France?"
    expected: "Paris"
    metrics:
      - type: exact_match
        threshold: 1.0
```

### 2. Run it

```bash
export ANTHROPIC_API_KEY=your-key
evalkit run evals/qa.yaml --output results/run1.jsonl
```

Exit codes: `0` = passed, `1` = failed, `2` = error.

### 3. Compare runs

```bash
evalkit compare results/baseline.jsonl results/new.jsonl
```

### 4. Generate HTML report

```bash
evalkit report results/ --output report.html
```

## Metrics

| Metric | Description | Params |
|--------|-------------|--------|
| `exact_match` | Normalized string comparison | `normalize`, `ignore_case`, `ignore_punctuation` |
| `semantic_sim` | Cosine similarity via sentence-transformers | `model` (default: all-MiniLM-L6-v2) |
| `llm_judge` | G-Eval: CoT evaluation, score 1-10 | `criteria`, `task`, `judge_model` |
| `rubric` | Binary pass/fail LLM assertion | `assertion`, `input`, `judge_model` |

## Design Decisions

- **AND logic**: All metrics must pass per case (conservative for CI gating)
- **YAML** for test suite definitions, **JSONL** for result logs
- **Temperature=0** for all LLM judge calls (reproducibility)
- **Versioned result schema** (`schema_version: 1`)

## CI Integration

Add to `.github/workflows/eval.yml`:

```yaml
- name: Run evals
  run: evalkit run evals/qa.yaml
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Non-zero exit blocks the PR.

## License

MIT
