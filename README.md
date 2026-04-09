# HedgeAgent

HedgeAgent is a local research scaffold for studying uncertainty-aware, tool-using language agents on symbolic spatial tasks under partial observability. The repository is designed around one-model-at-a-time evaluation with Ollama, reproducible artifacts, explicit schemas, and baseline policies that provide a sanity floor before model-driven benchmarking.

## What is implemented

- Deterministic symbolic grid-world task generation with train/val/test splits
- Strict Pydantic schemas for episodes, tools, agent decisions, traces, and metrics
- Tool layer for observation reveal, path planning, uncertainty estimation, verification, and state summarization
- Robust Ollama probing and request adapter with raw request/response capture
- Stepwise agent loop with bounded retries for structured output
- Baseline policies: always-act, always-query, always-abstain, uncertainty-threshold, oracle, random
- Evaluation harness with JSONL traces, aggregate metrics, slice metrics, summaries, and resumable runs
- Smoke, unit, and integration tests

## Quick start

1. Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

2. Inspect local Ollama availability:

```bash
python scripts/list_ollama_models.py
```

3. Run a smoke test:

```bash
python scripts/run_smoke.py
```

4. Run an evaluation:

```bash
python scripts/run_eval.py --agent uncertainty_threshold --split val --limit 25
python scripts/run_eval.py --agent ollama --model <local-model-name> --split val --limit 25
```

## Repository layout

- `configs/`: project, task, eval, and model defaults
- `src/hedgeagent/`: implementation
- `docs/`: architecture, evaluation protocol, decisions, experiment log
- `scripts/`: entry scripts for probing, smoke tests, and evaluation
- `tests/`: unit, integration, and smoke coverage
- `results/`: timestamped run artifacts
- `reports/`: human-readable deep evaluation reports
- `manifests/`: model status and generated dataset manifests

## Current assumptions

- Python 3.11+ is available
- Ollama is optional for bootstrap; the rest of the project works without it
- Models are evaluated sequentially and tracked in `manifests/model_manifest.json`

## Status

This repository is intentionally built around a symbolic environment first. The immediate goal is pipeline correctness, tool discipline, reproducibility, and evaluation stability before moving to larger models or richer environments.

