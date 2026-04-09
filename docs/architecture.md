# Architecture

## Overview

HedgeAgent is organized as a deterministic symbolic research stack with a strict separation between environment dynamics, tools, model backends, agent logic, and evaluation.

## Core flow

1. A dataset generator emits typed `EpisodeSpec` objects.
2. A `GridWorld` instance holds hidden state and the mutable observed state.
3. A policy or LLM agent receives a summarized visible state and chooses `ACT`, `QUERY`, `TOOL`, or `ABSTAIN`.
4. The runner executes the chosen step, validates outputs, records traces, and repeats until termination or max steps.
5. Evaluation aggregates episode outcomes, latency, schema failures, tool usage, abstention quality, and slice metrics.

## Main modules

- `schemas/`: typed contracts for all key objects
- `envs/`: symbolic partial-observability spatial environment
- `tools/`: deterministic tool implementations with logging hooks
- `models/`: backend abstraction and Ollama adapter
- `agents/`: LLM-backed and baseline policies
- `eval/`: episode runner, artifact management, reports, and model manifest handling
- `metrics/`: aggregate and sliced metrics computation
- `logging/`: structured JSONL sinks
- `prompts/`: versioned prompt assets and prompt builder

## Design choices

- Symbolic grid navigation is the first environment because it isolates uncertainty-aware reasoning from simulator integration noise.
- The runner is stepwise rather than single-shot so tool discipline, query behavior, and abstention can be observed directly.
- Tools are deterministic and local to ensure reproducibility and controlled debugging.
- Dataset generation is seeded and manifestable so later training or preference pipelines can reuse identical splits.

