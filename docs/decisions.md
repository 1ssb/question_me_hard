# Decisions

## 2026-04-08

- Started from a clean Git history at user request and rebuilt the repository from scratch.
- Chose a symbolic grid-world environment first to isolate uncertainty-aware reasoning from simulator complexity.
- Used Pydantic schemas for strict validation of agent outputs, traces, metrics, and task definitions.
- Kept the runtime dependency set minimal: `pydantic` and `PyYAML` only, with the Ollama adapter implemented over the Python standard library.
- Designed the agent loop as a bounded stepwise interaction so query behavior, tool usage, and abstention remain explicit evaluation targets.
- Added non-LLM baselines immediately so the project has a sanity floor before any model benchmarking.
- Probed the local Ollama environment before integrating evaluation. The `ollama` executable is present, but on 2026-04-08 there was no running daemon at `http://127.0.0.1:11434` and no locally discoverable models.
- Generated deterministic dataset manifests under `manifests/generated/datasets/` during the first smoke/eval runs instead of checking in large static split files by hand.
