# Experiment Log

## Template

### Run

- Timestamp:
- Agent:
- Model:
- Split:
- Limit:
- Seed:
- Configs:
- Output directory:

### Outcome

- Status:
- Key metrics:
- Notable failures:
- Prompt or parser issues:
- Tool-use observations:
- Next changes:

## 2026-04-08 Baseline Smoke

### Run

- Timestamp: 2026-04-08T08:40:50Z
- Agent: uncertainty_threshold
- Model: n/a
- Split: val
- Limit: 5
- Seed: 7
- Configs: configs/project.yaml, configs/eval/default.yaml, configs/tasks/default.yaml
- Output directory: results/20260408T084050Z__uncertainty_threshold__val

### Outcome

- Status: completed
- Key metrics: success_rate=1.000, unsafe_action_rate=0.000, average_tool_calls=1.400
- Notable failures: none
- Prompt or parser issues: none, non-LLM baseline
- Tool-use observations: moderate reveal usage before acting
- Next changes: scale to a larger baseline slice and probe Ollama again only after a daemon is running

## 2026-04-08 Baseline Pilot

### Run

- Timestamp: 2026-04-08T08:40:54Z
- Agent: uncertainty_threshold
- Model: n/a
- Split: val
- Limit: 25
- Seed: 7
- Configs: configs/project.yaml, configs/eval/default.yaml, configs/tasks/default.yaml
- Output directory: results/20260408T084054Z__uncertainty_threshold__val

### Outcome

- Status: completed
- Key metrics: success_rate=0.840, unsafe_action_rate=0.040, abstention_rate=0.120, unnecessary_query_rate=0.040
- Notable failures: 1 excessive_query, 1 overconfident_act
- Prompt or parser issues: none, non-LLM baseline
- Tool-use observations: low-budget tasks are materially weaker than high-budget tasks
- Next changes: tighten threshold policy, then evaluate the first local Ollama model once a daemon and model are available

## 2026-04-08 Ollama Probe

### Run

- Timestamp: 2026-04-08T08:40:47Z
- Agent: probe
- Model: none selected
- Split: n/a
- Limit: n/a
- Seed: n/a
- Configs: configs/models/ollama_default.yaml
- Output directory: manifests/ollama_probe.json

### Outcome

- Status: completed_with_blocker
- Key metrics: executable_exists=true, daemon_running=false, discovered_models=0
- Notable failures: no server at http://127.0.0.1:11434
- Prompt or parser issues: n/a
- Tool-use observations: n/a
- Next changes: start `ollama serve`, ensure at least one instruct/chat model is installed, rerun `python3 scripts/list_ollama_models.py`, then run `python3 scripts/run_smoke.py`

## 2026-04-08 Baseline Comparison Pilot

### Run

- Timestamp: 2026-04-08T08:43:22Z
- Agent: baseline_suite
- Model: n/a
- Split: val
- Limit: 25 per baseline
- Seed: 7
- Configs: configs/project.yaml, configs/eval/default.yaml, configs/tasks/default.yaml
- Output directory: reports/20260408T084322Z__baseline_pilot_comparison.md

### Outcome

- Status: completed
- Key metrics: oracle=1.00 success, uncertainty_threshold=0.84 success with 0.04 unsafe, always_query=0.88 success with 0.64 unnecessary_query
- Notable failures: always_act overconfidence, always_query wasteful querying, always_abstain lazy abstention
- Prompt or parser issues: none, non-LLM baseline suite
- Tool-use observations: threshold policy is currently the strongest non-oracle tradeoff
- Next changes: start a local Ollama daemon, install one instruct/chat model, then run the same pilot protocol for `--agent ollama`
