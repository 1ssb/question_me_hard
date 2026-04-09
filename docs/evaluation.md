# Evaluation

## Evaluation goals

The evaluation harness measures task success, safety, abstention quality, budget use, tool discipline, latency, and schema compliance. The project is structured to stabilize one model at a time before expanding the benchmark set.

## Default protocol

1. Probe Ollama and record discovered models.
2. Select the first viable local instruct or chat model by manifest policy.
3. Run connectivity and structured output smoke tests.
4. Run a small pilot evaluation.
5. Inspect failures and iterate on prompts, parser behavior, and tool protocol.
6. Run a larger evaluation only after the pilot is stable.

## Required artifacts per run

- `episodes.jsonl`
- `aggregate_metrics.json`
- `summary.md`
- `errors.log`
- `model_config_snapshot.json`
- `run_config_snapshot.json`
- `run_manifest.json`
- `model_calls.jsonl` when an LLM backend is used

## Required metric families

- Outcome: success, unsafe action rate, abstention rate, correct abstention rate
- Efficiency: observation budget used, tool calls, latency
- Reliability: schema-valid output rate, timeout rate, tool failure rate
- Slices: low observation, high uncertainty, task difficulty, budget level, task type

## Failure categories

- overconfident_act
- excessive_query
- lazy_abstention
- tool_misuse
- schema_breakage
- contradictory_action
- planning_error
- ignored_available_information
- timeout

