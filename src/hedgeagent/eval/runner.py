from __future__ import annotations

import json
from pathlib import Path
import random
import time

from hedgeagent.agents.base import BasePolicy, DecisionContext
from hedgeagent.config.types import EvalConfig, OllamaConfig, ProjectConfig
from hedgeagent.envs.grid import GridWorld
from hedgeagent.eval.reporting import build_summary_markdown, write_deep_report
from hedgeagent.logging.jsonl import JsonlWriter
from hedgeagent.metrics.aggregate import compute_aggregate_metrics
from hedgeagent.schemas.agent import AgentDecision, EpisodeResult, TraceStep
from hedgeagent.schemas.common import ActionType, FailureCategory
from hedgeagent.schemas.episode import EpisodeSpec, EpisodeState, TaskGenerationConfig
from hedgeagent.schemas.metrics import RunManifest
from hedgeagent.tasks.generator import generate_dataset_splits, save_dataset_splits
from hedgeagent.tools.registry import ToolRegistry, build_default_tool_registry
from hedgeagent.utils.files import ensure_dir, write_json, write_text
from hedgeagent.utils.git import get_git_commit_hash
from hedgeagent.utils.time import utc_now_compact, utc_now_iso


def load_or_generate_split(
    *,
    split: str,
    task_config: TaskGenerationConfig,
    project_config: ProjectConfig,
) -> list[EpisodeSpec]:
    dataset_dir = Path(project_config.default_dataset_dir)
    split_path = dataset_dir / f"{split}.jsonl"
    if not split_path.exists():
        dataset = generate_dataset_splits(task_config)
        save_dataset_splits(dataset, dataset_dir)
    episodes: list[EpisodeSpec] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episodes.append(EpisodeSpec.model_validate_json(line))
    return episodes


def _state_summary(state: EpisodeState) -> str:
    unknown = sum(cell == -1 for row in state.observed_map for cell in row)
    return f"task_id={state.task_id} budget={state.observation_budget_remaining} unknown={unknown}"


def _build_run_dir(
    *,
    eval_config: EvalConfig,
    agent_name: str,
    model_name: str | None,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        return ensure_dir(output_dir)
    stamp = utc_now_compact()
    label_parts = [stamp, agent_name]
    if model_name:
        label_parts.append(model_name.replace(":", "_").replace("/", "_"))
    label_parts.append(eval_config.split)
    return ensure_dir(Path(eval_config.output_root) / "__".join(label_parts))


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            completed.add(payload["task_id"])
    return completed


def _failure_from_result(
    *,
    decision: AgentDecision | None,
    verification: dict[str, object],
    schema_valid: bool,
    tool_failure: bool,
    query_count: int,
    unnecessary_query: bool,
    timed_out: bool,
    env: GridWorld,
) -> FailureCategory | None:
    if timed_out:
        return FailureCategory.TIMEOUT
    if not schema_valid:
        return FailureCategory.SCHEMA_BREAKAGE
    if tool_failure:
        return FailureCategory.TOOL_MISUSE
    if decision is None:
        return FailureCategory.CONTRADICTORY_ACTION
    if decision.action_type == ActionType.ABSTAIN:
        if env.guaranteed_path_exists():
            return FailureCategory.LAZY_ABSTENTION
        return None
    if decision.action_type == ActionType.ACT and not bool(verification.get("safe", False)):
        if env.unknown_fraction() > 0.0:
            return FailureCategory.OVERCONFIDENT_ACT
        return FailureCategory.PLANNING_ERROR
    if query_count and unnecessary_query:
        return FailureCategory.EXCESSIVE_QUERY
    return None


def run_episode(
    *,
    policy: BasePolicy,
    spec: EpisodeSpec,
    eval_config: EvalConfig,
    tool_registry: ToolRegistry,
    model_call_writer: JsonlWriter | None = None,
) -> EpisodeResult:
    env = GridWorld(spec)
    trace: list[TraceStep] = []
    tool_history = []
    rng = random.Random(eval_config.seed + spec.seed)
    start = time.perf_counter()
    latency_model_ms = 0.0
    tool_failure = False
    schema_valid = True
    final_decision: AgentDecision | None = None
    final_verification: dict[str, object] = {}
    unnecessary_query = False
    query_count = 0
    timed_out = False

    for step_index in range(eval_config.max_steps):
        elapsed = time.perf_counter() - start
        if elapsed > eval_config.timeout_seconds:
            timed_out = True
            break
        state = env.visible_state()
        context = DecisionContext(
            state=state,
            env=env,
            step_index=step_index,
            max_steps=eval_config.max_steps,
            tool_history=tool_history,
            trace=trace,
            rng=rng,
        )
        step_result = policy.decide(context)
        if step_result.model_call is not None:
            latency_model_ms += step_result.model_call.latency_ms
            if model_call_writer is not None:
                model_call_writer.write(
                    {
                        "task_id": spec.task_id,
                        "step_index": step_index,
                        **step_result.model_call.model_dump(mode="json"),
                    }
                )
        schema_valid = schema_valid and step_result.schema_valid
        decision = step_result.decision
        step_trace = TraceStep(
            step_index=step_index,
            decision=decision,
            model_call=step_result.model_call,
            state_summary=_state_summary(state),
        )
        if decision is None:
            trace.append(step_trace)
            break

        if decision.action_type == ActionType.QUERY:
            query_count += 1
            guaranteed_before_query = env.guaranteed_path_exists()
            tool_result = tool_registry.call("reveal_observation", env, decision.tool_args)
            step_trace.tool_result = tool_result
            step_trace.query_was_unnecessary = guaranteed_before_query or (
                tool_result.success and int(tool_result.payload.get("newly_revealed", 0)) == 0
            )
            unnecessary_query = unnecessary_query or step_trace.query_was_unnecessary
            tool_failure = tool_failure or not tool_result.success
            tool_history.append(tool_result)
            trace.append(step_trace)
            continue

        if decision.action_type == ActionType.TOOL:
            tool_result = tool_registry.call(decision.chosen_tool or "", env, decision.tool_args)
            step_trace.tool_result = tool_result
            tool_failure = tool_failure or not tool_result.success
            tool_history.append(tool_result)
            trace.append(step_trace)
            continue

        if decision.action_type == ActionType.ACT:
            final_decision = decision
            final_verification = env.verify_path(decision.final_answer or [])
            trace.append(step_trace)
            break

        if decision.action_type == ActionType.ABSTAIN:
            final_decision = decision
            final_verification = {
                "success": False,
                "safe": True,
                "reached_goal": False,
                "collisions": 0,
                "reason": "abstained",
            }
            trace.append(step_trace)
            break

    if final_decision is None and not timed_out and trace:
        last_decision = trace[-1].decision
        if last_decision and last_decision.action_type not in {ActionType.ACT, ActionType.ABSTAIN}:
            final_decision = last_decision
            final_verification = {
                "success": False,
                "safe": False,
                "reached_goal": False,
                "collisions": 0,
                "reason": "max_steps_exhausted",
            }

    failure_category = _failure_from_result(
        decision=final_decision,
        verification=final_verification,
        schema_valid=schema_valid,
        tool_failure=tool_failure,
        query_count=query_count,
        unnecessary_query=unnecessary_query,
        timed_out=timed_out,
        env=env,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    abstained = final_decision is not None and final_decision.action_type == ActionType.ABSTAIN
    correct_abstention = abstained and not env.guaranteed_path_exists()
    success = bool(final_verification.get("success", False))
    unsafe_action = final_decision is not None and final_decision.action_type == ActionType.ACT and not bool(
        final_verification.get("safe", False)
    )

    return EpisodeResult(
        task_id=spec.task_id,
        agent_name=policy.name,
        model_name=policy.model_name,
        success=success,
        unsafe_action=unsafe_action,
        abstained=abstained,
        correct_abstention=correct_abstention,
        unnecessary_query=unnecessary_query,
        observation_budget_used=env.observations_used,
        tool_calls=len(tool_history),
        schema_valid_output=schema_valid,
        latency_episode_ms=duration_ms,
        latency_model_ms=latency_model_ms,
        timeout=timed_out,
        tool_failure=tool_failure,
        failure_category=failure_category,
        trace=trace,
        final_verification=final_verification,
        raw_outcome={
            "initial_budget": spec.observation_budget,
            "budget_level": spec.metadata.get("budget_level"),
            "difficulty": spec.difficulty,
            "task_type": spec.task_type,
            "initial_unknown_fraction": spec.metadata.get("initial_unknown_fraction", 0.0),
            "final_unknown_fraction": env.unknown_fraction(),
            "query_count": query_count,
        },
    )


def evaluate_policy(
    *,
    policy: BasePolicy,
    episodes: list[EpisodeSpec],
    eval_config: EvalConfig,
    project_config: ProjectConfig,
    model_config: OllamaConfig | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, list[EpisodeResult]]:
    run_dir = _build_run_dir(
        eval_config=eval_config,
        agent_name=policy.name,
        model_name=policy.model_name,
        output_dir=output_dir,
    )
    episode_path = run_dir / "episodes.jsonl"
    error_path = run_dir / "errors.log"
    model_call_path = run_dir / "model_calls.jsonl"
    error_path.touch(exist_ok=True)
    completed = _load_completed_ids(episode_path) if eval_config.resume else set()
    episodes_writer = JsonlWriter(episode_path)
    model_writer = JsonlWriter(model_call_path) if eval_config.write_model_calls else None
    tool_registry = build_default_tool_registry()

    write_json(run_dir / "project_config_snapshot.json", project_config.model_dump(mode="json"))
    write_json(run_dir / "run_config_snapshot.json", eval_config.model_dump(mode="json"))
    if model_config is not None:
        write_json(run_dir / "model_config_snapshot.json", model_config.model_dump(mode="json"))

    results: list[EpisodeResult] = []
    for spec in episodes[: eval_config.limit]:
        if spec.task_id in completed:
            continue
        try:
            result = run_episode(
                policy=policy,
                spec=spec,
                eval_config=eval_config,
                tool_registry=tool_registry,
                model_call_writer=model_writer,
            )
            episodes_writer.write(result)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{spec.task_id}: {exc}\n")

    if not results and episode_path.exists():
        with episode_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    results.append(EpisodeResult.model_validate_json(line))

    aggregate = compute_aggregate_metrics(results, thresholds=eval_config.slice_thresholds)
    write_json(run_dir / "aggregate_metrics.json", aggregate.model_dump(mode="json"))
    run_id = run_dir.name
    summary = build_summary_markdown(
        run_id=run_id,
        agent_name=policy.name,
        model_name=policy.model_name,
        aggregate=aggregate,
        results=results,
        output_dir=run_dir,
    )
    write_text(run_dir / "summary.md", summary)
    report_path = Path(project_config.reports_dir) / f"{run_id}.md"
    write_deep_report(report_path, summary)

    manifest = RunManifest(
        run_id=run_id,
        timestamp_utc=utc_now_iso(),
        agent_name=policy.name,
        model_name=policy.model_name,
        split=eval_config.split,
        seed=eval_config.seed,
        limit=eval_config.limit,
        max_steps=eval_config.max_steps,
        git_commit=get_git_commit_hash(),
        config_snapshot_paths={
            "project": str((run_dir / "project_config_snapshot.json").resolve()),
            "eval": str((run_dir / "run_config_snapshot.json").resolve()),
            "model": str((run_dir / "model_config_snapshot.json").resolve()) if model_config is not None else "",
            "metrics": str((run_dir / "aggregate_metrics.json").resolve()),
            "summary": str((run_dir / "summary.md").resolve()),
        },
    )
    write_json(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))
    return run_dir, results
