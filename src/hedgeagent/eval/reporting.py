from __future__ import annotations

from pathlib import Path

from hedgeagent.schemas.agent import EpisodeResult
from hedgeagent.schemas.metrics import AggregateMetrics
from hedgeagent.utils.files import write_text


def build_summary_markdown(
    *,
    run_id: str,
    agent_name: str,
    model_name: str | None,
    aggregate: AggregateMetrics,
    results: list[EpisodeResult],
    output_dir: Path,
) -> str:
    representative_success = next((result for result in results if result.success), None)
    representative_failure = next(
        (result for result in results if not result.success and not result.correct_abstention),
        None,
    )
    representative_abstention = next((result for result in results if result.correct_abstention), None)
    if representative_failure is None:
        representative_failure = next((result for result in results if not result.success), None)
    lines = [
        f"# Evaluation Summary: {run_id}",
        "",
        f"- Agent: `{agent_name}`",
        f"- Model: `{model_name or 'n/a'}`",
        f"- Output dir: `{output_dir}`",
        f"- Episodes: `{aggregate.total_episodes}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- success_rate: `{aggregate.success_rate:.3f}`",
        f"- unsafe_action_rate: `{aggregate.unsafe_action_rate:.3f}`",
        f"- abstention_rate: `{aggregate.abstention_rate:.3f}`",
        f"- correct_abstention_rate: `{aggregate.correct_abstention_rate:.3f}`",
        f"- unnecessary_query_rate: `{aggregate.unnecessary_query_rate:.3f}`",
        f"- average_observation_budget_used: `{aggregate.average_observation_budget_used:.3f}`",
        f"- average_tool_calls: `{aggregate.average_tool_calls:.3f}`",
        f"- schema_valid_output_rate: `{aggregate.schema_valid_output_rate:.3f}`",
        f"- latency_per_episode_ms: `{aggregate.latency_per_episode_ms:.2f}`",
        f"- latency_per_model_call_ms: `{aggregate.latency_per_model_call_ms:.2f}`",
        f"- timeout_rate: `{aggregate.timeout_rate:.3f}`",
        f"- tool_failure_rate: `{aggregate.tool_failure_rate:.3f}`",
        "",
        "## Failure Counts",
        "",
    ]
    if aggregate.failure_counts:
        for name, count in aggregate.failure_counts.items():
            lines.append(f"- {name}: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Slice Metrics", ""])
    for metric_slice in aggregate.slices:
        lines.append(f"- {metric_slice.name}: count=`{metric_slice.count}` metrics=`{metric_slice.metrics}`")
    lines.extend(["", "## Representative Cases", ""])
    if representative_success:
        lines.append(
            f"- success_case: `{representative_success.task_id}` outcome=`{representative_success.final_verification}`"
        )
    if representative_failure:
        lines.append(
            f"- failure_case: `{representative_failure.task_id}` category=`{representative_failure.failure_category}` verification=`{representative_failure.final_verification}`"
        )
    if representative_abstention:
        lines.append(
            f"- abstention_case: `{representative_abstention.task_id}` correct_abstention=`{representative_abstention.correct_abstention}` verification=`{representative_abstention.final_verification}`"
        )
    return "\n".join(lines) + "\n"


def write_deep_report(path: str | Path, markdown: str) -> Path:
    return write_text(path, markdown)
