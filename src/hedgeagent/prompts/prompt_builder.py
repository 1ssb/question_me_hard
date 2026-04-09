from __future__ import annotations

from importlib import resources

from hedgeagent.schemas.agent import ToolResultEnvelope
from hedgeagent.schemas.episode import EpisodeState


SCHEMA_DESCRIPTION = """{
  "action_type": "ACT | QUERY | TOOL | ABSTAIN",
  "rationale_brief": "short visible explanation",
  "chosen_tool": "required only for TOOL",
  "tool_args": {"target": {"x": 0, "y": 0}, "radius": 1},
  "final_answer": {"proposed_path": [{"x": 0, "y": 0}], "plan_summary": "required only for ACT"},
  "confidence": 0.0,
  "abstain_reason": "required only for ABSTAIN",
  "expected_information_gain": 0.0,
  "notes_for_trace": "optional"
}"""


def load_prompt_template(version: str) -> str:
    return resources.files("hedgeagent.prompts").joinpath(version).read_text(encoding="utf-8")


def _format_tool_history(tool_history: list[ToolResultEnvelope]) -> str:
    if not tool_history:
        return "none"
    lines = []
    for result in tool_history[-5:]:
        status = "ok" if result.success else f"error={result.error}"
        lines.append(f"{result.name}: {status} payload={result.payload}")
    return "\n".join(lines)


def _format_state(state: EpisodeState) -> str:
    rows = []
    for y, row in enumerate(state.observed_map):
        chars = []
        for x, cell in enumerate(row):
            if (x, y) == state.start.as_tuple():
                chars.append("S")
            elif (x, y) == state.goal.as_tuple():
                chars.append("G")
            elif cell == -1:
                chars.append("?")
            elif cell == 1:
                chars.append("#")
            else:
                chars.append(".")
        rows.append("".join(chars))
    stats = [
        f"task_id={state.task_id}",
        f"budget_remaining={state.observation_budget_remaining}",
        f"observations_used={state.observations_used}",
        f"difficulty={state.difficulty}",
        f"task_type={state.task_type}",
        f"start=({state.start.x},{state.start.y}) goal=({state.goal.x},{state.goal.y})",
    ]
    return "\n".join(stats + rows)


def build_decision_prompt(
    *,
    state: EpisodeState,
    tool_history: list[ToolResultEnvelope],
    step_index: int,
    max_steps: int,
    version: str,
) -> str:
    template = load_prompt_template(version)
    semantic_hints = "\n".join(state.semantic_hints) if state.semantic_hints else "none"
    return template.format(
        allowed_tools=", ".join(
            [
                "plan_path",
                "estimate_uncertainty",
                "verify_action",
                "summarize_state",
            ]
        ),
        state_summary=_format_state(state),
        tool_history=_format_tool_history(tool_history),
        semantic_hints=semantic_hints,
        schema_description=SCHEMA_DESCRIPTION,
        step_index=step_index,
        max_steps=max_steps,
    )
