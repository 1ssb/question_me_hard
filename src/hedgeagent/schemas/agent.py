from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .common import ActionType, FailureCategory, Point, StrictModel


class FinalAnswer(StrictModel):
    proposed_path: list[Point] = Field(default_factory=list)
    plan_summary: str = ""


class AgentDecision(StrictModel):
    action_type: ActionType
    rationale_brief: str = Field(min_length=1, max_length=400)
    chosen_tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    final_answer: FinalAnswer | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    abstain_reason: str | None = None
    expected_information_gain: float | None = Field(default=None, ge=0.0, le=1.0)
    notes_for_trace: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "AgentDecision":
        if self.action_type == ActionType.ACT and self.final_answer is None:
            raise ValueError("ACT decisions require final_answer.")
        if self.action_type == ActionType.ABSTAIN and not self.abstain_reason:
            raise ValueError("ABSTAIN decisions require abstain_reason.")
        if self.action_type == ActionType.TOOL and not self.chosen_tool:
            raise ValueError("TOOL decisions require chosen_tool.")
        if self.action_type == ActionType.QUERY and "target" not in self.tool_args:
            raise ValueError("QUERY decisions require tool_args.target.")
        return self


class ToolResultEnvelope(StrictModel):
    name: str
    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = Field(ge=0.0)


class ModelCallRecord(StrictModel):
    model_name: str
    prompt: str
    response_text: str
    latency_ms: float = Field(ge=0.0)
    raw_request: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = None
    repaired: bool = False


class TraceStep(StrictModel):
    step_index: int = Field(ge=0)
    decision: AgentDecision | None = None
    tool_result: ToolResultEnvelope | None = None
    model_call: ModelCallRecord | None = None
    state_summary: str = ""
    query_was_unnecessary: bool = False


class EpisodeResult(StrictModel):
    task_id: str
    agent_name: str
    model_name: str | None = None
    success: bool
    unsafe_action: bool
    abstained: bool
    correct_abstention: bool
    unnecessary_query: bool
    observation_budget_used: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    schema_valid_output: bool
    latency_episode_ms: float = Field(ge=0.0)
    latency_model_ms: float = Field(ge=0.0)
    timeout: bool
    tool_failure: bool
    failure_category: FailureCategory | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    final_verification: dict[str, Any] = Field(default_factory=dict)
    raw_outcome: dict[str, Any] = Field(default_factory=dict)

