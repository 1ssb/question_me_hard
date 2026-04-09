from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


class CellState(IntEnum):
    UNKNOWN = -1
    FREE = 0
    BLOCKED = 1


class ActionType(StrEnum):
    ACT = "ACT"
    QUERY = "QUERY"
    TOOL = "TOOL"
    ABSTAIN = "ABSTAIN"


class ToolName(StrEnum):
    REVEAL_OBSERVATION = "reveal_observation"
    PLAN_PATH = "plan_path"
    ESTIMATE_UNCERTAINTY = "estimate_uncertainty"
    VERIFY_ACTION = "verify_action"
    SUMMARIZE_STATE = "summarize_state"


class FailureCategory(StrEnum):
    OVERCONFIDENT_ACT = "overconfident_act"
    EXCESSIVE_QUERY = "excessive_query"
    LAZY_ABSTENTION = "lazy_abstention"
    TOOL_MISUSE = "tool_misuse"
    SCHEMA_BREAKAGE = "schema_breakage"
    CONTRADICTORY_ACTION = "contradictory_action"
    PLANNING_ERROR = "planning_error"
    IGNORED_AVAILABLE_INFORMATION = "ignored_available_information"
    TIMEOUT = "timeout"

