from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .common import CellState, Point, StrictModel


class ObservationRecord(StrictModel):
    center: Point
    radius: int = Field(ge=1)
    newly_revealed: int = Field(ge=0)
    budget_after: int = Field(ge=0)
    revealed_points: list[Point] = Field(default_factory=list)


class EpisodeSpec(StrictModel):
    task_id: str
    split: str
    seed: int
    width: int = Field(gt=1)
    height: int = Field(gt=1)
    hidden_map: list[list[int]]
    observed_map: list[list[int]]
    start: Point
    goal: Point
    semantic_hints: list[str] = Field(default_factory=list)
    observation_budget: int = Field(ge=0)
    observation_radius: int = Field(ge=1)
    difficulty: str = "medium"
    task_type: str = "navigation"
    noise_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    success_criteria: str = "Return a collision-free path from start to goal or abstain."

    @model_validator(mode="after")
    def validate_maps(self) -> "EpisodeSpec":
        if len(self.hidden_map) != self.height or len(self.observed_map) != self.height:
            raise ValueError("Map height does not match episode dimensions.")
        for row in self.hidden_map:
            if len(row) != self.width:
                raise ValueError("Hidden map width mismatch.")
            if any(cell not in {CellState.FREE.value, CellState.BLOCKED.value} for cell in row):
                raise ValueError("Hidden map must contain only free/block values.")
        for row in self.observed_map:
            if len(row) != self.width:
                raise ValueError("Observed map width mismatch.")
            if any(cell not in {state.value for state in CellState} for cell in row):
                raise ValueError("Observed map contains invalid cell values.")
        for point in (self.start, self.goal):
            if point.x >= self.width or point.y >= self.height:
                raise ValueError("Start or goal lies outside the map.")
        if self.hidden_map[self.start.y][self.start.x] != CellState.FREE.value:
            raise ValueError("Start must be free in the hidden map.")
        if self.hidden_map[self.goal.y][self.goal.x] != CellState.FREE.value:
            raise ValueError("Goal must be free in the hidden map.")
        return self


class EpisodeState(StrictModel):
    task_id: str
    split: str
    seed: int
    width: int
    height: int
    observed_map: list[list[int]]
    start: Point
    goal: Point
    semantic_hints: list[str] = Field(default_factory=list)
    observation_budget_remaining: int = Field(ge=0)
    observation_radius: int = Field(ge=1)
    observations_used: int = Field(ge=0)
    difficulty: str = "medium"
    task_type: str = "navigation"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskGenerationConfig(StrictModel):
    seed: int = 7
    train_size: int = Field(default=100, ge=1)
    val_size: int = Field(default=40, ge=1)
    test_size: int = Field(default=40, ge=1)
    width: int = Field(default=9, ge=4)
    height: int = Field(default=9, ge=4)
    obstacle_density: float = Field(default=0.18, ge=0.0, le=0.6)
    obstacle_density_jitter: float = Field(default=0.12, ge=0.0, le=0.4)
    observation_budget_min: int = Field(default=1, ge=0)
    observation_budget_max: int = Field(default=4, ge=0)
    observation_radius: int = Field(default=1, ge=1)
    initial_reveal_radius: int = Field(default=1, ge=1)
    semantic_hint_probability: float = Field(default=0.35, ge=0.0, le=1.0)
    noise_probability: float = Field(default=0.0, ge=0.0, le=0.5)
    max_generation_attempts: int = Field(default=250, ge=10)

    @model_validator(mode="after")
    def validate_budget(self) -> "TaskGenerationConfig":
        if self.observation_budget_max < self.observation_budget_min:
            raise ValueError("observation_budget_max must be >= observation_budget_min")
        return self

