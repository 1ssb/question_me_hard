from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class MetricSlice(StrictModel):
    name: str
    count: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class AggregateMetrics(StrictModel):
    total_episodes: int = Field(ge=0)
    success_rate: float = 0.0
    unsafe_action_rate: float = 0.0
    abstention_rate: float = 0.0
    correct_abstention_rate: float = 0.0
    unnecessary_query_rate: float = 0.0
    average_observation_budget_used: float = 0.0
    average_tool_calls: float = 0.0
    schema_valid_output_rate: float = 0.0
    latency_per_episode_ms: float = 0.0
    latency_per_model_call_ms: float = 0.0
    timeout_rate: float = 0.0
    tool_failure_rate: float = 0.0
    slices: list[MetricSlice] = Field(default_factory=list)
    failure_counts: dict[str, int] = Field(default_factory=dict)


class RunManifest(StrictModel):
    run_id: str
    timestamp_utc: str
    agent_name: str
    model_name: str | None = None
    split: str
    seed: int
    limit: int
    max_steps: int
    git_commit: str | None = None
    config_snapshot_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelManifestEntry(StrictModel):
    model_name: str
    status: str
    discovered_at: str
    smoke_tested_at: str | None = None
    schema_validated_at: str | None = None
    pilot_evaluated_at: str | None = None
    full_evaluated_at: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

