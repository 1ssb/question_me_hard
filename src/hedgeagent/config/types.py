from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = "hedgeagent"
    results_dir: str = "results"
    reports_dir: str = "reports"
    manifests_dir: str = "manifests"
    default_seed: int = 7
    default_dataset_dir: str = "manifests/generated/datasets"
    default_prompt_version: str = "decision_prompt_v1.txt"


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: str = "val"
    limit: int = 25
    seed: int = 7
    max_steps: int = 6
    timeout_seconds: float = 20.0
    resume: bool = True
    write_model_calls: bool = True
    output_root: str = "results"
    slice_thresholds: dict[str, float] = Field(default_factory=dict)


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.1
    max_tokens: int = 400
    seed: int = 7
    timeout_seconds: float = 30.0
    max_retries: int = 2
    prompt_mode: str = "generate"
    preferred_model_substrings: list[str] = Field(default_factory=list)

    def endpoint(self) -> str:
        return self.base_url.rstrip("/")

    def results_path(self, project: ProjectConfig) -> Path:
        return Path(project.results_dir)

