from pathlib import Path

from hedgeagent.agents.llm_agent import LLMPolicy
from hedgeagent.config.types import EvalConfig, ProjectConfig
from hedgeagent.eval.runner import evaluate_policy
from hedgeagent.models.base import BaseLLMClient, ModelResponse
from hedgeagent.schemas.episode import TaskGenerationConfig
from hedgeagent.tasks.generator import generate_dataset_splits


class FakeClient(BaseLLMClient):
    def __init__(self) -> None:
        self.model_name = "fake-local-model"

    def complete(self, prompt: str, system_prompt: str | None = None) -> ModelResponse:
        del prompt, system_prompt
        return ModelResponse(
            model_name=self.model_name,
            text='{"action_type":"ABSTAIN","rationale_brief":"Visible uncertainty remains high.","tool_args":{},"confidence":0.8,"abstain_reason":"test"}',
            latency_ms=1.5,
            raw_request={"prompt": "stub"},
            raw_response={"response": "stub"},
        )


def test_llm_policy_runs_end_to_end(tmp_path: Path) -> None:
    dataset = generate_dataset_splits(TaskGenerationConfig(train_size=1, val_size=2, test_size=1, seed=4))
    eval_config = EvalConfig(split="val", limit=2, seed=4, max_steps=3, output_root=str(tmp_path / "results"))
    project_config = ProjectConfig(
        results_dir=str(tmp_path / "results"),
        reports_dir=str(tmp_path / "reports"),
        default_dataset_dir=str(tmp_path / "datasets"),
    )
    run_dir, results = evaluate_policy(
        policy=LLMPolicy(FakeClient()),
        episodes=dataset["val"],
        eval_config=eval_config,
        project_config=project_config,
        output_dir=tmp_path / "run",
    )
    assert run_dir.exists()
    assert len(results) == 2
    assert all(result.abstained for result in results)

