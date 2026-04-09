from pathlib import Path

from hedgeagent.agents.baselines import UncertaintyThresholdPolicy
from hedgeagent.config.types import EvalConfig, ProjectConfig
from hedgeagent.eval.runner import evaluate_policy
from hedgeagent.schemas.episode import TaskGenerationConfig
from hedgeagent.tasks.generator import generate_dataset_splits


def test_smoke_baseline_eval(tmp_path: Path) -> None:
    dataset = generate_dataset_splits(TaskGenerationConfig(train_size=2, val_size=3, test_size=2, seed=8))
    eval_config = EvalConfig(split="val", limit=3, seed=8, max_steps=4, output_root=str(tmp_path / "results"))
    project_config = ProjectConfig(
        results_dir=str(tmp_path / "results"),
        reports_dir=str(tmp_path / "reports"),
        default_dataset_dir=str(tmp_path / "datasets"),
    )
    run_dir, results = evaluate_policy(
        policy=UncertaintyThresholdPolicy(),
        episodes=dataset["val"],
        eval_config=eval_config,
        project_config=project_config,
        output_dir=tmp_path / "baseline_run",
    )
    assert run_dir.exists()
    assert (run_dir / "aggregate_metrics.json").exists()
    assert (run_dir / "summary.md").exists()
    assert len(results) == 3

