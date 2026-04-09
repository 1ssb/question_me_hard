from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hedgeagent.agents.baselines import AlwaysActPolicy, UncertaintyThresholdPolicy
from hedgeagent.agents.llm_agent import LLMPolicy
from hedgeagent.config.loader import load_model
from hedgeagent.config.types import EvalConfig, OllamaConfig, ProjectConfig
from hedgeagent.eval.model_manifest import record_discovery, update_model_manifest
from hedgeagent.eval.runner import evaluate_policy, load_or_generate_split
from hedgeagent.models.ollama_client import OllamaClient, probe_ollama, select_preferred_model
from hedgeagent.schemas.episode import TaskGenerationConfig


def main() -> None:
    project_config = load_model(Path("configs/project.yaml"), ProjectConfig)
    eval_config = load_model(Path("configs/eval/default.yaml"), EvalConfig)
    task_config = load_model(Path("configs/tasks/default.yaml"), TaskGenerationConfig)
    ollama_config = load_model(Path("configs/models/ollama_default.yaml"), OllamaConfig)
    eval_config.split = "val"
    eval_config.limit = 5
    eval_config.max_steps = 4
    episodes = load_or_generate_split(split=eval_config.split, task_config=task_config, project_config=project_config)

    baseline_dir, _ = evaluate_policy(
        policy=UncertaintyThresholdPolicy(),
        episodes=episodes,
        eval_config=eval_config,
        project_config=project_config,
    )
    evaluate_policy(
        policy=AlwaysActPolicy(),
        episodes=episodes,
        eval_config=eval_config,
        project_config=project_config,
    )

    probe = probe_ollama(ollama_config)
    record_discovery(Path("manifests/model_manifest.json"), probe.get("models", []))
    chosen_model = select_preferred_model(probe, ollama_config.preferred_model_substrings)
    print(f"baseline_smoke={baseline_dir}")
    if chosen_model:
        model_dir, results = evaluate_policy(
            policy=LLMPolicy(OllamaClient(model_name=chosen_model, config=ollama_config)),
            episodes=episodes,
            eval_config=eval_config,
            project_config=project_config,
            model_config=ollama_config,
        )
        status = "schema_validated" if all(result.schema_valid_output for result in results) else "smoke_tested"
        update_model_manifest(
            Path("manifests/model_manifest.json"),
            chosen_model,
            status=status,
            metadata={"last_smoke_dir": str(model_dir.resolve()), "episodes": len(results)},
        )
        print(f"ollama_smoke={model_dir}")
    else:
        print("ollama_smoke=skipped_no_local_model")


if __name__ == "__main__":
    main()
