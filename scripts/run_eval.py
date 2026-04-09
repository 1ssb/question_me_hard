from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hedgeagent.agents.baselines import (
    AlwaysAbstainPolicy,
    AlwaysActPolicy,
    AlwaysQueryPolicy,
    OraclePolicy,
    RandomPolicy,
    UncertaintyThresholdPolicy,
)
from hedgeagent.agents.llm_agent import LLMPolicy
from hedgeagent.config.loader import load_model
from hedgeagent.config.types import EvalConfig, OllamaConfig, ProjectConfig
from hedgeagent.eval.model_manifest import record_discovery, update_model_manifest
from hedgeagent.eval.runner import evaluate_policy, load_or_generate_split
from hedgeagent.models.ollama_client import OllamaClient, probe_ollama, select_preferred_model
from hedgeagent.schemas.episode import TaskGenerationConfig


def build_agent(name: str, model_name: str | None, ollama_config: OllamaConfig):
    if name == "always_act":
        return AlwaysActPolicy(), None
    if name == "always_query_until_budget_exhausted":
        return AlwaysQueryPolicy(), None
    if name == "always_abstain":
        return AlwaysAbstainPolicy(), None
    if name == "uncertainty_threshold":
        return UncertaintyThresholdPolicy(), None
    if name == "shortest_path_oracle":
        return OraclePolicy(), None
    if name == "random_policy":
        return RandomPolicy(), None
    if name == "ollama":
        if not model_name:
            raise ValueError("A model name is required for the ollama agent.")
        return LLMPolicy(OllamaClient(model_name=model_name, config=ollama_config)), model_name
    raise ValueError(f"Unknown agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="uncertainty_threshold")
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    project_config = load_model(Path("configs/project.yaml"), ProjectConfig)
    eval_config = load_model(Path("configs/eval/default.yaml"), EvalConfig)
    task_config = load_model(Path("configs/tasks/default.yaml"), TaskGenerationConfig)
    ollama_config = load_model(Path("configs/models/ollama_default.yaml"), OllamaConfig)
    if args.split is not None:
        eval_config.split = args.split
    if args.limit is not None:
        eval_config.limit = args.limit
    if args.seed is not None:
        eval_config.seed = args.seed

    chosen_model = args.model
    if args.agent == "ollama" and not chosen_model:
        probe = probe_ollama(ollama_config)
        record_discovery(Path("manifests/model_manifest.json"), probe.get("models", []))
        chosen_model = select_preferred_model(probe, ollama_config.preferred_model_substrings)
        if not chosen_model:
            raise SystemExit("No local Ollama model available.")

    agent, manifest_model = build_agent(args.agent, chosen_model, ollama_config)
    episodes = load_or_generate_split(split=eval_config.split, task_config=task_config, project_config=project_config)
    run_dir, results = evaluate_policy(
        policy=agent,
        episodes=episodes,
        eval_config=eval_config,
        project_config=project_config,
        model_config=ollama_config if args.agent == "ollama" else None,
        output_dir=args.output_dir,
    )
    if manifest_model:
        update_model_manifest(
            Path("manifests/model_manifest.json"),
            manifest_model,
            status="full_evaluated" if eval_config.limit > 10 else "pilot_evaluated",
            metadata={"last_run_dir": str(run_dir.resolve()), "episodes": len(results)},
        )
    print(run_dir)


if __name__ == "__main__":
    main()
