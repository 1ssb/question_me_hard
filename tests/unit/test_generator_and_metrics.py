from hedgeagent.metrics.aggregate import compute_aggregate_metrics
from hedgeagent.schemas.agent import EpisodeResult
from hedgeagent.schemas.common import FailureCategory
from hedgeagent.schemas.episode import TaskGenerationConfig
from hedgeagent.tasks.generator import generate_dataset_splits


def test_dataset_generation_produces_splits() -> None:
    config = TaskGenerationConfig(train_size=3, val_size=2, test_size=2, seed=11)
    dataset = generate_dataset_splits(config)
    assert len(dataset["train"]) == 3
    assert len(dataset["val"]) == 2
    assert len(dataset["test"]) == 2


def test_aggregate_metrics_counts_failures() -> None:
    results = [
        EpisodeResult(
            task_id="a",
            agent_name="baseline",
            success=True,
            unsafe_action=False,
            abstained=False,
            correct_abstention=False,
            unnecessary_query=False,
            observation_budget_used=1,
            tool_calls=1,
            schema_valid_output=True,
            latency_episode_ms=10.0,
            latency_model_ms=0.0,
            timeout=False,
            tool_failure=False,
            failure_category=None,
            final_verification={"success": True},
            raw_outcome={
                "initial_budget": 1,
                "initial_unknown_fraction": 0.5,
                "difficulty": "easy",
                "budget_level": "low",
                "task_type": "navigation",
            },
        ),
        EpisodeResult(
            task_id="b",
            agent_name="baseline",
            success=False,
            unsafe_action=True,
            abstained=False,
            correct_abstention=False,
            unnecessary_query=True,
            observation_budget_used=2,
            tool_calls=2,
            schema_valid_output=True,
            latency_episode_ms=12.0,
            latency_model_ms=0.0,
            timeout=False,
            tool_failure=False,
            failure_category=FailureCategory.OVERCONFIDENT_ACT,
            final_verification={"success": False},
            raw_outcome={
                "initial_budget": 3,
                "initial_unknown_fraction": 0.6,
                "difficulty": "hard",
                "budget_level": "high",
                "task_type": "navigation",
            },
        ),
    ]
    aggregate = compute_aggregate_metrics(
        results,
        thresholds={"low_observation_max_budget": 2, "high_uncertainty_min_fraction": 0.35},
    )
    assert aggregate.total_episodes == 2
    assert aggregate.failure_counts["overconfident_act"] == 1
    assert aggregate.success_rate == 0.5

