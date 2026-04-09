from hedgeagent.envs.grid import GridWorld
from hedgeagent.schemas.agent import FinalAnswer
from hedgeagent.schemas.common import Point
from hedgeagent.schemas.episode import EpisodeSpec
from hedgeagent.tools.registry import build_default_tool_registry


def build_episode() -> EpisodeSpec:
    return EpisodeSpec(
        task_id="unit-001",
        split="test",
        seed=1,
        width=5,
        height=5,
        hidden_map=[
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        observed_map=[
            [0, 0, -1, -1, -1],
            [1, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0],
        ],
        start=Point(x=0, y=0),
        goal=Point(x=4, y=4),
        observation_budget=2,
        observation_radius=1,
        metadata={"initial_unknown_fraction": 0.72, "budget_level": "low"},
    )


def test_grid_planner_and_verifier() -> None:
    env = GridWorld(build_episode())
    plan = env.plan_path(use_hidden=True, optimistic_unknown=True)
    assert plan["reachable"] is True
    verification = env.verify_path(FinalAnswer(proposed_path=list(plan["path"]), plan_summary="oracle"))
    assert verification["success"] is True


def test_reveal_and_uncertainty_tools() -> None:
    env = GridWorld(build_episode())
    registry = build_default_tool_registry()
    reveal = registry.call("reveal_observation", env, {"target": {"x": 2, "y": 2}, "radius": 1})
    assert reveal.success is True
    assert reveal.payload["newly_revealed"] > 0
    uncertainty = registry.call("estimate_uncertainty", env, {})
    assert uncertainty.success is True
    assert 0.0 <= uncertainty.payload["unknown_fraction"] <= 1.0

