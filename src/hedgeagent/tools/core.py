from __future__ import annotations

from hedgeagent.envs.grid import GridWorld
from hedgeagent.schemas.agent import FinalAnswer
from hedgeagent.schemas.common import Point


def reveal_observation(env: GridWorld, args: dict[str, object]) -> dict[str, object]:
    target_data = args.get("target")
    if not isinstance(target_data, dict):
        raise ValueError("target must be a mapping with x/y.")
    center = Point.model_validate(target_data)
    radius = int(args.get("radius", env.spec.observation_radius))
    record = env.reveal(center=center, radius=radius)
    return {
        "center": record.center.model_dump(mode="json"),
        "radius": record.radius,
        "newly_revealed": record.newly_revealed,
        "budget_after": record.budget_after,
        "revealed_points": [point.model_dump(mode="json") for point in record.revealed_points],
    }


def plan_path(env: GridWorld, args: dict[str, object]) -> dict[str, object]:
    use_hidden = bool(args.get("use_hidden", False))
    optimistic_unknown = bool(args.get("optimistic_unknown", True))
    result = env.plan_path(use_hidden=use_hidden, optimistic_unknown=optimistic_unknown)
    return {
        "reachable": result["reachable"],
        "path_length": result["path_length"],
        "unknown_cells_on_path": result["unknown_cells_on_path"],
        "path": [point.model_dump(mode="json") for point in result["path"]],
        "use_hidden": use_hidden,
        "optimistic_unknown": optimistic_unknown,
    }


def estimate_uncertainty(env: GridWorld, args: dict[str, object]) -> dict[str, object]:
    optimistic = env.plan_path(use_hidden=False, optimistic_unknown=True)
    pessimistic = env.plan_path(use_hidden=False, optimistic_unknown=False)
    frontier_count = len(env.frontier_points())
    total = env.spec.width * env.spec.height
    unknown_count = sum(cell == -1 for row in env.observed_map for cell in row)
    blocked_known = sum(cell == 1 for row in env.observed_map for cell in row)
    return {
        "unknown_fraction": unknown_count / total if total else 0.0,
        "known_blocked_fraction": blocked_known / total if total else 0.0,
        "frontier_count": frontier_count,
        "optimistic_path_exists": bool(optimistic["reachable"]),
        "pessimistic_path_exists": bool(pessimistic["reachable"]),
        "optimistic_unknown_cells_on_path": optimistic["unknown_cells_on_path"],
        "observation_budget_remaining": env.observation_budget_remaining,
    }


def verify_action(env: GridWorld, args: dict[str, object]) -> dict[str, object]:
    answer_data = args.get("final_answer")
    if not isinstance(answer_data, dict):
        raise ValueError("final_answer must be provided for verify_action.")
    answer = FinalAnswer.model_validate(answer_data)
    return env.verify_path(answer)


def summarize_state(env: GridWorld, args: dict[str, object]) -> dict[str, object]:
    del args
    unknown_count = sum(cell == -1 for row in env.observed_map for cell in row)
    known_free = sum(cell == 0 for row in env.observed_map for cell in row)
    known_blocked = sum(cell == 1 for row in env.observed_map for cell in row)
    summary_lines = [
        f"task_id={env.spec.task_id}",
        f"budget_remaining={env.observation_budget_remaining}",
        f"observations_used={env.observations_used}",
        f"known_free={known_free}",
        f"known_blocked={known_blocked}",
        f"unknown={unknown_count}",
        f"frontiers={len(env.frontier_points())}",
        f"start=({env.spec.start.x},{env.spec.start.y}) goal=({env.spec.goal.x},{env.spec.goal.y})",
        env.ascii_map(include_hidden=False),
    ]
    return {
        "summary_text": "\n".join(summary_lines),
        "known_free": known_free,
        "known_blocked": known_blocked,
        "unknown": unknown_count,
        "frontier_count": len(env.frontier_points()),
    }

