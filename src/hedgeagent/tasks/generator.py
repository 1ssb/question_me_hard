from __future__ import annotations

import json
import random
from pathlib import Path

from hedgeagent.schemas.common import CellState, Point
from hedgeagent.schemas.episode import EpisodeSpec, TaskGenerationConfig
from hedgeagent.utils.files import ensure_dir


def _random_free_point(rng: random.Random, width: int, height: int, occupied: set[tuple[int, int]]) -> Point:
    while True:
        point = Point(x=rng.randrange(width), y=rng.randrange(height))
        if point.as_tuple() not in occupied:
            occupied.add(point.as_tuple())
            return point


def _reveal_region(observed_map: list[list[int]], hidden_map: list[list[int]], center: Point, radius: int) -> None:
    height = len(hidden_map)
    width = len(hidden_map[0])
    for y in range(max(0, center.y - radius), min(height, center.y + radius + 1)):
        for x in range(max(0, center.x - radius), min(width, center.x + radius + 1)):
            observed_map[y][x] = hidden_map[y][x]


def _neighbors(point: Point, width: int, height: int) -> list[Point]:
    candidates: list[Point] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = point.x + dx
        y = point.y + dy
        if 0 <= x < width and 0 <= y < height:
            candidates.append(Point(x=x, y=y))
    return candidates


def _path_exists(hidden_map: list[list[int]], start: Point, goal: Point) -> bool:
    width = len(hidden_map[0])
    height = len(hidden_map)
    queue = [start]
    visited = {start.as_tuple()}
    while queue:
        current = queue.pop(0)
        if current.as_tuple() == goal.as_tuple():
            return True
        for neighbor in _neighbors(current, width, height):
            if neighbor.as_tuple() in visited:
                continue
            if hidden_map[neighbor.y][neighbor.x] == CellState.BLOCKED.value:
                continue
            visited.add(neighbor.as_tuple())
            queue.append(neighbor)
    return False


def _difficulty(distance: int, density: float, budget: int) -> str:
    score = distance + int(density * 10) - budget
    if score <= 8:
        return "easy"
    if score <= 14:
        return "medium"
    return "hard"


def _semantic_hints(rng: random.Random, start: Point, goal: Point, density: float) -> list[str]:
    hints = []
    dx = "east" if goal.x > start.x else "west" if goal.x < start.x else "aligned"
    dy = "south" if goal.y > start.y else "north" if goal.y < start.y else "aligned"
    hints.append(f"Goal lies mostly {dx} and {dy} relative to start.")
    if rng.random() < 0.5:
        if density < 0.16:
            hints.append("Obstacle density is relatively sparse.")
        elif density > 0.24:
            hints.append("Obstacle density is relatively high.")
    return hints


def generate_episode(task_id: str, split: str, seed: int, config: TaskGenerationConfig) -> EpisodeSpec:
    rng = random.Random(seed)
    width = config.width
    height = config.height
    occupied: set[tuple[int, int]] = set()
    start = _random_free_point(rng, width, height, occupied)
    goal = _random_free_point(rng, width, height, occupied)
    density = min(
        0.55,
        max(0.02, config.obstacle_density + rng.uniform(-config.obstacle_density_jitter, config.obstacle_density_jitter)),
    )
    hidden_map: list[list[int]]
    for _attempt in range(config.max_generation_attempts):
        hidden_map = []
        for y in range(height):
            row: list[int] = []
            for x in range(width):
                if (x, y) in {start.as_tuple(), goal.as_tuple()}:
                    row.append(CellState.FREE.value)
                else:
                    row.append(CellState.BLOCKED.value if rng.random() < density else CellState.FREE.value)
            hidden_map.append(row)
        if _path_exists(hidden_map, start, goal):
            break
    else:
        raise RuntimeError(f"Could not generate a solvable map for {task_id}")

    observed_map = [[CellState.UNKNOWN.value for _x in range(width)] for _y in range(height)]
    _reveal_region(observed_map, hidden_map, start, config.initial_reveal_radius)
    _reveal_region(observed_map, hidden_map, goal, config.initial_reveal_radius)
    budget = rng.randint(config.observation_budget_min, config.observation_budget_max)
    distance = abs(start.x - goal.x) + abs(start.y - goal.y)
    hints = _semantic_hints(rng, start, goal, density) if rng.random() < config.semantic_hint_probability else []
    difficulty = _difficulty(distance=distance, density=density, budget=budget)
    metadata = {
        "goal_distance": distance,
        "obstacle_density": round(density, 3),
        "budget_level": "low" if budget <= 2 else "high",
        "initial_unknown_fraction": sum(cell == -1 for row in observed_map for cell in row) / (width * height),
    }
    return EpisodeSpec(
        task_id=task_id,
        split=split,
        seed=seed,
        width=width,
        height=height,
        hidden_map=hidden_map,
        observed_map=observed_map,
        start=start,
        goal=goal,
        semantic_hints=hints,
        observation_budget=budget,
        observation_radius=config.observation_radius,
        difficulty=difficulty,
        task_type="navigation",
        noise_probability=config.noise_probability,
        metadata=metadata,
    )


def generate_dataset_splits(config: TaskGenerationConfig) -> dict[str, list[EpisodeSpec]]:
    dataset: dict[str, list[EpisodeSpec]] = {"train": [], "val": [], "test": []}
    counts = {"train": config.train_size, "val": config.val_size, "test": config.test_size}
    offset = 0
    for split, count in counts.items():
        for index in range(count):
            seed = config.seed + offset + index
            task_id = f"{split}-{seed:06d}"
            dataset[split].append(generate_episode(task_id=task_id, split=split, seed=seed, config=config))
        offset += 10_000
    return dataset


def save_dataset_splits(dataset: dict[str, list[EpisodeSpec]], output_dir: str | Path) -> Path:
    target = ensure_dir(output_dir)
    for split, episodes in dataset.items():
        split_path = target / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")
    return target
