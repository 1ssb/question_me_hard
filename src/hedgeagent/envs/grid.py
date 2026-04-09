from __future__ import annotations

from collections import deque
from copy import deepcopy

from hedgeagent.schemas.agent import FinalAnswer
from hedgeagent.schemas.common import CellState, Point
from hedgeagent.schemas.episode import EpisodeSpec, EpisodeState, ObservationRecord


class GridWorld:
    def __init__(self, spec: EpisodeSpec) -> None:
        self.spec = spec
        self.hidden_map = deepcopy(spec.hidden_map)
        self.observed_map = deepcopy(spec.observed_map)
        self.observation_budget_remaining = spec.observation_budget
        self.observations_used = 0
        self.observation_history: list[ObservationRecord] = []

    def visible_state(self) -> EpisodeState:
        return EpisodeState(
            task_id=self.spec.task_id,
            split=self.spec.split,
            seed=self.spec.seed,
            width=self.spec.width,
            height=self.spec.height,
            observed_map=deepcopy(self.observed_map),
            start=self.spec.start,
            goal=self.spec.goal,
            semantic_hints=list(self.spec.semantic_hints),
            observation_budget_remaining=self.observation_budget_remaining,
            observation_radius=self.spec.observation_radius,
            observations_used=self.observations_used,
            difficulty=self.spec.difficulty,
            task_type=self.spec.task_type,
            metadata=deepcopy(self.spec.metadata),
        )

    def within_bounds(self, point: Point) -> bool:
        return 0 <= point.x < self.spec.width and 0 <= point.y < self.spec.height

    def cell_hidden(self, point: Point) -> int:
        return self.hidden_map[point.y][point.x]

    def cell_observed(self, point: Point) -> int:
        return self.observed_map[point.y][point.x]

    def reveal(self, center: Point, radius: int | None = None) -> ObservationRecord:
        if self.observation_budget_remaining <= 0:
            raise ValueError("Observation budget exhausted.")
        effective_radius = radius or self.spec.observation_radius
        newly_revealed = 0
        revealed_points: list[Point] = []
        for y in range(max(0, center.y - effective_radius), min(self.spec.height, center.y + effective_radius + 1)):
            for x in range(max(0, center.x - effective_radius), min(self.spec.width, center.x + effective_radius + 1)):
                if self.observed_map[y][x] == CellState.UNKNOWN.value:
                    newly_revealed += 1
                    revealed_points.append(Point(x=x, y=y))
                self.observed_map[y][x] = self.hidden_map[y][x]
        self.observation_budget_remaining -= 1
        self.observations_used += 1
        record = ObservationRecord(
            center=center,
            radius=effective_radius,
            newly_revealed=newly_revealed,
            budget_after=self.observation_budget_remaining,
            revealed_points=revealed_points,
        )
        self.observation_history.append(record)
        return record

    def frontier_points(self) -> list[Point]:
        frontiers: list[Point] = []
        seen: set[tuple[int, int]] = set()
        for y in range(self.spec.height):
            for x in range(self.spec.width):
                if self.observed_map[y][x] != CellState.UNKNOWN.value:
                    continue
                point = Point(x=x, y=y)
                if any(
                    self._traversable(neighbor, use_hidden=False, optimistic_unknown=False)
                    for neighbor in self.neighbors(point)
                ):
                    if point.as_tuple() not in seen:
                        frontiers.append(point)
                        seen.add(point.as_tuple())
        return frontiers

    def neighbors(self, point: Point) -> list[Point]:
        candidates: list[Point] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x = point.x + dx
            y = point.y + dy
            if 0 <= x < self.spec.width and 0 <= y < self.spec.height:
                candidates.append(Point(x=x, y=y))
        return candidates

    def _traversable(self, point: Point, *, use_hidden: bool, optimistic_unknown: bool) -> bool:
        if not self.within_bounds(point):
            return False
        if use_hidden:
            return self.hidden_map[point.y][point.x] == CellState.FREE.value
        value = self.observed_map[point.y][point.x]
        if value == CellState.BLOCKED.value:
            return False
        if value == CellState.UNKNOWN.value and not optimistic_unknown:
            return False
        return True

    def plan_path(
        self,
        *,
        use_hidden: bool = False,
        optimistic_unknown: bool = True,
    ) -> dict[str, object]:
        start = self.spec.start
        goal = self.spec.goal
        if not self._traversable(start, use_hidden=use_hidden, optimistic_unknown=optimistic_unknown):
            return {"reachable": False, "path": [], "path_length": None, "unknown_cells_on_path": None}
        if not self._traversable(goal, use_hidden=use_hidden, optimistic_unknown=optimistic_unknown):
            return {"reachable": False, "path": [], "path_length": None, "unknown_cells_on_path": None}

        queue: deque[Point] = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start.as_tuple(): None}
        while queue:
            current = queue.popleft()
            if current.as_tuple() == goal.as_tuple():
                break
            for neighbor in self.neighbors(current):
                if not self._traversable(neighbor, use_hidden=use_hidden, optimistic_unknown=optimistic_unknown):
                    continue
                if neighbor.as_tuple() in parents:
                    continue
                parents[neighbor.as_tuple()] = current.as_tuple()
                queue.append(neighbor)

        if goal.as_tuple() not in parents:
            return {"reachable": False, "path": [], "path_length": None, "unknown_cells_on_path": None}

        path_rev: list[Point] = []
        cursor: tuple[int, int] | None = goal.as_tuple()
        while cursor is not None:
            path_rev.append(Point(x=cursor[0], y=cursor[1]))
            cursor = parents[cursor]
        path = list(reversed(path_rev))
        unknown_cells_on_path = sum(
            1 for point in path if self.observed_map[point.y][point.x] == CellState.UNKNOWN.value
        )
        return {
            "reachable": True,
            "path": path,
            "path_length": max(0, len(path) - 1),
            "unknown_cells_on_path": unknown_cells_on_path,
        }

    def verify_path(self, answer: FinalAnswer | list[Point]) -> dict[str, object]:
        path = answer.proposed_path if isinstance(answer, FinalAnswer) else answer
        if not path:
            return {
                "success": False,
                "safe": False,
                "reached_goal": False,
                "collisions": 0,
                "reason": "empty_path",
            }
        if path[0].as_tuple() != self.spec.start.as_tuple():
            return {
                "success": False,
                "safe": False,
                "reached_goal": False,
                "collisions": 0,
                "reason": "path_does_not_start_at_start",
            }
        collisions = 0
        for current, nxt in zip(path, path[1:]):
            if abs(current.x - nxt.x) + abs(current.y - nxt.y) != 1:
                return {
                    "success": False,
                    "safe": False,
                    "reached_goal": False,
                    "collisions": collisions,
                    "reason": "non_adjacent_step",
                }
            if self.hidden_map[nxt.y][nxt.x] == CellState.BLOCKED.value:
                collisions += 1
        reached_goal = path[-1].as_tuple() == self.spec.goal.as_tuple()
        safe = collisions == 0
        return {
            "success": safe and reached_goal,
            "safe": safe,
            "reached_goal": reached_goal,
            "collisions": collisions,
            "reason": "ok" if safe and reached_goal else "goal_miss" if safe else "collision",
        }

    def guaranteed_path_exists(self) -> bool:
        result = self.plan_path(use_hidden=False, optimistic_unknown=False)
        return bool(result["reachable"])

    def hidden_path_exists(self) -> bool:
        result = self.plan_path(use_hidden=True, optimistic_unknown=True)
        return bool(result["reachable"])

    def unknown_fraction(self) -> float:
        total = self.spec.width * self.spec.height
        unknown = sum(cell == CellState.UNKNOWN.value for row in self.observed_map for cell in row)
        return unknown / total if total else 0.0

    def ascii_map(self, include_hidden: bool = False) -> str:
        source = self.hidden_map if include_hidden else self.observed_map
        rows: list[str] = []
        for y, row in enumerate(source):
            chars: list[str] = []
            for x, cell in enumerate(row):
                if (x, y) == self.spec.start.as_tuple():
                    chars.append("S")
                elif (x, y) == self.spec.goal.as_tuple():
                    chars.append("G")
                elif cell == CellState.UNKNOWN.value:
                    chars.append("?")
                elif cell == CellState.BLOCKED.value:
                    chars.append("#")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        return "\n".join(rows)
