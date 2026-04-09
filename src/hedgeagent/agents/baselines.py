from __future__ import annotations

from dataclasses import dataclass

from hedgeagent.agents.base import BasePolicy, DecisionContext, PolicyStepResult
from hedgeagent.schemas.agent import AgentDecision, FinalAnswer
from hedgeagent.schemas.common import ActionType, Point


def _path_to_answer(path: list[Point], summary: str) -> FinalAnswer:
    return FinalAnswer(proposed_path=path, plan_summary=summary)


def _optimistic_path(context: DecisionContext) -> list[Point]:
    result = context.env.plan_path(use_hidden=False, optimistic_unknown=True)
    return list(result["path"])


def _fallback_manhattan(context: DecisionContext) -> list[Point]:
    points = [context.state.start]
    cursor = Point(x=context.state.start.x, y=context.state.start.y)
    while cursor.x != context.state.goal.x:
        cursor = Point(x=cursor.x + (1 if context.state.goal.x > cursor.x else -1), y=cursor.y)
        points.append(cursor)
    while cursor.y != context.state.goal.y:
        cursor = Point(x=cursor.x, y=cursor.y + (1 if context.state.goal.y > cursor.y else -1))
        points.append(cursor)
    return points


def _query_target(context: DecisionContext) -> Point:
    path = _optimistic_path(context)
    for point in path:
        if context.state.observed_map[point.y][point.x] == -1:
            return point
    frontiers = context.env.frontier_points()
    if frontiers:
        return min(frontiers, key=lambda point: abs(point.x - context.state.goal.x) + abs(point.y - context.state.goal.y))
    return context.state.goal


class AlwaysActPolicy(BasePolicy):
    name = "always_act"

    def decide(self, context: DecisionContext) -> PolicyStepResult:
        path = _optimistic_path(context) or _fallback_manhattan(context)
        return PolicyStepResult(
            decision=AgentDecision(
                action_type=ActionType.ACT,
                rationale_brief="Commit immediately using an optimistic visible-state path.",
                tool_args={},
                final_answer=_path_to_answer(path, "Optimistic visible-map path."),
                confidence=0.9,
            )
        )


class AlwaysQueryPolicy(BasePolicy):
    name = "always_query_until_budget_exhausted"

    def decide(self, context: DecisionContext) -> PolicyStepResult:
        if context.state.observation_budget_remaining > 0:
            target = _query_target(context)
            return PolicyStepResult(
                decision=AgentDecision(
                    action_type=ActionType.QUERY,
                    rationale_brief="Use remaining budget before acting.",
                    tool_args={"target": target.model_dump(mode="json"), "radius": context.state.observation_radius},
                    confidence=0.7,
                    expected_information_gain=0.5,
                )
            )
        path = _optimistic_path(context) or _fallback_manhattan(context)
        return PolicyStepResult(
            decision=AgentDecision(
                action_type=ActionType.ACT,
                rationale_brief="Budget exhausted, act with the best available optimistic path.",
                tool_args={},
                final_answer=_path_to_answer(path, "Post-query optimistic path."),
                confidence=0.75,
            )
        )


class AlwaysAbstainPolicy(BasePolicy):
    name = "always_abstain"

    def decide(self, context: DecisionContext) -> PolicyStepResult:
        del context
        return PolicyStepResult(
            decision=AgentDecision(
                action_type=ActionType.ABSTAIN,
                rationale_brief="Abstain regardless of state.",
                tool_args={},
                confidence=1.0,
                abstain_reason="policy_default",
            )
        )


@dataclass
class UncertaintyThresholdPolicy(BasePolicy):
    threshold: float = 0.25
    name: str = "uncertainty_threshold"

    def decide(self, context: DecisionContext) -> PolicyStepResult:
        plan = context.env.plan_path(use_hidden=False, optimistic_unknown=True)
        unknown_fraction = 1.0
        if plan["reachable"] and plan["path"]:
            path = list(plan["path"])
            unknown_cells = sum(1 for point in path if context.state.observed_map[point.y][point.x] == -1)
            unknown_fraction = unknown_cells / max(1, len(path))
            if unknown_fraction <= self.threshold:
                return PolicyStepResult(
                    decision=AgentDecision(
                        action_type=ActionType.ACT,
                        rationale_brief="Visible path uncertainty is below threshold.",
                        tool_args={},
                        final_answer=_path_to_answer(path, "Threshold policy path."),
                        confidence=0.85,
                    )
                )
        if context.state.observation_budget_remaining > 0:
            target = _query_target(context)
            return PolicyStepResult(
                decision=AgentDecision(
                    action_type=ActionType.QUERY,
                    rationale_brief="Path uncertainty is still high, query more state.",
                    tool_args={"target": target.model_dump(mode="json"), "radius": context.state.observation_radius},
                    confidence=0.65,
                    expected_information_gain=min(1.0, max(0.2, unknown_fraction)),
                )
            )
        return PolicyStepResult(
            decision=AgentDecision(
                action_type=ActionType.ABSTAIN,
                rationale_brief="Budget exhausted and uncertainty remains high.",
                tool_args={},
                confidence=0.8,
                abstain_reason="uncertainty_above_threshold",
            )
        )


class OraclePolicy(BasePolicy):
    name = "shortest_path_oracle"

    def decide(self, context: DecisionContext) -> PolicyStepResult:
        plan = context.env.plan_path(use_hidden=True, optimistic_unknown=True)
        path = list(plan["path"])
        return PolicyStepResult(
            decision=AgentDecision(
                action_type=ActionType.ACT,
                rationale_brief="Use the hidden-map shortest path oracle.",
                tool_args={},
                final_answer=_path_to_answer(path, "Oracle hidden-map shortest path."),
                confidence=1.0,
            )
        )


class RandomPolicy(BasePolicy):
    name = "random_policy"

    def decide(self, context: DecisionContext) -> PolicyStepResult:
        choices = ["ACT", "ABSTAIN"]
        if context.state.observation_budget_remaining > 0:
            choices.append("QUERY")
        pick = context.rng.choice(choices)
        if pick == "QUERY":
            target = _query_target(context)
            return PolicyStepResult(
                decision=AgentDecision(
                    action_type=ActionType.QUERY,
                    rationale_brief="Randomly query for more information.",
                    tool_args={"target": target.model_dump(mode="json"), "radius": context.state.observation_radius},
                    confidence=0.4,
                    expected_information_gain=0.5,
                )
            )
        if pick == "ABSTAIN":
            return PolicyStepResult(
                decision=AgentDecision(
                    action_type=ActionType.ABSTAIN,
                    rationale_brief="Randomly abstain.",
                    tool_args={},
                    confidence=0.4,
                    abstain_reason="random_choice",
                )
            )
        path = _optimistic_path(context) or _fallback_manhattan(context)
        return PolicyStepResult(
            decision=AgentDecision(
                action_type=ActionType.ACT,
                rationale_brief="Randomly act using an optimistic path.",
                tool_args={},
                final_answer=_path_to_answer(path, "Random optimistic path."),
                confidence=0.4,
            )
        )

