from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from hedgeagent.envs.grid import GridWorld
from hedgeagent.schemas.agent import AgentDecision, ModelCallRecord, ToolResultEnvelope, TraceStep
from hedgeagent.schemas.episode import EpisodeState


@dataclass
class DecisionContext:
    state: EpisodeState
    env: GridWorld
    step_index: int
    max_steps: int
    tool_history: list[ToolResultEnvelope]
    trace: list[TraceStep]
    rng: random.Random


@dataclass
class PolicyStepResult:
    decision: AgentDecision | None
    model_call: ModelCallRecord | None = None
    schema_valid: bool = True


class BasePolicy(ABC):
    name: str = "base"
    model_name: str | None = None

    @abstractmethod
    def decide(self, context: DecisionContext) -> PolicyStepResult:
        raise NotImplementedError

