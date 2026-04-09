from __future__ import annotations

import time
from collections.abc import Callable

from hedgeagent.envs.grid import GridWorld
from hedgeagent.schemas.agent import ToolResultEnvelope
from . import core


ToolFn = Callable[[GridWorld, dict[str, object]], dict[str, object]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, tool_fn: ToolFn) -> None:
        self._tools[name] = tool_fn

    def call(self, name: str, env: GridWorld, args: dict[str, object] | None = None) -> ToolResultEnvelope:
        if name not in self._tools:
            return ToolResultEnvelope(name=name, success=False, payload={}, error="unknown_tool", latency_ms=0.0)
        start = time.perf_counter()
        try:
            payload = self._tools[name](env, args or {})
            return ToolResultEnvelope(
                name=name,
                success=True,
                payload=payload,
                error=None,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResultEnvelope(
                name=name,
                success=False,
                payload={},
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("reveal_observation", core.reveal_observation)
    registry.register("plan_path", core.plan_path)
    registry.register("estimate_uncertainty", core.estimate_uncertainty)
    registry.register("verify_action", core.verify_action)
    registry.register("summarize_state", core.summarize_state)
    return registry
