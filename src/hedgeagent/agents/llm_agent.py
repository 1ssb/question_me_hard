from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from hedgeagent.agents.base import BasePolicy, DecisionContext, PolicyStepResult
from hedgeagent.models.base import BaseLLMClient, ModelResponse
from hedgeagent.prompts.prompt_builder import SCHEMA_DESCRIPTION, build_decision_prompt
from hedgeagent.schemas.agent import AgentDecision, ModelCallRecord
from hedgeagent.schemas.common import ActionType


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JSONDecodeError("No JSON object found in model output.", text, 0)
    return text[start : end + 1]


class LLMPolicy(BasePolicy):
    name = "ollama"

    def __init__(self, client: BaseLLMClient, prompt_version: str = "decision_prompt_v1.txt") -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.model_name = getattr(client, "model_name", None)

    def _to_model_record(self, response: ModelResponse, parse_error: str | None = None, repaired: bool = False) -> ModelCallRecord:
        return ModelCallRecord(
            model_name=response.model_name,
            prompt=str(response.raw_request.get("prompt") or response.raw_request.get("messages")),
            response_text=response.text,
            latency_ms=response.latency_ms,
            raw_request=response.raw_request,
            raw_response=response.raw_response,
            parse_error=parse_error,
            repaired=repaired,
        )

    def _normalize_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        repaired = False
        normalized = dict(payload)
        action_type = normalized.get("action_type")
        tool_args = normalized.get("tool_args")
        if isinstance(tool_args, dict):
            tool_args = dict(tool_args)
            if action_type == ActionType.QUERY.value and "target" not in tool_args and "center" in tool_args:
                tool_args["target"] = tool_args.pop("center")
                repaired = True
            normalized["tool_args"] = tool_args
        if action_type == ActionType.QUERY.value and normalized.get("chosen_tool") == "reveal_observation":
            normalized["chosen_tool"] = None
            repaired = True
        return normalized, repaired

    def _parse_decision(self, response: ModelResponse) -> AgentDecision:
        payload = json.loads(_extract_json_object(response.text))
        normalized_payload, _repaired = self._normalize_payload(payload)
        return AgentDecision.model_validate(normalized_payload)

    def _repair(self, original_prompt: str, invalid_text: str, error_text: str) -> ModelResponse:
        repair_prompt = (
            "The previous response was invalid. Return only one JSON object that matches this schema exactly.\n"
            f"{SCHEMA_DESCRIPTION}\n"
            "For QUERY decisions, use tool_args.target, not center.\n"
            f"Original task prompt:\n{original_prompt}\n"
            f"Invalid response:\n{invalid_text}\n"
            f"Validation error:\n{error_text}\n"
        )
        return self.client.complete(prompt=repair_prompt, system_prompt="Return only JSON.")

    def decide(self, context: DecisionContext) -> PolicyStepResult:
        prompt = build_decision_prompt(
            state=context.state,
            tool_history=context.tool_history,
            step_index=context.step_index,
            max_steps=context.max_steps,
            version=self.prompt_version,
        )
        response = self.client.complete(prompt=prompt, system_prompt="Return only JSON.")
        if response.error:
            return PolicyStepResult(
                decision=None,
                model_call=self._to_model_record(response, parse_error=response.error),
                schema_valid=False,
            )
        try:
            payload = json.loads(_extract_json_object(response.text))
            normalized_payload, repaired = self._normalize_payload(payload)
            decision = AgentDecision.model_validate(normalized_payload)
            return PolicyStepResult(
                decision=decision,
                model_call=self._to_model_record(response, repaired=repaired),
                schema_valid=True,
            )
        except (JSONDecodeError, ValueError) as exc:
            repaired = self._repair(prompt, response.text, str(exc))
            if repaired.error:
                return PolicyStepResult(
                    decision=None,
                    model_call=self._to_model_record(repaired, parse_error=repaired.error, repaired=True),
                    schema_valid=False,
                )
            try:
                payload = json.loads(_extract_json_object(repaired.text))
                normalized_payload, normalized = self._normalize_payload(payload)
                decision = AgentDecision.model_validate(normalized_payload)
                return PolicyStepResult(
                    decision=decision,
                    model_call=self._to_model_record(repaired, repaired=True if normalized else True),
                    schema_valid=True,
                )
            except (JSONDecodeError, ValueError) as repair_exc:
                return PolicyStepResult(
                    decision=None,
                    model_call=self._to_model_record(repaired, parse_error=str(repair_exc), repaired=True),
                    schema_valid=False,
                )
