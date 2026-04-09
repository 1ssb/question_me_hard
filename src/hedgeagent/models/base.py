from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from hedgeagent.schemas.common import StrictModel


class ModelResponse(StrictModel):
    model_name: str
    text: str
    latency_ms: float = Field(ge=0.0)
    raw_request: dict[str, Any]
    raw_response: dict[str, Any]
    error: str | None = None


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> ModelResponse:
        raise NotImplementedError

