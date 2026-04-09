from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any
from urllib import error, request

from hedgeagent.config.types import OllamaConfig
from hedgeagent.models.base import BaseLLMClient, ModelResponse


def _run_command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": "command_not_found"}


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> tuple[bool, dict[str, Any]]:
    try:
        if payload is None:
            req = request.Request(url, method="GET")
        else:
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=timeout) as response:
            return True, json.loads(response.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return False, {"error": str(exc)}


def _parse_cli_models(stdout: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return models
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        models.append({"name": parts[0], "raw": line})
    return models


def probe_ollama(config: OllamaConfig) -> dict[str, Any]:
    executable = shutil.which("ollama")
    version = _run_command(["ollama", "--version"]) if executable else {"ok": False, "stdout": "", "stderr": ""}
    help_info = _run_command(["ollama", "--help"]) if executable else {"ok": False, "stdout": "", "stderr": ""}
    cli_list = _run_command(["ollama", "list"]) if executable else {"ok": False, "stdout": "", "stderr": ""}
    processes = _run_command(["pgrep", "-a", "-x", "ollama"])
    http_ok, http_tags = _http_json(f"{config.endpoint()}/api/tags", timeout=min(5.0, config.timeout_seconds))
    version_ok, version_payload = _http_json(f"{config.endpoint()}/api/version", timeout=min(5.0, config.timeout_seconds))
    models = []
    if http_ok:
        for item in http_tags.get("models", []):
            models.append(
                {
                    "name": item.get("name"),
                    "size": item.get("size"),
                    "modified_at": item.get("modified_at"),
                    "details": item.get("details", {}),
                }
            )
    elif cli_list.get("ok"):
        models = _parse_cli_models(cli_list.get("stdout", ""))
    return {
        "executable_exists": executable is not None,
        "executable_path": executable,
        "version": version.get("stdout", ""),
        "help_snippet": "\n".join(help_info.get("stdout", "").splitlines()[:12]),
        "daemon_running": http_ok or version_ok,
        "daemon_version": version_payload.get("version") if version_ok else None,
        "running_processes": processes.get("stdout", ""),
        "models": [model for model in models if model.get("name")],
        "cli_list_ok": cli_list.get("ok", False),
        "base_url": config.base_url,
        "http_tags_error": None if http_ok else http_tags.get("error"),
    }


def select_preferred_model(probe: dict[str, Any], preferred_substrings: list[str]) -> str | None:
    models = probe.get("models", [])
    if not models:
        return None

    def rank(model: dict[str, Any]) -> tuple[int, int, str]:
        name = str(model.get("name", "")).lower()
        preference_score = len(preferred_substrings)
        for index, token in enumerate(preferred_substrings):
            if token.lower() in name:
                preference_score = index
                break
        size = int(model.get("size", 10**18) or 10**18)
        return (preference_score, size, name)

    return sorted(models, key=rank)[0]["name"]


class OllamaClient(BaseLLMClient):
    def __init__(self, model_name: str, config: OllamaConfig) -> None:
        self.model_name = model_name
        self.config = config

    def _build_request(self, prompt: str, system_prompt: str | None = None) -> tuple[str, dict[str, Any]]:
        options = {
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "num_predict": self.config.max_tokens,
        }
        if self.config.prompt_mode == "chat":
            return (
                f"{self.config.endpoint()}/api/chat",
                {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": options,
                },
            )
        return (
            f"{self.config.endpoint()}/api/generate",
            {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt or "",
                "stream": False,
                "options": options,
            },
        )

    def complete(self, prompt: str, system_prompt: str | None = None) -> ModelResponse:
        url, payload = self._build_request(prompt=prompt, system_prompt=system_prompt)
        last_error = None
        for _attempt in range(self.config.max_retries + 1):
            start = time.perf_counter()
            ok, response_payload = _http_json(url, payload=payload, timeout=self.config.timeout_seconds)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if ok:
                if self.config.prompt_mode == "chat":
                    text = response_payload.get("message", {}).get("content", "")
                else:
                    text = response_payload.get("response", "")
                if text:
                    return ModelResponse(
                        model_name=response_payload.get("model", self.model_name),
                        text=text,
                        latency_ms=latency_ms,
                        raw_request=payload,
                        raw_response=response_payload,
                        error=None,
                    )
                last_error = "empty_response"
            else:
                last_error = response_payload.get("error", "request_failed")
        return ModelResponse(
            model_name=self.model_name,
            text="",
            latency_ms=0.0,
            raw_request=payload,
            raw_response={"error": last_error},
            error=last_error,
        )
