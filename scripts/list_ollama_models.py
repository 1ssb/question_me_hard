from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hedgeagent.config.loader import load_model
from hedgeagent.config.types import OllamaConfig
from hedgeagent.eval.model_manifest import record_discovery
from hedgeagent.models.ollama_client import probe_ollama, select_preferred_model
from hedgeagent.utils.files import write_json


def main() -> None:
    config = load_model(Path("configs/models/ollama_default.yaml"), OllamaConfig)
    probe = probe_ollama(config)
    manifest_path = Path("manifests/model_manifest.json")
    record_discovery(manifest_path, probe.get("models", []))
    selected = select_preferred_model(probe, config.preferred_model_substrings)
    payload = {"probe": probe, "selected_model": selected}
    write_json(Path("manifests/ollama_probe.json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
