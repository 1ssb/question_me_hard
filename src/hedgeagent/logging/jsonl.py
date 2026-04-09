from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hedgeagent.utils.files import ensure_dir


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def write(self, payload: Any) -> None:
        serializable = payload
        if hasattr(payload, "model_dump"):
            serializable = payload.model_dump(mode="json")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, sort_keys=True))
            handle.write("\n")

