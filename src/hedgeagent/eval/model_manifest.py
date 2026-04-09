from __future__ import annotations

import json
from pathlib import Path

from hedgeagent.schemas.metrics import ModelManifestEntry
from hedgeagent.utils.files import ensure_dir
from hedgeagent.utils.time import utc_now_iso


def load_model_manifest(path: str | Path) -> dict[str, ModelManifestEntry]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {item["model_name"]: ModelManifestEntry.model_validate(item) for item in payload}


def save_model_manifest(path: str | Path, entries: dict[str, ModelManifestEntry]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump([entry.model_dump(mode="json") for entry in entries.values()], handle, indent=2, sort_keys=True)
    return target


def update_model_manifest(
    path: str | Path,
    model_name: str,
    status: str,
    *,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    entries = load_model_manifest(path)
    now = utc_now_iso()
    entry = entries.get(model_name)
    if entry is None:
        entry = ModelManifestEntry(model_name=model_name, status=status, discovered_at=now, metadata=metadata or {})
    entry.status = status
    entry.last_error = error
    if metadata:
        merged = dict(entry.metadata)
        merged.update(metadata)
        entry.metadata = merged
    if status == "smoke_tested":
        entry.smoke_tested_at = now
    elif status == "schema_validated":
        entry.schema_validated_at = now
    elif status == "pilot_evaluated":
        entry.pilot_evaluated_at = now
    elif status == "full_evaluated":
        entry.full_evaluated_at = now
    entries[model_name] = entry
    return save_model_manifest(path, entries)


def record_discovery(path: str | Path, models: list[dict[str, object]]) -> Path:
    entries = load_model_manifest(path)
    now = utc_now_iso()
    for model in models:
        name = str(model["name"])
        existing = entries.get(name)
        if existing is None:
            entries[name] = ModelManifestEntry(
                model_name=name,
                status="discovered",
                discovered_at=now,
                metadata=dict(model),
            )
        else:
            existing.metadata.update(dict(model))
    return save_model_manifest(path, entries)

