from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from apps.backend.runtime.models import api as model_api


@dataclass
class CheckpointEntry:
    name: str
    title: str
    path: str
    model_name: str
    filename: str
    short_hash: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def list_checkpoints(models_root: str = "models", vendored_hf_root: str = "apps/backend/huggingface") -> List[CheckpointEntry]:
    """Compatibility wrapper delegating to the runtime model registry."""

    entries: List[CheckpointEntry] = []
    for record in model_api.list_checkpoints():
        entries.append(
            CheckpointEntry(
                name=record.name,
                title=record.title,
                path=record.path,
                model_name=record.model_name,
                filename=record.filename,
                short_hash=record.short_hash,
                metadata={"format": record.format.value, **(record.metadata or {})},
            )
        )
    entries.sort(key=lambda e: e.title.lower())
    return entries


def describe_checkpoints(models_root: str = "models", vendored_hf_root: str = "apps/backend/huggingface") -> List[Dict[str, Any]]:
    """Return metadata describing the discovered checkpoints."""

    return model_api.list_checkpoints_as_dict()


__all__ = ["CheckpointEntry", "list_checkpoints", "describe_checkpoints"]
