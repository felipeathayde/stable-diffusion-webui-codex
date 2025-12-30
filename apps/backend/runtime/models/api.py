"""Public facade for checkpoint/VAE registry operations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from . import registry
from .types import CheckpointRecord, VAERecord


def list_checkpoints(*, refresh: bool = False) -> List[CheckpointRecord]:
    return registry.list_checkpoints(refresh=refresh)


def list_checkpoints_as_dict(*, refresh: bool = False) -> List[Dict[str, object]]:
    return [record.as_dict() for record in list_checkpoints(refresh=refresh)]


def list_vaes(*, refresh: bool = False) -> List[VAERecord]:
    return registry.list_vaes(refresh=refresh)


def list_vaes_as_dict(*, refresh: bool = False) -> List[Dict[str, object]]:
    return [record.as_dict() for record in list_vaes(refresh=refresh)]


def find_checkpoint(name_or_path: str) -> Optional[CheckpointRecord]:
    reg = registry.get_registry()
    record = reg.get_checkpoint(name_or_path)
    if record is not None:
        return record
    # Try matching by title (which includes extension like 'model.gguf')
    for candidate in reg.list_checkpoints(refresh=False):
        if candidate.title == name_or_path:
            return candidate
    # Try matching by filename or stem (without extension)
    name_stem = Path(name_or_path).stem
    for candidate in reg.list_checkpoints(refresh=False):
        if candidate.name == name_stem:
            return candidate
        if Path(candidate.filename).name == name_or_path:
            return candidate
    # Attempt to match by path when a file/directory path is supplied.
    path = Path(name_or_path)
    if path.is_file() or path.is_dir():
        for candidate in reg.list_checkpoints(refresh=False):
            if Path(candidate.filename) == path or Path(candidate.path) == path:
                return candidate
    return None


def refresh() -> None:
    registry.refresh()


__all__ = [
    "find_checkpoint",
    "list_checkpoints",
    "list_checkpoints_as_dict",
    "list_vaes",
    "list_vaes_as_dict",
    "refresh",
]
