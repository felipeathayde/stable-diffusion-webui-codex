from __future__ import annotations

import os
from typing import Optional

from apps.server.backend.registry.checkpoints import list_checkpoints
from apps.server.backend.runtime.models.loader import forge_loader, load_engine_from_diffusers


def _find_checkpoint_by_name(name: str):
    for entry in list_checkpoints():
        if entry.name == name or entry.title == name:
            return entry
    return None


def load_engine(name_or_path: str):
    """Load an engine instance by checkpoint name or path.

    - If a diffusers repo directory is detected (has model_index.json), load via native diffusers loader.
    - If a file path to a state dict is given, use forge_loader.
    - Otherwise, resolve by registry name and load accordingly.
    """
    path = name_or_path
    if os.path.isdir(path):
        mi = os.path.join(path, "model_index.json")
        if os.path.isfile(mi):
            return load_engine_from_diffusers(path)
        raise ValueError(f"Not a diffusers repo (missing model_index.json): {path}")
    if os.path.isfile(path):
        return forge_loader(path)

    entry = _find_checkpoint_by_name(name_or_path)
    if entry is None:
        raise ValueError(f"Checkpoint not found: {name_or_path}")
    if entry.metadata.get("format") == "diffusers" or os.path.isfile(os.path.join(entry.path, "model_index.json")):
        return load_engine_from_diffusers(entry.path)
    return forge_loader(entry.filename)


__all__ = ["load_engine"]

