from __future__ import annotations

import os
from pathlib import Path

from apps.backend.runtime.models.safety import safe_torch_load


def controlnet_cache_root() -> Path:
    root = os.environ.get("CODEX_CONTROLNET_CACHE")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "codex" / "controlnet"


def resolve_weights_file(*relative: str) -> Path:
    root = controlnet_cache_root()
    path = root.joinpath(*relative)
    if not path.is_file():
        raise FileNotFoundError(
            f"ControlNet preprocessor weights not found: {path}.\n"
            "Populate the file manually or configure CODEX_CONTROLNET_CACHE."
        )
    return path


def load_state_dict(path: Path):
    return safe_torch_load(str(path), map_location="cpu")
