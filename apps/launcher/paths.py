from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import logging

LOGGER = logging.getLogger("codex.launcher.paths")


def _normalize_path(value: str | os.PathLike[str] | None, *, fallback: Path) -> Path:
    """Normalise optional paths, raising with context on failure."""
    if value is None:
        return fallback
    try:
        path = Path(value).expanduser()
        return path.resolve()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to normalise path '{value}': {exc}") from exc


@dataclass(frozen=True)
class CodexPaths:
    """Resolved canonical paths for the runtime."""

    project_root: Path
    data_dir: Path
    models_dir: Path
    extensions_dir: Path
    extensions_builtin_dir: Path
    outputs_dir: Path
    configs_dir: Path
    default_config: Path
    default_checkpoint: Path


def resolve_paths(
    *,
    project_root: Path,
    data_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> CodexPaths:
    """Resolve launcher paths with strict normalisation."""
    data = _normalize_path(data_dir, fallback=project_root / "data")
    models = _normalize_path(models_dir, fallback=data / "models")
    extensions = data / "extensions"
    extensions_builtin = project_root / "extensions-builtin"
    outputs = data / "outputs"
    configs = project_root / "configs"
    default_cfg = configs / "v1-inference.yaml"
    default_ckpt = project_root / "model.ckpt"

    LOGGER.debug(
        "Resolved runtime paths",
        extra={
            "project_root": str(project_root),
            "data": str(data),
            "models": str(models),
        },
    )
    return CodexPaths(
        project_root=project_root,
        data_dir=data,
        models_dir=models,
        extensions_dir=extensions,
        extensions_builtin_dir=extensions_builtin,
        outputs_dir=outputs,
        configs_dir=configs,
        default_config=default_cfg,
        default_checkpoint=default_ckpt,
    )
