"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical path resolution helpers for the launcher.
Normalizes optional path overrides and computes canonical runtime directories (data/models/output/configs) rooted at `CODEX_ROOT`.

Symbols (top-level; keep in sync; no ghosts):
- `LOGGER` (constant): Module logger for launcher path resolution.
- `_normalize_path` (function): Normalizes optional paths and raises explicit errors on failure.
- `CodexPaths` (dataclass): Resolved canonical paths for the runtime.
- `resolve_paths` (function): Resolves launcher paths with strict normalization.
"""

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

    codex_root: Path
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
    codex_root: Path,
    data_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> CodexPaths:
    """Resolve launcher paths with strict normalisation."""
    data = _normalize_path(data_dir, fallback=codex_root / "data")
    models = _normalize_path(models_dir, fallback=codex_root / "models")
    extensions = codex_root / "extensions"
    extensions_builtin = codex_root / "extensions-builtin"
    outputs = codex_root / "output"
    configs = codex_root / "configs"
    default_cfg = configs / "v1-inference.yaml"
    default_ckpt = codex_root / "model.ckpt"

    LOGGER.debug(
        "Resolved runtime paths",
        extra={
            "codex_root": str(codex_root),
            "data": str(data),
            "models": str(models),
        },
    )
    return CodexPaths(
        codex_root=codex_root,
        data_dir=data,
        models_dir=models,
        extensions_dir=extensions,
        extensions_builtin_dir=extensions_builtin,
        outputs_dir=outputs,
        configs_dir=configs,
        default_config=default_cfg,
        default_checkpoint=default_ckpt,
    )
