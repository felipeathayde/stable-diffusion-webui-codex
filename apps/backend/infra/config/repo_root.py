"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Repo root resolution helpers (CODEX_ROOT-based).
Provides a single source of truth for locating the repository on disk and constructing repo-relative filesystem paths.

Symbols (top-level; keep in sync; no ghosts):
- `get_repo_root` (function): Resolves and validates the repo root path from `CODEX_ROOT` (fails fast when invalid).
- `repo_path` (function): Convenience wrapper for `get_repo_root().joinpath(...)`.
- `repo_scratch_dir` (function): Returns the repo-local scratch directory root (`CODEX_ROOT/.tmp`).
- `repo_scratch_path` (function): Convenience wrapper for `repo_scratch_dir().joinpath(...)`.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repo root directory.

    CODEX_ROOT must be set by the launcher (run-webui.{bat,sh}).
    """
    env = (os.environ.get("CODEX_ROOT") or "").strip()
    if not env:
        raise EnvironmentError("CODEX_ROOT not set. Launch via run-webui.{bat,sh} or set CODEX_ROOT to the repo root.")

    try:
        root = Path(env).expanduser().resolve(strict=True)
    except Exception as exc:
        raise EnvironmentError(f"CODEX_ROOT is invalid: {env!r}: {exc}") from exc

    if not root.is_dir():
        raise EnvironmentError(f"CODEX_ROOT must point to a directory; got: {root}")

    if not (root / "apps").is_dir():
        raise EnvironmentError(f"CODEX_ROOT does not look like the repo root (missing apps/): {root}")

    return root


def repo_path(*parts: str) -> Path:
    return get_repo_root().joinpath(*parts)

def repo_scratch_dir() -> Path:
    """Return the repo-local scratch directory root ('.tmp/')."""
    return repo_path(".tmp")


def repo_scratch_path(*parts: str) -> Path:
    """Return a path under the repo-local scratch directory ('.tmp/')."""
    return repo_scratch_dir().joinpath(*parts)


__all__ = ["get_repo_root", "repo_path", "repo_scratch_dir", "repo_scratch_path"]
