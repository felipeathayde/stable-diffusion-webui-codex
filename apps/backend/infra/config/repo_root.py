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


__all__ = ["get_repo_root", "repo_path"]
