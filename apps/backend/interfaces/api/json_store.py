"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small JSON load/save helpers for API persistence files.
Used by UI persistence endpoints (tabs/workflows/presets), settings schema, and paths options, with atomic save semantics.

Symbols (top-level; keep in sync; no ghosts):
- `_load_json` (function): Loads JSON from disk (returns `{}` for missing file; raises on invalid/unreadable content).
- `_save_json` (function): Saves JSON atomically (temp+fsync+replace) and raises on write/serialization errors.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _load_json(path: str) -> dict:
    path_obj = Path(path)
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file {path_obj}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read JSON file {path_obj}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load JSON file {path_obj}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"JSON file {path_obj} must contain an object at the top level (got {type(payload).__name__})."
        )
    return payload


def _save_json(path: str, data: dict) -> None:
    if not isinstance(data, dict):
        raise TypeError(f"_save_json expects dict data, got {type(data).__name__}.")
    target = Path(path)
    temp_path: str | None = None
    try:
        os.makedirs(target.parent, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f"{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except OSError as exc:
        raise RuntimeError(f"Failed to write JSON file {target}: {exc}") from exc
    except TypeError as exc:
        raise RuntimeError(f"Failed to serialize JSON for {target}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to save JSON file {target}: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
