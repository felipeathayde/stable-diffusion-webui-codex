"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Public facade for checkpoint and VAE registry operations.
Provides lightweight wrappers for listing assets and resolving checkpoints by name/path/title or SHA identifiers.

Symbols (top-level; keep in sync; no ghosts):
- `list_checkpoints` (function): Return discovered checkpoint records (optionally refresh the registry cache).
- `list_checkpoints_as_dict` (function): Return JSON-friendly checkpoint records.
- `list_vaes` (function): Return discovered VAE records (optionally refresh the registry cache).
- `list_vaes_as_dict` (function): Return JSON-friendly VAE records.
- `find_checkpoint` (function): Resolve a checkpoint record by name/title/filename/stem/path.
- `_HEX_RE` (constant): Regex used to validate hex-only SHA strings.
- `find_checkpoint_by_sha` (function): Resolve a checkpoint record by sha256 (64 hex) or short-hash (10 hex).
- `refresh` (function): Force a registry rescan.
"""

from __future__ import annotations

from pathlib import Path
import re
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


_HEX_RE = re.compile(r"^[0-9a-f]+$")


def find_checkpoint_by_sha(sha256: str) -> Optional[CheckpointRecord]:
    """Resolve a checkpoint record by SHA256/short-hash.

    Accepts the full 64-hex sha256 or the 10-char short hash stored in
    `models/.hashes.json`. Returns `None` when no checkpoint matches.
    """

    if not isinstance(sha256, str):
        return None
    sha = sha256.strip().lower()
    if not sha:
        return None
    if len(sha) not in (10, 64):
        return None
    if _HEX_RE.fullmatch(sha) is None:
        return None

    reg = registry.get_registry()
    for candidate in reg.list_checkpoints(refresh=False):
        cand_sha = (candidate.sha256 or "").strip().lower()
        cand_short = (candidate.short_hash or "").strip().lower()
        if sha == cand_sha or sha == cand_short:
            return candidate
    return None


def refresh() -> None:
    registry.refresh()


__all__ = [
    "find_checkpoint",
    "find_checkpoint_by_sha",
    "list_checkpoints",
    "list_checkpoints_as_dict",
    "list_vaes",
    "list_vaes_as_dict",
    "refresh",
]
