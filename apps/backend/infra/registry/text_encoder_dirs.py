"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Directory inventory for vendored text encoders in the Hugging Face cache.
Lists `{org}/{repo}/{subdir}` directories such as `text_encoder`, `text_encoder_2`, `t5`, and `clip` under `apps/backend/huggingface`.

Symbols (top-level; keep in sync; no ghosts):
- `list_text_encoder_dirs` (function): Returns a sorted list of `{name,path}` entries for vendored text encoder directories.
"""

from __future__ import annotations

import os
from typing import List, Dict

from .base import _iter_dirs


def list_text_encoder_dirs(vendored_hf_root: str = "apps/backend/huggingface") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not os.path.isdir(vendored_hf_root):
        return out
    try:
        for org in _iter_dirs(vendored_hf_root):
            org_name = os.path.basename(org)
            for repo in _iter_dirs(org):
                repo_name = os.path.basename(repo)
                for sub in ("text_encoder", "text_encoder_2", "t5", "clip"):
                    p = os.path.join(repo, sub)
                    if os.path.isdir(p):
                        out.append({
                            "name": f"{org_name}/{repo_name}/{sub}",
                            "path": p,
                        })
    except Exception:
        return out
    return sorted(out, key=lambda d: d["name"].lower())


__all__ = ["list_text_encoder_dirs"]
