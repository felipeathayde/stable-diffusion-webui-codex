"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Inventory and metadata for text encoders in vendored Hugging Face trees.
Lists which repos contain encoder/tokenizer subfolders and can extract lightweight `config.json` fields (hidden size, vocab size, projections) for
diagnostics and documentation.

Symbols (top-level; keep in sync; no ghosts):
- `_find_text_encoder_roots` (function): Returns present encoder/tokenizer subfolders under a repo directory.
- `list_text_encoders` (function): Returns a mapping `{repo_name: [subdirs...]}` for vendored HF repos.
- `_read_json` (function): Best-effort JSON reader used for `config.json` extraction.
- `describe_text_encoders` (function): Returns richer metadata entries for vendored HF repos/subdirs.
"""

from __future__ import annotations

import os
from typing import Dict, List
import json

from .base import _iter_dirs


def _find_text_encoder_roots(repo_dir: str) -> List[str]:
    hits: List[str] = []
    for sub in ("text_encoder", "text_encoder_2", "t5", "clip", "tokenizer", "tokenizer_2"):
        p = os.path.join(repo_dir, sub)
        if os.path.isdir(p):
            hits.append(sub)
    return hits


def list_text_encoders(vendored_hf_root: str = "apps/backend/huggingface") -> Dict[str, List[str]]:
    """Return a mapping of available text encoders in vendored Hugging Face trees.

    Keys: repo names; values: available subfolders (text_encoder, text_encoder_2, t5, ...).
    This is used for diagnostics and documentation, not for user-configurable overrides.
    """
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(vendored_hf_root):
        return out
    try:
        for org in _iter_dirs(vendored_hf_root):
            for repo in _iter_dirs(org):
                name = os.path.basename(repo)
                subs = _find_text_encoder_roots(repo)
                if subs:
                    out[name] = sorted(subs)
    except Exception:
        return out
    return out


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def describe_text_encoders(vendored_hf_root: str = "apps/backend/huggingface") -> Dict[str, List[dict]]:
    """Rich metadata for text encoders in vendored repos.

    Each item: {subdir, type (clip|t5|unknown), hidden_size?, projection?, vocab_size?}
    """
    mapping: Dict[str, List[dict]] = {}
    if not os.path.isdir(vendored_hf_root):
        return mapping
    for org in _iter_dirs(vendored_hf_root):
        for repo in _iter_dirs(org):
            name = os.path.basename(repo)
            infos: List[dict] = []
            for sub in ("text_encoder", "text_encoder_2", "t5", "clip", "tokenizer", "tokenizer_2"):
                p = os.path.join(repo, sub)
                if not os.path.isdir(p):
                    continue
                cfg = _read_json(os.path.join(p, "config.json")) if os.path.isfile(os.path.join(p, "config.json")) else {}
                kind = "t5" if "t5" in sub or cfg.get("model_type") == "t5" else ("clip" if "clip" in sub or cfg.get("model_type", "").startswith("clip") else "unknown")
                infos.append({
                    "subdir": sub,
                    "type": kind,
                    "hidden_size": cfg.get("hidden_size") or cfg.get("d_model"),
                    "projection": bool(cfg.get("text_projection", False) or cfg.get("add_text_projection", False)),
                    "vocab_size": cfg.get("vocab_size"),
                })
            if infos:
                mapping[name] = infos
    return mapping


__all__ = ["list_text_encoders", "describe_text_encoders"]
