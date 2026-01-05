"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vendored Hugging Face tree scanners (org/repo inventory).
Provides a shared, stable directory-walk for `apps/backend/huggingface/{org}/{repo}` so tokenizers and metadata endpoints don’t drift on sorting or traversal semantics.

Symbols (top-level; keep in sync; no ghosts):
- `iter_vendored_hf_repos` (function): Yields `(org, repo, path)` tuples for vendored HF repos under a given root.
"""

from __future__ import annotations

import os
from typing import Iterator, Tuple


def _iter_dirs_sorted(root: str) -> Iterator[str]:
    try:
        names = sorted(os.listdir(root), key=lambda s: s.lower())
    except Exception:
        return
    for name in names:
        full = os.path.join(root, name)
        if os.path.isdir(full):
            yield full


def iter_vendored_hf_repos(vendored_hf_root: str) -> Iterator[Tuple[str, str, str]]:
    """Yield `(org, repo, repo_path)` for `{vendored_hf_root}/{org}/{repo}`."""
    if not vendored_hf_root or not os.path.isdir(vendored_hf_root):
        return
    for org_dir in _iter_dirs_sorted(vendored_hf_root):
        org = os.path.basename(org_dir)
        for repo_dir in _iter_dirs_sorted(org_dir):
            repo = os.path.basename(repo_dir)
            yield org, repo, repo_dir


__all__ = ["iter_vendored_hf_repos"]

