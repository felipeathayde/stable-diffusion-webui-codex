from __future__ import annotations

import os
from typing import List, Dict

from .base import _iter_dirs


def list_tokenizers(vendored_hf_root: str = "apps/backend/huggingface") -> List[Dict[str, str]]:
    """Return a list of tokenizer directories under vendored HF cache.

    Each item: { name: 'org/repo/tokenizer[_2]', path: '/abs/path/to/.../tokenizer[_2]' }
    """
    out: List[Dict[str, str]] = []
    if not os.path.isdir(vendored_hf_root):
        return out
    try:
        for org in _iter_dirs(vendored_hf_root):
            org_name = os.path.basename(org)
            for repo in _iter_dirs(org):
                repo_name = os.path.basename(repo)
                for sub in ("tokenizer", "tokenizer_2"):
                    p = os.path.join(repo, sub)
                    if os.path.isdir(p):
                        out.append({
                            "name": f"{org_name}/{repo_name}/{sub}",
                            "path": p,
                        })
    except Exception:
        return out
    return sorted(out, key=lambda d: d["name"].lower())


__all__ = ["list_tokenizers"]
