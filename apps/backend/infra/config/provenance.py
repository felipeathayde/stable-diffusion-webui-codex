from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

CODEX_GENERATED_BY = "stable-diffusion-webui-codex"
CODEX_REPO_URL = "https://github.com/sangoi-exe/stable-diffusion-webui-codex"

_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def best_effort_git_commit(repo_root: Path) -> str | None:
    """Return the current git commit sha (40 hex) when .git is available.

    Best-effort: returns None in packaged deploys that omit `.git/`.
    """
    head = repo_root / ".git" / "HEAD"
    if not head.is_file():
        return None
    try:
        raw = head.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    if raw.startswith("ref:"):
        ref = raw.split(":", 1)[1].strip()
        if not ref:
            return None
        ref_path = repo_root / ".git" / ref
        if ref_path.is_file():
            try:
                commit = ref_path.read_text(encoding="utf-8").strip()
            except Exception:
                commit = ""
            return commit or None
        packed = repo_root / ".git" / "packed-refs"
        if packed.is_file():
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    commit, name = line.split(" ", 1)
                    if name.strip() == ref:
                        return commit.strip() or None
            except Exception:
                return None
        return None

    # Detached head case: HEAD contains the commit sha directly.
    if _FULL_SHA_RE.fullmatch(raw):
        return raw.lower()
    return None


@lru_cache(maxsize=4)
def _cached_generation_provenance(repo_root: Path) -> tuple[tuple[str, str], ...]:
    provenance: dict[str, str] = {
        "generated_by": CODEX_GENERATED_BY,
        "repo": CODEX_REPO_URL,
    }
    commit = best_effort_git_commit(repo_root)
    if commit:
        provenance["commit"] = commit
    return tuple(provenance.items())


def generation_provenance(repo_root: Path) -> dict[str, str]:
    """Return stable, non-sensitive provenance fields for generation outputs."""
    return dict(_cached_generation_provenance(repo_root))


__all__ = [
    "CODEX_GENERATED_BY",
    "CODEX_REPO_URL",
    "best_effort_git_commit",
    "generation_provenance",
]
