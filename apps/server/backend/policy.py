from __future__ import annotations

"""Runtime policy guard — fails fast on forbidden imports/paths.

Controlled by env vars:
- CODEX_POLICY_STRICT=1 (default): on violation, print error and exit(1)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


FORBIDDEN_MODULE_PREFIXES = (
    "legacy",
    "DEPRECATED",
    "wan_gguf",
    "backend.",  # old façade
)

FORBIDDEN_PATH_FRAGMENTS = (
    "apps/server/backend/engines/video/wan/",
)


def _strict() -> bool:
    return (os.environ.get("CODEX_POLICY_STRICT", "1") or "1").strip() not in ("0", "false", "False")


def enforce_repo_policies() -> None:
    # 1) Loaded modules
    bad: list[str] = []
    for name in tuple(sys.modules.keys()):
        for pref in FORBIDDEN_MODULE_PREFIXES:
            if name.startswith(pref):
                bad.append(f"forbidden module loaded: {name}")
                break
    # 2) Forbidden paths present in active tree
    for frag in FORBIDDEN_PATH_FRAGMENTS:
        if (ROOT / frag).exists():
            bad.append(f"forbidden path exists: {frag}")
    if bad:
        sys.stderr.write("[POLICY] Violations detected:\n" + "\n".join(" - " + b for b in bad) + "\n")
        sys.stderr.flush()
        if _strict():
            os._exit(1)

