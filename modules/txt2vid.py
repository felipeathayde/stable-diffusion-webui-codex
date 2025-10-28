"""Legacy txt2vid wrapper removed in Codex backend."""

from __future__ import annotations

raise RuntimeError(
    "modules.txt2vid has been removed. Call apps.backend.use_cases.txt2vid.run_txt2vid"
    " through the Codex backend instead."
)

