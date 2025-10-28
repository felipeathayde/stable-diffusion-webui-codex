"""Legacy img2vid wrapper removed in Codex backend."""

from __future__ import annotations

raise RuntimeError(
    "modules.img2vid has been removed. Use apps.backend.use_cases.img2vid.run_img2vid"
    " with Codex request objects via the backend services."
)

