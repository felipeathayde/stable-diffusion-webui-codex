"""Legacy progress tracking removed in Codex backend."""

raise RuntimeError(
    "modules.progress has been removed. Use apps.backend.core.progress_tracker and"
    " apps.backend.core.state for progress reporting."
)
