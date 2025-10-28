"""Legacy shared state wrapper removed in Codex backend."""

raise RuntimeError(
    "modules.shared has been removed from Codex backend. Use apps.backend.core.state"
    " and apps.backend.codex.options for shared state and configuration."
)
