"""Legacy state shim removed in Codex backend."""

raise RuntimeError(
    "modules.shared_state is obsolete. Use apps.backend.core.state.BackendState instead."
)
